"""Small neural constructive routing policy, trained by imitation of driver order.

This is a neural combinatorial optimization prototype, not a pointer-network
reproduction or a reinforcement-learning model. It uses a learned candidate
scorer with a masked greedy decoder. A second MLP estimates travel minutes.
"""
import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from data_input import inspect_stops
from model_integration import distance

from project_paths import MODEL_DIR
MODEL_PATH = MODEL_DIR/'neural_policy.npz'


class MLP:
    def __init__(self, inputs, hidden=24, seed=7):
        rng=np.random.default_rng(seed)
        self.w1=rng.normal(0,np.sqrt(2/inputs),(inputs,hidden))
        self.b1=np.zeros(hidden)
        self.w2=rng.normal(0,.1,(hidden,1));self.b2=np.zeros(1)
        self.mean=np.zeros(inputs);self.std=np.ones(inputs)

    def predict(self,x):
        z=(np.asarray(x,dtype=float)-self.mean)/self.std
        return (np.tanh(z@self.w1+self.b1)@self.w2+self.b2).ravel()

    def fit(self,x,y,validation,task='classification',epochs=45,seed=7):
        x=np.asarray(x,float);y=np.asarray(y,float).reshape(-1,1)
        self.mean=x.mean(0);self.std=np.maximum(x.std(0),.01)
        z=(x-self.mean)/self.std
        rng=np.random.default_rng(seed)
        params=[self.w1,self.b1,self.w2,self.b2]
        moments=[np.zeros_like(p) for p in params];velocities=[np.zeros_like(p) for p in params]
        best_loss=float('inf');best=None;history=[];step=0;patience=0
        for epoch in range(epochs):
            indices=rng.permutation(len(x))
            for start in range(0,len(x),256):
                ix=indices[start:start+256];a=z[ix];target=y[ix]
                h=np.tanh(a@self.w1+self.b1);out=h@self.w2+self.b2
                if task=='classification':
                    sigmoid=1/(1+np.exp(-np.clip(out,-30,30)));delta=(sigmoid-target)/len(ix)
                else: delta=2*(out-target)/len(ix)
                dh=(delta@self.w2.T)*(1-h*h)
                grads=[a.T@dh+1e-4*self.w1,dh.sum(0),h.T@delta+1e-4*self.w2,delta.sum(0)]
                step+=1
                for p,g,m,v in zip(params,grads,moments,velocities):
                    np.clip(g,-5,5,out=g)
                    m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
                    p-=.005*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            vx,vy=validation;pred=self.predict(vx)
            if task=='classification':loss=float(np.mean(np.logaddexp(0,pred)-np.asarray(vy)*pred))
            else: loss=float(np.mean((pred-np.asarray(vy))**2))
            history.append(dict(epoch=epoch+1,validation_loss=loss))
            if loss<best_loss-1e-5:
                best_loss=loss;best=[p.copy() for p in params];patience=0
            else:patience+=1
            if patience>=8:break
        for p,chosen in zip(params,best):p[:]=chosen
        return history

    def state(self,prefix):return {prefix+k:getattr(self,k) for k in ['w1','b1','w2','b2','mean','std']}

    @classmethod
    def restore(cls,archive,prefix):
        obj=cls(archive[prefix+'w1'].shape[0])
        for k in ['w1','b1','w2','b2','mean','std']:setattr(obj,k,archive[prefix+k].copy())
        return obj


def edge_features(a,b,depot,minutes=None):
    same=bool(a.get('zone_id') and a.get('zone_id')==b.get('zone_id'))
    prefix=bool(a.get('zone_id') and b.get('zone_id') and a['zone_id'].split('-')[0]==b['zone_id'].split('-')[0])
    spatial=[np.log1p(distance(a,b)),float(same),float(prefix),(b['lat']-a['lat'])*10,(b['lon']-a['lon'])*10,np.log1p(distance(depot,b))]
    if minutes is None:return spatial
    return [np.log1p(max(0,minutes))]+spatial


@lru_cache(maxsize=2)
def load_models(path,modified):
    with np.load(path,allow_pickle=False) as archive:
        return MLP.restore(archive,'policy_'),MLP.restore(archive,'travel_'),float(archive['threshold'])


def models():
    if not MODEL_PATH.is_file():raise ValueError('Train the neural model first: python run_study.py --data-dir PATH')
    return load_models(str(MODEL_PATH),MODEL_PATH.stat().st_mtime_ns)


def repair_payload(payload,travel_net,threshold):
    data=copy.deepcopy(payload);stops={s['stop_id']:s for s in data['stops']};stops['__DEPOT__']=data['depot']
    edges=data['travel_matrix']
    if not edges:raise ValueError('Neural policy requires a complete travel-time matrix')
    spatial=np.array([edge_features(stops[e['from_id']],stops[e['to_id']],data['depot']) for e in edges])
    estimates=np.maximum(.01,np.expm1(np.clip(travel_net.predict(spatial),-10,10)))
    repairs=[]
    for e,estimated in zip(edges,estimates):
        observed=e.get('travel_min');age=e.get('age_days',0)
        missing=observed is None or not np.isfinite(observed) or observed<0
        outlier=not missing and abs(np.log1p(observed)-np.log1p(estimated))>threshold
        if missing or outlier or age>30:
            repairs.append(dict(from_id=e['from_id'],to_id=e['to_id'],reason='missing' if missing else 'stale' if age>30 else 'outlier',old_min=observed,new_min=float(estimated)))
            e['travel_min']=float(estimated)
    return data,repairs


def construct(payload,policy,travel_net,threshold,enabled):
    if any(s['lat'] is None or s['lon'] is None for s in payload['stops']):
        raise ValueError('Correct missing coordinates before using the neural policy')
    data,repairs=repair_payload(payload,travel_net,threshold) if enabled else (copy.deepcopy(payload),[])
    edges={(e['from_id'],e['to_id']):e for e in data['travel_matrix']}
    if not edges:raise ValueError('Neural policy requires a travel-time matrix')
    unique={s['stop_id']:s for s in data['stops']}
    current,current_id,elapsed=data['depot'],'__DEPOT__',0.
    order=[];issues=inspect_stops(data['stops'])
    # A fixed fallback keeps the unprotected planner runnable on missing edges.
    # It sees only observed values, never the hidden reference matrix.
    observed=[e['travel_min'] for e in edges.values() if e.get('travel_min') is not None and np.isfinite(e['travel_min']) and e['travel_min']>=0]
    fallback=float(np.median(observed)) if observed else 10.
    def travel(a,b):
        t=edges[a,b].get('travel_min')
        return fallback if t is None or not np.isfinite(t) or t<0 else float(t)
    while unique:
        candidates=list(unique.values())
        times=np.array([travel(current_id,s['stop_id']) for s in candidates])
        x=np.array([edge_features(current,s,data['depot'],t) for s,t in zip(candidates,times)])
        logits=policy.predict(x)-.15*times
        for i,s in enumerate(candidates):
            if s['window_end'] is not None:logits[i]-=.05*max(0,elapsed+times[i]-s['window_end'])
        chosen=int(np.argmax(logits));stop=candidates[chosen]
        elapsed=max(elapsed+times[chosen],stop['window_start'])+stop['service_min']
        order.append(stop['stop_id']);current,current_id=stop,stop['stop_id'];del unique[current_id]
    elapsed+=travel(current_id,'__DEPOT__')
    if repairs:
        issues.extend(dict(stop_id=e['to_id'] if e['to_id']!='__DEPOT__' else e['from_id'],reason=f'Travel edge {e["from_id"]} → {e["to_id"]}: {e["reason"]}; learned estimate needs review') for e in repairs)
    return dict(order=order,planned_duration_min=float(elapsed),issues=issues,
                model_name='Neural constructive policy v1 (driver imitation)',repair_count=len(repairs),
                flagged_edges=[[e["from_id"],e["to_id"]] for e in repairs])


def solve(payload,countermeasures):
    return construct(payload,*models(),countermeasures)
