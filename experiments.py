"""Controlled inference-time defects and reference replay. No model retraining here."""
import copy
import hashlib
import math
import random
import time

from model_integration import evaluate
from neural_policy import construct

CONDITIONS=('Reference','Stale only','Low quality only','Combined')
STAGES=(0,10,20,30,40,50)


def degrade(reference,condition,stage,seed):
    if condition not in CONDITIONS or stage not in STAGES: raise ValueError('Unknown study condition/stage')
    data=copy.deepcopy(reference)
    # Outcomes and the actual sequence are NEVER part of the observation payload.
    data.pop('actual_order',None)
    route_seed=int(hashlib.sha256(reference['route_id'].encode()).hexdigest()[:8],16)+seed
    rng=random.Random(route_seed)
    positions=list(range(len(data['travel_matrix'])));rng.shuffle(positions)
    count=round(len(positions)*stage/100)
    stale=set(positions[:count]) if condition in ('Stale only','Combined') else set()
    # Independent second ranking permits overlap in the combined condition.
    second=positions.copy();random.Random(route_seed+991).shuffle(second)
    low=set(second[:count]) if condition in ('Low quality only','Combined') else set()
    changes=[]
    for i,edge in enumerate(data['travel_matrix']):
        old=edge['travel_min'];value=old;age=0;types=[]
        if i in stale:
            # Simulated old assumptions miss a congestion increase in the reference world.
            value=old/rng.uniform(1.7,3.0);age=30+stage*3;types.append('stale')
        if i in low:
            value=None if rng.random()<.5 else value*rng.choice([.02,15.])
            types.append('low_quality')
        edge['travel_min']=value;edge['age_days']=age
        edge['source_feed']='fast_feed' if i in low else 'legacy_feed' if i in stale else 'reference_feed'
        if types:changes.append(dict(from_id=edge['from_id'],to_id=edge['to_id'],before_min=old,after_min=value,age_days=age,defects=types,source_feed=edge['source_feed']))
    payload={k:data[k] for k in ('schema_version','route_id','depot','stops','speed_kmh','travel_matrix')}
    return payload,changes


def experiment(reference,condition,stage,seed,enabled,networks,solver=None):
    payload,changes=degrade(reference,condition,stage,seed)
    started=time.perf_counter()
    prediction=solver(payload,enabled) if solver is not None else construct(payload,*networks,enabled)
    runtime=time.perf_counter()-started
    metrics=evaluate(reference,prediction)
    expected={(r['from_id'],r['to_id']) for r in changes}
    predicted={tuple(x) for x in prediction.get('flagged_edges',[])}
    n=len(payload['travel_matrix']);tp=len(expected&predicted);fp=len(predicted-expected);fn=len(expected-predicted)
    clean_times=sorted(e['travel_min'] for e in reference['travel_matrix'])
    rare_cutoff=clean_times[max(0,int(.95*len(clean_times))-1)]
    legitimate_rare={(e['from_id'],e['to_id']) for e in reference['travel_matrix'] if e['travel_min']>=rare_cutoff}-expected
    actual=reference.get('actual_order')
    pairs=set(zip(actual,actual[1:])) if actual else set()
    proposed=set(zip(prediction['order'],prediction['order'][1:]))
    row=dict(route_id=reference['route_id'],date=reference.get('route_date'),station=reference.get('station'),condition=condition,
             stage_pct=stage,seed=seed,countermeasures=enabled,source='case_json_with_simulated_defects',
             total_edges=n,changed_edges=len(expected),stale_edges=sum('stale' in r['defects'] for r in changes),
             low_quality_edges=sum('low_quality' in r['defects'] for r in changes),
             mean_age_days=sum(e['age_days'] for e in payload['travel_matrix'])/n,
             duration_min=metrics['duration_min'],on_time_pct=metrics['on_time_pct'],
             stops_scheduled=metrics['completed'],total_stops=len(reference['stops']),
             runtime_seconds=runtime,human_review_seconds=None,repairs=prediction.get('repair_count',0),
             tp=tp,fp=fp,fn=fn,tn=n-tp-fp-fn,
             precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
             legitimate_rare_edges=len(legitimate_rare),rare_false_positives=len(predicted&legitimate_rare),
             adjacency_match_pct=100*len(pairs&proposed)/len(pairs) if pairs else None,
             outputs_withheld=0,invalid_outputs=int(set(prediction['order'])!={s['stop_id'] for s in reference['stops']}))
    return row,prediction,changes,metrics
