"""Train the small neural policy, freeze it, then run the eight-condition study.

Usage: python run_study.py --data-dir /path/to/data/raw --routes 24
"""
import argparse
import sys
from pathlib import Path
if Path(__file__).parent.parent.name == "training":
    sys.path.insert(0,str(Path(__file__).parents[2]/"website"/"routecheck"))
import copy
import gzip
import hashlib
import json
import time
import shutil
from evidence import save_trial, model_fingerprint
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from case_data import catalog,find_route,folder_signature,iter_routes,load_case_route
from experiments import CONDITIONS,STAGES,experiment
from neural_policy import MLP,edge_features

from project_paths import APP_DIR, MODEL_DIR, RESULTS_DIR
ROOT=APP_DIR


def examples(routes,seed=7):
    rng=np.random.default_rng(seed);px=[];py=[];tx=[];ty=[]
    for data in routes:
        nodes={s['stop_id']:s for s in data['stops']};nodes['__DEPOT__']=data['depot']
        edges={(e['from_id'],e['to_id']):e['travel_min'] for e in data['travel_matrix']}
        remaining={s['stop_id'] for s in data['stops']};current='__DEPOT__'
        for target in data['actual_order']:
            negatives=sorted(remaining-{target})
            take=rng.choice(negatives,min(4,len(negatives)),replace=False) if negatives else []
            for candidate,label in [(target,1)]+[(str(x),0) for x in take]:
                px.append(edge_features(nodes[current],nodes[candidate],data['depot'],edges[current,candidate]));py.append(label)
            remaining.remove(target);current=target
        sampled=rng.choice(len(data['travel_matrix']),min(1000,len(data['travel_matrix'])),replace=False)
        for index in sampled:
            edge=data['travel_matrix'][int(index)];a,b=edge['from_id'],edge['to_id']
            tx.append(edge_features(nodes[a],nodes[b],data['depot']));ty.append(np.log1p(edge['travel_min']))
    return np.array(px),np.array(py),np.array(tx),np.array(ty)


def train_and_run(folder,route_count=24):
    started=time.perf_counter();out=RESULTS_DIR;out.mkdir(exist_ok=True)
    model_dir=MODEL_DIR;model_dir.mkdir(exist_ok=True)
    signature=folder_signature(folder);entries=catalog(folder)
    # Source order is explicit; use full complete routes, never random stop splits.
    selected=[];excluded=[]
    for entry in entries:
        try:data=load_case_route(folder,entry['route_id'])
        except (ValueError,KeyError,TypeError) as error:
            excluded.append(dict(route_id=entry['route_id'],reason=str(error)));continue
        if data['issues'] or not data['actual_order']:
            excluded.append(dict(route_id=entry['route_id'],reason='Unresolved source-quality flag or missing actual sequence'));continue
        selected.append(data)
        if len(selected)>=route_count:break
    if len(selected)<12:raise ValueError('At least 12 complete routes are needed')
    selected.sort(key=lambda d:(d['route_date'],d['route_id']))
    dates=sorted({d['route_date'] for d in selected})
    if len(dates)<5:raise ValueError('Need at least five distinct route dates for chronological splits')
    first=dates[max(0,int(len(dates)*.6)-1)];second=dates[max(1,int(len(dates)*.8)-1)]
    train=[d for d in selected if d['route_date']<=first]
    val=[d for d in selected if first<d['route_date']<=second]
    test=[d for d in selected if d['route_date']>second]
    if not min(len(train),len(val),len(test)):raise ValueError('Empty chronological split')
    print(f'Full routes: train={len(train)}, validation={len(val)}, test={len(test)}',flush=True)
    x,y,tx,ty=examples(train);vx,vy,vtx,vty=examples(val,19)
    policy=MLP(7,seed=7);travel=MLP(6,seed=19)
    policy_history=policy.fit(x,y,(vx,vy));travel_history=travel.fit(tx,ty,(vtx,vty),task='regression')
    # Calibrate the outlier threshold on clean validation edges only.
    threshold=max(float(np.quantile(np.abs(travel.predict(vtx)-vty),.99)),float(np.log(3)))
    np.savez_compressed(model_dir/'neural_policy.npz',**policy.state('policy_'),**travel.state('travel_'),threshold=np.array(threshold))
    training_seconds=time.perf_counter()-started
    metadata=dict(architecture='7 → 24 tanh → 1 edge-choice logit; 6 → 24 tanh → 1 log travel-minute estimate',
                  method='Supervised neural constructive policy, driver imitation with negative candidate sampling and masked greedy decoding. No reinforcement learning or pointer-network claim.',
                  seed=7,policy_examples=len(y),travel_examples=len(ty),threshold=threshold,
                  policy_history=policy_history,travel_history=travel_history,training_seconds=training_seconds,
                  train_routes=[d['route_id'] for d in train],validation_routes=[d['route_id'] for d in val],test_routes=[d['route_id'] for d in test])
    (model_dir/'model_metadata.json').write_text(json.dumps(metadata,indent=2))
    route_rows=[dict(route_id=d['route_id'],date=d['route_date'],station=d['station'],stops=len(d['stops']),packages=d['package_count'],dataset_hash=d['dataset_hash'],split='train' if d in train else 'validation' if d in val else 'test') for d in selected]
    pd.DataFrame(route_rows).to_csv(out/'selected_routes.csv',index=False)
    pd.DataFrame([dict(network='policy',**r) for r in policy_history]+[dict(network='travel',**r) for r in travel_history]).to_csv(out/'training_history.csv',index=False)
    results=[];timestamp=datetime.now(timezone.utc).isoformat()
    study_id=datetime.now(timezone.utc).strftime('study_%Y%m%dT%H%M%S_%fZ')
    study_dir=out/'studies'/study_id
    trial_dir=study_dir/'trials';trial_dir.mkdir(parents=True)
    weight_hash=model_fingerprint()
    with gzip.open(out/'modification_log.jsonl.gz','wt',encoding='utf-8') as audit:
        for index,data in enumerate(test):
            for seed in (7,19,43):
                for condition in CONDITIONS:
                    for stage in STAGES:
                        for enabled in (False,True):
                            row,prediction,changes,metrics=experiment(data,condition,stage,seed,enabled,(policy,travel,threshold))
                            row['created_at_utc']=timestamp
                            row['study_id']=study_id;row['trial_id']=f'trial_{len(results)+1:04d}'
                            row['evidence_file']=str(Path('studies')/study_id/'trials'/(row['trial_id']+'.json'))
                            save_trial(trial_dir,row['trial_id'],row,prediction,metrics,data['dataset_hash'],weight_hash,'../modification_log.jsonl.gz')
                            results.append(row)
                            if not enabled:
                                for change in changes:
                                    audit.write(json.dumps(dict(route_id=data['route_id'],condition=condition,stage_pct=stage,seed=seed,simulated_stage=STAGES.index(stage),logged_at_utc=timestamp,**change))+'\n')
            print(f'Evaluated held-out route {index+1}/{len(test)}',flush=True)
    frame=pd.DataFrame(results)
    clean=frame[(frame.condition=='Reference') & (~frame.countermeasures)].groupby('route_id').duration_min.first().to_dict()
    frame['relative_to_clean_pct']=[100*(v/clean[r]-1) for r,v in zip(frame.route_id,frame.duration_min)]
    frame.to_csv(out/'results.csv',index=False)
    means=frame.groupby(['condition','stage_pct','countermeasures'],as_index=False).agg(mean_duration_min=('duration_min','mean'),std_duration_min=('duration_min','std'),mean_relative_pct=('relative_to_clean_pct','mean'),mean_runtime_ms=('runtime_seconds',lambda s:1000*s.mean()),mean_repairs=('repairs','mean'),runs=('route_id','size'),mean_on_time_pct=('on_time_pct','mean'))
    means.to_csv(out/'summary.csv',index=False)
    pairs=frame.pivot(index=['route_id','condition','stage_pct','seed'],columns='countermeasures',values='duration_min').reset_index().rename(columns={False:'without_min',True:'with_min'})
    pairs['recovery_min']=pairs.without_min-pairs.with_min
    pairs.to_csv(out/'paired_comparisons.csv',index=False)
    rng=np.random.default_rng(91);intervals=[]
    for (condition,stage),group in pairs.groupby(['condition','stage_pct']):
        route_effects=group.groupby('route_id').recovery_min.mean().to_numpy()
        samples=np.array([rng.choice(route_effects,len(route_effects),replace=True).mean() for _ in range(1000)])
        intervals.append(dict(condition=condition,stage_pct=int(stage),mean_recovery_min=float(route_effects.mean()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),independent_routes=len(route_effects)))
    pd.DataFrame(intervals).to_csv(out/'uncertainty.csv',index=False)
    manifest=dict(study_id=study_id,neural_weights_sha256=weight_hash,individual_trial_files=len(frame),created_at_utc=timestamp,source_url='https://registry.opendata.aws/amazon-last-mile-challenges/',schema_url='https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md',
                  source_version='2021 release; local model-build files with 6112 routes from 2018',retrieval_date='Unknown: user supplied local files; inspected 2026-10-20',source_files=[dict(name=n,bytes=size,mtime_ns=mtime) for n,size,mtime in signature],
                  rights='AWS registry lists CC BY-NC 4.0. Raw data remain in the user folder and are excluded from the application archive. Review terms before public redistribution.',
                  subset_rule=f'First {route_count} complete routes in route_data.json source order with actual sequences and no unresolved quality flags. Preserve all stops and package records for each selected route.',
                  selected_route_count=len(selected),source_route_count=len(entries),split_rule=f'Chronological groups by full route date: train <= {first}; validation <= {second}; test later. No route or date overlaps.',
                  split_counts=dict(train=len(train),validation=len(val),test=len(test)),date_range=[selected[0]['route_date'],selected[-1]['route_date']],
                  exclusion_log=excluded,seeds=[7,19,43],simulated_stages=list(STAGES),conditions=list(CONDITIONS),experiment_rows=len(frame),
                  degradation_stage='Frozen-model inference inputs only. No retraining during progression. Stale observations simulate a frozen feed missing a reference congestion increase. No observed Amazon incident is claimed.',
                  edge_volume='Total directed edge count is fixed per route. The legacy/fast-feed contribution share grows by the stage percentage; combined sets can overlap.',
                  countermeasure='Flag age >30 days, missing values or validation-calibrated log residuals; replace using the training-only neural travel estimator. No current reference values or defect labels enter repair.',
                  evaluation='Replay recommended order against untouched supplied travel-time estimates and planned package service durations, waiting for package-window intersections and returning to depot. Primary metric: estimated reference replay minutes, lower better. Historical estimates are not fresh verified outcomes or optimal routes.',
                  human_effort='No participant study conducted. Runtime is measured separately. Human review duration is recorded only when a person submits an app review.',
                  total_wall_seconds=time.perf_counter()-started,compute_cost='Local CPU only; no paid AI or routing API calls. Electricity and human verification costs not measured.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    worst=pairs.sort_values('recovery_min').iloc[0]
    final=means[means.stage_pct==50]
    lines=['# RouteCheck experiment report','',f'{len(selected)} full routes: {len(train)} training, {len(val)} validation, {len(test)} held-out test routes. {len(frame)} measured runs across three corruption seeds.','',
           'The neural policy learns driver next-stop preferences. The travel estimator learns reference edge times. Both networks freeze before test experiments.','',
           '## Stage 50% results','',final.to_markdown(index=False) if False else final.to_csv(index=False),'',
           '## Countermeasure failure / boundary','',f'The worst paired case was {worst.route_id}, {worst.condition}, stage {int(worst.stage_pct)}%, seed {int(worst.seed)}. Without repair: {worst.without_min:.2f} min; with repair: {worst.with_min:.2f} min; recovery: {worst.recovery_min:.2f} min. Negative recovery means repair made this route worse.','',
           'Even on nominal inputs the detector may replace legitimate unusual edges. Inspect false-positive and legitimate-rare-edge columns in results.csv. A median/learned replacement can remove useful local information.','',
           '## Limits and decision checkpoint','',
           f'Only {len(test)} independent held-out routes support the conclusion. Repeats and stops do not increase that independent count. Bootstrap intervals resample route-level paired averages, not individual edges. Results are exploratory and cannot establish universal robustness. Staleness and feed quality are controlled simulations, not measured Amazon failures.',
           'A data-quality lead should inspect false positives and update the source records. A model owner should expand the held-out station/date evaluation before proposing any offline pilot. Stop using a recommendation if locations, windows or travel inputs remain unresolved. An operator must approve any corrected or overridden route; this app never dispatches vehicles.','',
           'The untouched travel matrix supplies a consistent offline evaluation reference. It does not supply fresh ground truth, road distance, observed arrival times or an optimal route. Planned service times are summed per stop. Conflicting package windows are excluded and logged before the study.','',
           '## Reproduction','',f'Run `python run_study.py --data-dir /path/to/data/raw --routes {route_count}`. See manifest.json, selected_routes.csv, model_metadata.json, training_history.csv and modification_log.jsonl.gz for provenance.']
    (out/'REPORT.md').write_text('\n'.join(lines))
    for name in ['results.csv','summary.csv','paired_comparisons.csv','selected_routes.csv','training_history.csv','uncertainty.csv','manifest.json','REPORT.md','modification_log.jsonl.gz']:
        shutil.copy2(out/name,study_dir/name)
    shutil.copy2(model_dir/'neural_policy.npz',study_dir/'neural_policy.npz')
    shutil.copy2(model_dir/'model_metadata.json',study_dir/'model_metadata.json')
    print(f'Completed {len(frame)} measured runs in {time.perf_counter()-started:.1f}s',flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);parser.add_argument('--routes',type=int,default=24)
    args=parser.parse_args()
    if not 12<=args.routes<=100:parser.error('--routes must be between 12 and 100')
    train_and_run(args.data_dir,args.routes)
