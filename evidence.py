"""Write individual trial evidence files plus a plain-language reading guide."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from project_paths import MODEL_DIR, RESULTS_DIR


def model_fingerprint():
    path=MODEL_DIR/'neural_policy.npz'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'untrained'


def save_trial(directory,trial_id,row,prediction,metrics,reference_hash,model_hash,log_file=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    record=dict(trial_id=trial_id,recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                settings=row,reference_dataset_sha256=reference_hash,neural_weights_sha256=model_hash,
                prediction=dict(order=prediction['order'],planned_duration_min=prediction['planned_duration_min'],
                                model_name=prediction['model_name'],flagged_edges=prediction.get('flagged_edges',[])),
                reference_replay=metrics,modification_log=log_file,
                interpretation='Defects and feed ages are simulated. Replay uses the untouched historical travel estimates and planned service durations. It is not an optimality proof or observed delivery time.')
    (directory/(trial_id+'.json')).write_text(json.dumps(record,indent=2))
    flag='on' if row['countermeasures'] else 'off'
    lines=[f'# {trial_id}','',f'Route: {row["route_id"]}',f'Condition: {row["condition"]}',
           f'Simulated degraded share: {row["stage_pct"]}%',f'Random seed: {row["seed"]}',f'Countermeasure: {flag}','',
           f'Reference replay duration: {row["duration_min"]} minutes.',f'Stops scheduled: {row["stops_scheduled"]} of {row["total_stops"]}.',
           f'Compute time: {row["runtime_seconds"]:.6f} seconds. Human review time: not measured.',
           f'Changed directed edges: {row["changed_edges"]} of {row["total_edges"]}. Mean simulated edge age: {row["mean_age_days"]:.2f} days.',
           f'Repairs: {row["repairs"]}. True positives: {row["tp"]}. False positives: {row["fp"]}. Missed defects: {row["fn"]}. True negatives: {row["tn"]}.','',
           'Read the JSON beside this file for the full order, replay arrivals, model fingerprint and flagged edges. Compare against the matching route, condition, stage and seed with the other countermeasure setting.',
           'The modification log records each original and changed value. Unprotected runs do not activate a detector, so their missed-defect count is not an evaluation of the protected detector.',
           'Lower replay duration is the primary objective. Negative recovery in paired_comparisons.csv means that repair made the result worse. Historical replay estimates are not fresh ground truth.']
    labels=('Route:', 'Condition:', 'Simulated degraded share:', 'Random seed:', 'Countermeasure:')
    lines=['- '+line if line.startswith(labels) else line for line in lines]
    (directory/(trial_id+'.md')).write_text('\n'.join(lines))
    return record


def save_live_comparison(dataset,result):
    """Persist both actual planner calls, even if the user never saves history."""
    import os
    directory=Path(os.environ.get('ROUTECHECK_EVIDENCE_DIR',str(RESULTS_DIR/'interactive_trials')))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    paths=[]
    for suffix,enabled,key,pred_key in [('before',False,'before','before_prediction'),('current',result['countermeasures'],'after','after_prediction')]:
        pred=result[pred_key];metric=result[key]
        row=dict(route_id=dataset['route_id'],condition='Imported model orders (reference replay only)' if result.get('external_results') else 'Source input (no injected defects)',stage_pct=0,seed=None,
                 countermeasures=enabled,duration_min=metric['duration_min'],stops_scheduled=metric['completed'],
                 total_stops=result['total_stops'],runtime_seconds=result.get('before_runtime_seconds' if key=='before' else 'after_runtime_seconds',0),
                 runtime_note='When countermeasures are off, current is a copy of before, so current copy time is not a second inference.',
                 changed_edges=0,total_edges=len(dataset['travel_matrix']),mean_age_days=0,
                 repairs=pred.get('repair_count',0),tp=None,fp=None,fn=None,tn=None)
        trial_id=f'interactive_{stamp}_{suffix}'
        fingerprint=model_fingerprint() if pred['model_name'].startswith('Neural') else 'External-or-baseline model; no neural weight claim'
        record=save_trial(directory,trial_id,row,pred,metric,dataset['dataset_hash'],fingerprint)
        if result.get('external_results'):
            record['neural_weights_sha256'] = 'Not verified; see imported checkpoint metadata'
            record['settings']['runtime_seconds'] = None
            record['settings']['runtime_note'] = 'External inference runtime was not measured by RouteCheck.'
            record['interpretation']=result['provenance_note']
            record['import_sha256']=result['import_sha256']
            record['imported_document']=result['imported_document']
            (directory/(trial_id+'.json')).write_text(json.dumps(record,indent=2))
            (directory/(trial_id+'.md')).write_text(
                f"# {trial_id}\n\nRoute: {dataset['route_id']}\n\nModel: {pred['model_name']}\n\n"
                f"Reference replay: {metric['duration_min']} minutes; {metric['completed']} stops.\n\n"
                + result['provenance_note'] + '\n')
        paths.append(str(directory/(trial_id+'.json')))
    return paths
