# RouteCheck experiment report

24 full routes: 15 training, 5 validation, 4 held-out test routes. 576 measured runs across three corruption seeds.

The neural policy learns driver next-stop preferences. The travel estimator learns reference edge times. Both networks freeze before test experiments.

## Stage 50% results

condition,stage_pct,countermeasures,mean_duration_min,std_duration_min,mean_relative_pct,mean_runtime_ms,mean_repairs,runs,mean_on_time_pct
Combined,50,False,548.2083333333334,45.63253341707653,16.562242968193704,53.13643757835962,0.0,12,93.35833333333333
Combined,50,True,505.44166666666666,45.19109844635198,7.191504124270935,98.49421524753173,11016.583333333334,12,90.625
Low quality only,50,False,557.0,40.444395049364886,18.483537673811885,51.0682187450584,0.0,12,88.99166666666667
Low quality only,50,True,507.85833333333335,47.466226430117125,7.693868974817476,91.65894108688614,7175.5,12,96.93333333333334
Reference,50,False,472.125,47.17295305575007,0.0,51.508503510073446,0.0,12,96.05
Reference,50,True,476.47499999999997,41.22488491631766,1.0604583130180423,90.30067374987993,7.75,12,96.05
Stale only,50,False,485.2416666666666,46.32198493025518,2.8815038776100383,51.99493058413888,0.0,12,96.61666666666667
Stale only,50,True,488.06666666666666,43.92287317422198,3.502816783521253,94.20481607958209,7424.75,12,96.93333333333334


## Countermeasure failure / boundary

The worst paired case was RouteID_001b4ee3-c4f2-467f-932b-c85524d1021f, Stale only, stage 50%, seed 43. Without repair: 483.60 min; with repair: 505.20 min; recovery: -21.60 min. Negative recovery means repair made this route worse.

Even on nominal inputs the detector may replace legitimate unusual edges. Inspect false-positive and legitimate-rare-edge columns in results.csv. A median/learned replacement can remove useful local information.

## Limits and decision checkpoint

Only 4 independent held-out routes support the conclusion. Repeats and stops do not increase that independent count. Bootstrap intervals resample route-level paired averages, not individual edges. Results are exploratory and cannot establish universal robustness. Staleness and feed quality are controlled simulations, not measured Amazon failures.
A data-quality lead should inspect false positives and update the source records. A model owner should expand the held-out station/date evaluation before proposing any offline pilot. Stop using a recommendation if locations, windows or travel inputs remain unresolved. An operator must approve any corrected or overridden route; this app never dispatches vehicles.

The untouched travel matrix supplies a consistent offline evaluation reference. It does not supply fresh ground truth, road distance, observed arrival times or an optimal route. Planned service times are summed per stop. Conflicting package windows are excluded and logged before the study.

## Reproduction

Run `python run_study.py --data-dir /path/to/data/raw --routes 24`. See manifest.json, selected_routes.csv, model_metadata.json, training_history.csv and modification_log.jsonl.gz for provenance.