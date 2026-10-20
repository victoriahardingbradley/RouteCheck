# Reading the trial evidence

Start with `REPORT.md` for the study's findings and limitations. Open `results.csv` for the index of every measured trial.

## A real paired example

The following files belong to the same held-out route, simulated condition, stage and seed:

- Without the fix: `trial_0047`
- With the fix: `trial_0048`
- Route: `RouteID_001b4ee3-c4f2-467f-932b-c85524d1021f`
- Condition: Combined
- Simulated stage: 50% per defect type
- Seed: 7

The reference replay takes 558.1 minutes without the fix and 503.6 minutes with it. Paired recovery is 54.5 minutes. Positive recovery means shorter reference replay; negative means the intervention made this route worse. Check the on-time outcome too because duration and time-window adherence can trade off.

Use the `evidence_file` column to locate each JSON file. The matching `.md` explains it in plain language. The JSON contains the order, estimated reference arrivals, altered-input settings, model/reference fingerprints and flagged edges.

## What to compare

Hold route ID, condition, stage and seed fixed. Change only the `countermeasures` value. These two trials share the same corrupted observations and untouched reference outcomes.

`paired_comparisons.csv` calculates recovery for every pair. `uncertainty.csv` reports bootstrap intervals over route averages. The independent test count is four routes; multiple stops, seeds and stages are repeated observations, not hundreds of independent cases.

## How the modification log works

`modification_log.jsonl.gz` is compressed text, one JSON object per changed edge. Filter it by route ID, condition, stage and seed. Each line records the original/modified travel minutes, simulated feed age, source feed and defect type. Paired with/without trials deliberately share a single modification log. It is evaluation evidence, not a model input.

## Human effort and runtime

`runtime_seconds` measures the planner and its filter for that trial. Logging and evaluation overhead are outside that per-call number. Whole-study wall time appears in the manifest. Human review time stays blank until a person reports it in the application. A fast model call does not imply that reviewing the output is easy.

## What the evidence does not establish

It does not prove an optimal route, fresh travel conditions, deployment readiness or universal robustness. Some ordinary and stale-only cases get worse with repair. Those failures remain in the files. The data is historical and anonymized; defects are controlled simulations, not observed Amazon incidents.

No slide deck was created. `fallback_demo.gif` is a silent recording assembled from actual app-screen captures, for use if the live demo cannot run.
