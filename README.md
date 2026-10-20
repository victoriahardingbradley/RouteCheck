# RouteCheck

A Python + Streamlit application for the William & Mary AI Club Case-a-thon 2026. It reads your Amazon routing JSON files, uses a trained neural constructive routing policy, simulates stale and low-quality travel inputs, and tests a learned countermeasure. All trial evidence is saved. No presentation deck is included, as requested.

## Where the Desktop installation lives

```text
caseathon/
  Case-a-thon Prompt.pdf                  existing brief, unchanged
  Run RouteCheck.command                 double-click to start
  setup_routecheck.command               installs Python dependencies if needed
  ROUTECHECK_README.md                    overview and quick start
  website/
    routecheck/
      app.py                             Streamlit interface
      charts.py                          Plotly figures
      case_data.py                       original JSON importer
      data_input.py                      optional CSV validator
      storage.py                         local experiment/review history
      project_paths.py                   paths shared with model code
      requirements*.txt                  dependency versions
      .streamlit/config.toml             theme and local settings
      tests/                             validation and workflow tests
      .venv/                             installed Python 3.12 environment
  training/
    routecheck/
      neural_policy.py                   learned scorer + masked route decoder
      model_integration.py               adapter and reference replay
      experiments.py                     controlled defects and diagnostics
      run_study.py                       repeatable training/evaluation command
      evidence.py                        per-trial JSON and readable summaries
      models/
        neural_policy.npz                trained neural weights
        model_metadata.json              architecture, learning history and split IDs
  data/
    raw/                                 your original JSON files, unchanged
    processed/routecheck/
      routecheck.sqlite3                 saved app runs and human reviews
  results/
    routecheck/
      results.csv                        latest measured trials and evidence paths
      summary.csv                        grouped outcomes
      paired_comparisons.csv              matched with/without differences
      uncertainty.csv                    bootstrap intervals over test routes
      selected_routes.csv                every full route and its split
      training_history.csv               validation loss by training epoch
      manifest.json                      source, units, filters, dates and method
      modification_log.jsonl.gz           exact injected changes
      REPORT.md                          findings and failure cases
      studies/<study-id>/                 preserved study snapshots
        trials/trial_0001.json            full evidence for one trial
        trials/trial_0001.md              explanation of the same trial
        ...                              one pair per trial
      interactive_trials/                automatic evidence from app comparisons
      fallback_demo.gif                  recorded silent app walkthrough
```

Earlier files directly inside `website/`, including the old app and HTML, are preserved. The new app is inside `website/routecheck/`. Generated demo routes have been removed. Old SQLite records labeled `synthetic_demo` are removed along with their linked demo reviews; imported-data history is preserved. Small fictional records inside tests are only test fixtures.

## Run on this Mac

Double-click `Run RouteCheck.command` in `caseathon`. The app opens at http://localhost:8501 (open the link yourself if the browser does not open). Keep the terminal window open. Press Ctrl+C in it to stop.

Terminal alternative:

```bash
cd ~/Desktop/caseathon/website/routecheck
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

If port 8501 is occupied, add `--server.port 8502`.

Dependencies are installed in this project's own `.venv`. To recreate it, double-click `setup_routecheck.command`, or use Python 3.12:

```bash
cd ~/Desktop/caseathon/website/routecheck
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m streamlit run app.py
```

The standalone ZIP has the same code in a flat `RouteCheck/` folder. In that folder use `python3.12 -m venv .venv`, activate it, install `requirements-dev.txt`, and run `python -m streamlit run app.py`. Its default model/results folders are `models/` and `experiments/` instead of the Desktop layout. Python 3.12 on macOS arm64 was tested. Windows/Linux were not executed here. Direct requirements are pinned; `requirements-lock.txt` captures the full tested dependency set.

## Where to input data

1. Open the app.
2. The **Case data folder** defaults to your `caseathon/data/raw` directory.
3. Click **Find routes** and select a route.
4. Click **Load route**.
5. Use the comparison controls below.

Required JSON files: `route_data.json`, `package_data.json`, `travel_times.json`. `actual_sequences.json` is optional for historical comparisons. The importer reads one top-level route at a time, including from the 1.7 GB travel file, instead of loading that entire file into memory. It accepts the source dataset's `NaN` missing-window convention. Raw files are read only and remain outside the application archive.

The **Use a CSV instead** section remains available for corrected or smaller data. Download the blank template, supply your records, enter the depot coordinates, and load a route. Exact headers:

```text
route_id,stop_id,lat,lon,window_start,window_end,service_min
```

Coordinates are decimal degrees. Windows are minutes relative to departure, with negative values allowed for windows before departure. Blank windows mean unconstrained. Service is minutes. Blank locations are accepted and flagged, but cannot be invented. Stop IDs retain leading zeros. Limit: 5 MB, 1,000 rows, 250 distinct stops per CSV route. The optional CSV road matrix uses `from_id,to_id,travel_min,distance_km` and every directed pair, with `__DEPOT__` representing the depot. The neural policy needs a complete travel-time matrix; CSVs without a matrix use the baseline heuristic automatically.

## A single screen

There are no pages, tabs or sidebar navigation. The app uses normal Streamlit controls in one centered column:

1. Enter the case folder, find routes and load one.
2. Choose Reference, Stale only, Low quality only or Combined, set the affected share and toggle the countermeasure.
3. Click **Run comparison**. A simple table and bar chart show the two outcomes.

The route map, changed records, earlier results, individual trial evidence and review form sit in optional expanders below. The defaults keep most technical detail out of the main workflow. CSV input remains optional.

Changing settings never runs a trial automatically. The app reminds you if the displayed result belongs to earlier settings. Every click on **Run comparison** writes both trial records, readable explanations and a Pending review item. Ordinary Streamlit reruns do not create duplicate trials.

Inside **Previous experiments and trial files**, choose the measured study or your own comparisons. Individual trial JSON files, failure cases, confidence intervals and detector counts remain available. **Review a saved comparison** accepts notes, a decision and optional stop-order override plus self-reported review time.

## The neural model

This is a small **neural constructive policy for combinatorial route sequencing**, trained through supervised imitation of historical driver choices. It is not a pointer-network reproduction, a pretrained LLM, or a reinforcement-learning implementation.

The choice network has 7 inputs, a 24-unit tanh hidden layer and a scalar candidate score. Features encode observed travel time, geometric separation, zone relationships, direction and depot distance. Driver next-stop choices provide positive training examples; up to four unvisited alternatives provide negatives. A masked greedy decoder repeatedly chooses an unvisited stop using the learned score with travel/time-window penalties. It visits every unique delivery once and returns to depot. It does not claim a global optimum.

A second network has 6 spatial/zone inputs, 24 tanh hidden units and a predicted log travel-minute value. It learns reference travel times from training routes. A clean-validation 99th-percentile residual threshold flags unusual inputs. The filter also flags missing travel estimates and simulated feed ages above 30 days. Flagged times are replaced by this learned estimate. It receives no test-route reference edge time, actual driver order, scan outcomes or simulated corruption labels. Legitimate unusual edges can be flagged too.

The model uses NumPy and explicit backpropagation/Adam, avoiding a large deep-learning dependency. Trained weights, seeds, normalization parameters, validation histories and split IDs ship with the project. Both networks freeze before held-out experiments. They do not retrain at each degradation stage.

## Measured experiment design

The supplied files contain 6,112 training-corpus routes. The pilot selects the first 24 complete routes in source order, then splits full routes by distinct date. It preserves all stops on each selected route. Existing unresolved source defects lead to exclusion and a logged reason.

The completed study contains **15 training, 5 validation and 4 later-date test routes**, with no overlapping dates or routes. Held-out status is relative to this local corpus split; the separate official 3,072-route evaluation corpus was not supplied or used. Six stages × four conditions × two modes × three seeds × four test routes = **576 measured trials**. The independent test count is four, not 576.

Conditions:

1. **Reference:** unchanged supplied travel-time estimates.
2. **Stale only:** a seeded fraction of edges uses simulated older, cheaper travel assumptions that miss a congestion increase in the reference world. Simulated age increases with the stage.
3. **Low quality only:** a seeded fraction contains missing or implausibly small/large travel times.
4. **Combined:** independent stale and low-quality selections, with possible overlap.

The source matrix supplies the controlled offline reference world, not fresh ground truth. Every with/without pair uses identical source data, corrupted observations, seed, routes and frozen model. The detector uses only observed values and available feed-age metadata. Defect labels are used after prediction to count true/false positives and negatives. Total directed edge volume stays fixed; the low-quality/legacy feed share increases. The horizontal axis is a **simulated progression**, not a measured Amazon operational timeline.

The primary objective is reference replay duration in minutes: supplied travel time + planned package service + waiting for windows + depot return. Lower is better. Service seconds are summed per stop and converted to minutes. Package-window intersections supply conservative stop windows. Stops with no specified window are unconstrained; estimated on-time rate uses windowed stops only. Unscheduled windowed stops remain in the denominator. No actual arrival timestamps exist in these files. The travel file has no road distances, so that metric remains unavailable for case JSON.

Secondary measures include adjacency agreement with driver order, on-time percentage, scheduled-stop coverage, invalid/withheld outputs, detector confusion counts, legitimate-rare-edge false positives and compute latency. Adjacent-pair agreement is a simple diagnostic, not the official Amazon sequence score. Human routes are not assumed optimal. Bootstrap intervals resample route-level paired averages, preserving dependence among seeds/stages on one route. Four test routes provide weak evidence for generalization.

See `REPORT.md` for measured results, including nominal and stale-only cases where the countermeasure causes harm. Negative paired recovery means the fix made a route worse. No guaranteed improvement is built into the charts.

## Every trial has a file

Open `results/routecheck/results.csv` in the Desktop project to find each `trial_id` and its `evidence_file`. Under `studies/<study-id>/trials/`, every trial has:

- A `.json` with settings, route ID, seed, model/reference fingerprints, recommended order, all reference replay arrivals, flagged edges, metrics and compute runtime.
- A `.md` with a readable explanation and instructions for finding its paired comparison.

`modification_log.jsonl.gz` records each changed edge's original/modified time, simulated age, source feed, condition, stage and seed. Open it with a gzip-aware tool or Python. Each trial points to this shared log; paired trials deliberately share the same changes. The log is not fed into the model.

Each new batch creates a new timestamped study folder with its own result tables, model weights and manifest. Older study folders are preserved. Top-level result tables identify the latest run. Interactive trials get unique timestamped filenames in `interactive_trials/`.

## Reproduce training and all experiments

Desktop layout:

```bash
cd ~/Desktop/caseathon
website/routecheck/.venv/bin/python training/routecheck/run_study.py --data-dir data/raw --routes 24
```

Standalone folder:

```bash
python run_study.py --data-dir /path/to/caseathon/data/raw --routes 24
```

The run takes roughly one to two minutes on this Mac. Runtime varies by machine and disk. It saves new evidence for every trial, retains previous study snapshots, and updates the default neural weights. Stop/restart Streamlit after retraining so all model changes are clear to the presenter. First scanning the large case files may take a moment. No paid inference or routing services are used.

## Connect your teammate's code

Keep the website unchanged. Make the module importable, then set:

```bash
export ROUTECHECK_MODEL="teammate_model:solve"
```

The callable receives `solve(payload: dict, countermeasures: bool) -> dict`. Payload keys are `schema_version`, `route_id`, `depot`, `stops`, `speed_kmh`, `travel_matrix`. Stops contain `stop_id`, `lat`, `lon`, `window_start`, `window_end` (possibly None), `service_min`, and optional zone/package counts. Edges use `from_id`, `to_id`, `travel_min` and optional `distance_km`; the depot ID is `__DEPOT__`. Simulation observations also include feed age/source metadata. Missing damaged travel times can be None. Input objects are copied before calls.

Return:

```python
{
    "order": ["AD", "AF"],  # every intended delivery ID, no depot
    "planned_duration_min": 120.0,
    "issues": [{"stop_id": "AD", "reason": "Travel estimate needs checking"}],
    "model_name": "Team model v2"
}
```

Unknown IDs, invented repeated visits, nonfinite/negative durations and malformed issue records fail validation visibly. Errors never silently fall back to made-up results. The comparison function and batch study currently target the provided neural policy, so a different teammate model requires a documented experiment adapter in experiments.py. model_integration.py remains the common adapter/evaluator module. Do not represent the existing study as evidence for an untested replacement model.

## Human responsibility

A route planner uses recommendations. A data-quality reviewer checks questionable source values and leaves notes, corrections or an override. A model owner reviews performance and false positives before any offline pilot. The app must defer when source locations/windows remain unresolved. Approval is a local review record and never dispatches a vehicle.

Correct a source error in the CSV/JSON and reload it to recompute metrics. A route-order override must contain every delivery ID once, without the depot. It records the proposed order and decision; the original model charts are not recalculated for that override. Review time is self-reported and stored separately from compute latency. No participant study or trust/readiness claim is made.

Before any real pilot, the model owner should expand station/date coverage and agree acceptance criteria with an operations reviewer. No vehicle or live service is connected.

## Tests

```bash
cd ~/Desktop/caseathon/website/routecheck
.venv/bin/python -m pytest -q
```

Tests cover CSV validation, JSON units and missing-window semantics, streaming parsing, reference isolation, known matrix replay arithmetic, invalid adapter outputs, neural learning/weight restoration, frozen inference, corruption repeatability, chronology, per-trial evidence completeness, persistence, review overrides and UI navigation/save/review workflows. Tests use temporary databases and evidence folders. They do not modify raw data or genuine review records.

## Sources, permissions and disclosure

The original brief is `Case-a-thon Prompt.pdf` in your Desktop project, particularly pages 3, 4, 7 and 8. The source dataset and schema are documented at:

- https://registry.opendata.aws/amazon-last-mile-challenges/
- https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md

The registry lists CC BY-NC 4.0. The raw files remain in your folder and are excluded from the app archive. Derived experiment logs include selected reference values and remain part of this local noncommercial project. Check applicable terms before public redistribution. The original retrieval date is unknown; file sizes/mtime, selected-route content hashes and the 2021 release description appear in the manifest. No person/location reidentification was attempted.

See `AI_USE.md` for the coding-assistance disclosure. No commercial API calls or paid compute were needed. Electricity and human verification costs were not measured.

## Cloud deployment

Use the separately prepared RouteCheck-cloud folder to deploy with bundled real routes and durable Neon PostgreSQL storage. Follow DEPLOYMENT.md. New trials and reviews are stored in PostgreSQL when ROUTECHECK_DATABASE_URL is configured; the ordinary local app continues using SQLite.

## Teammate model integration

Upload the teammate project's actual metrics.csv or pilot-output ZIP under **Teammate model: pilot results**. For individual stop orders, use **Load results from your teammate's model** after loading the matching route. See MODEL_INTEGRATION.md for the prepared exporter and live callable option. Source-code ZIPs and mock API fixtures are not measured results. The supplied teammate model is CatBoost repair; the original 576-trial neural study remains separate.
