# Put your model's results into RouteCheck

There are two independent ways to connect a model. Exporting JSON is the simplest: your teammate can keep training on their own computer or notebook and the website needs no PyTorch, GPU, or training data.

The supplied `caseathon-main.zip` has now been inspected. The actual model is CatBoost travel-time repair, with the same deterministic greedy planner used for raw, rule and CatBoost estimates. It is not a neural combinatorial optimizer. No trained model.cbm or measured output files were included. Its API fixtures explicitly use invented data; RouteCheck does not treat them as findings or connect to that mock API.

## Easiest path: upload the actual pilot outputs

Your teammate runs their usual `python -m caseathon.experiment --config configs/pilot.json` in their own prepared environment. They send you `outputs/pilot/metrics.csv`, or a ZIP of that output folder with manifest.json and the other result files. In RouteCheck, open **Teammate model: pilot results** and click **Import pilot results**. You do not need to load any route first; routes outside the hosted four-route excerpt work here.

The website reads the actual native column names (route_id, scenario, seed, time_point, condition, method, duration_seconds and the other route outcomes), converts seconds to minutes only for display, and charts raw/rule/CatBoost across all four conditions and time points. It averages seeds within each route before averaging independent routes. Invalid outputs remain in the evidence and are counted, not treated as successful routes. Before saving, it checks paired method completeness, duplicate trial keys, required units/types, and matching reference durations if supplied. A ZIP's manifest result count must match the CSV.

Every original trial row gets its own JSON and readable Markdown artifact in SQLite locally or Neon in the cloud, alongside the imported results archive and source hash. You can download the archive and select any individual trial. Reopening the app retrieves imported batches from the database. These are reported experiment measurements; the app cannot independently verify a trained checkpoint from metrics.csv alone. A CSV-only import lacks the full provenance provided by manifest.json. Nothing in this upload retrains a model or recomputes its simulated reference.

## Send your teammate the prepared model handoff

`Teammate-model-integration` is a copy of their source ZIP with just an exporter and an export call added to evaluate_holdout. After running their existing smoke/pilot command, it writes `routecheck_trials/comparison_*.json`, one paired order export per route/scenario/seed/time/condition. It converts the model's integer stop indices to original string stop IDs and removes the repeated depot endpoints. It preserves reported raw/rule/CatBoost outcomes and settings in metadata. The model, planner, simulation and training configurations remain unchanged. Its test suite and a new export-loop test pass.

To inspect an individual exported plan and use RouteCheck's human review, load that route in the local app and upload its comparison JSON using Option 1 below. RouteCheck's replay uses the historical matrix, which differs from the teammate's evolving simulated reference. Both sets of numbers are retained with explicit provenance; they are never mixed into the old neural-policy study.


## Option 1: export and upload the two orders

After inference, write a UTF-8 JSON file in this format. The two IDs below illustrate the schema only; use every delivery stop ID from the chosen real route, exactly once. Keep IDs as strings and exclude the Station/depot. The before order must be your actual baseline output; after must be the actual current model output. `countermeasures` describes whether your own current inference applied a countermeasure.

```json
{
  "schema_version": 1,
  "route_id": "your exact RouteID",
  "countermeasures": true,
  "before": {"model_name": "your baseline model", "order": ["stop A", "stop B"]},
  "after": {"model_name": "your current model", "order": ["stop B", "stop A"]},
  "metadata": {"checkpoint": "your checkpoint ID", "seed": 7}
}
```

Copy the single file `model_export.py` next to your Python inference script and use this convenience function. It uses only Python's standard library, so it works independently of RouteCheck. You can also simply use Python's json module to write the dictionary above:

```python
from model_export import write_comparison
write_comparison(
    "route_results.json",
    route_id=route_id,
    before_order=baseline_stop_ids,
    after_order=model_stop_ids,
    model_name="Team routing model v1",
    countermeasures=True,
    metadata={"checkpoint": "checkpoint-001", "seed": 7},
)
```

On the website, load that same route, open **Load results from your teammate's model**, select the JSON, and click **Import and save model results**. Both trial files, the original export and its SHA-256 are retained for downloads and human review. Imported comparisons appear under My comparisons, separate from the original study. Settings changes do not import or run anything until a button is clicked.

The website independently replays both orders on the loaded historical matrix and service times. It never trusts a model's claimed duration, success rate or stop count. It checks route ID and requires every delivery exactly once; unknown, repeated or missing stops are rejected. With no travel matrix, replay uses explicitly labeled straight-line estimates. Input exports describe what the teammate ran: RouteCheck cannot independently verify their checkpoint, countermeasure or simulated scenario. Original model metrics may be preserved in metadata, but never replace the independently calculated figures. A results import does not rerun the four defect scenarios or reproduce that model's training study.

## Option 2: execute the model from the website

Keep training separate. Copy the inference-only Python module and its checkpoint into the deployment repository, add its actual runtime dependencies to requirements.txt, and implement this function:

```python
# teammate_adapter.py
# Load your trained checkpoint once with a cache in your inference module.
def solve(payload, countermeasures):
    # Replace this body with YOUR model's actual inference call.
    # payload["stops"] contains stop_id, lat, lon, window_start,
    # window_end (None = unconstrained), and service_min.
    # payload["depot"] contains lat/lon.
    # payload["travel_matrix"] has from_id,to_id,travel_min,distance_km;
    # depot ID is "__DEPOT__". Times are MINUTES, not seconds.
    # In simulated scenarios, travel_min may be None or extreme;
    # age_days and source_feed are available. Implement your input handling
    # and countermeasure according to the Boolean argument.
    raise NotImplementedError("Map the teammate's actual inference call here")
    # Return exactly:
    # {"order": stop_ids, "planned_duration_min": finite_minutes,
    #  "model_name": "Team routing model v1", "issues": []}
```

Configure the callable in Streamlit Secrets (alongside the existing Neon URL):

```toml
ROUTECHECK_MODEL = "teammate_adapter:solve"
```

Or set ROUTECHECK_MODEL as a local environment variable before launching Streamlit. This setting is server-owned; uploads cannot execute Python or choose arbitrary modules. The website passes only observed inputs to the callable, never actual driver sequence, outcome labels, hidden clean matrix values, or injected defect labels. Both paired calls receive independent copies. The toggle is passed to your function: your model must implement its actual meaning. If it cannot handle missing inputs or apply a countermeasure, leave that limitation explicit rather than claiming it does.

Live matrix comparisons now respect this callable in all four scenarios; previously they always used the bundled neural policy. Baseline and current calls are validated and replayed against the same untouched reference. Existing 576-trial study results continue to describe the bundled policy, not the new model. Changing the live callable does not alter the historical study.

The deployed excerpt contains four test routes. Use the local app with the original files or validated CSV uploads for a route outside that excerpt. Do not add private checkpoints, huge training datasets, secrets, or training jobs to Streamlit without checking their size and runtime needs first.
