"""RouteCheck's Streamlit interface. Routing and experiments live in training code."""
import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from project_paths import MODEL_DIR, RAW_DIR, RESULTS_DIR, TRAINING_DIR
sys.path.insert(0, str(TRAINING_DIR))

from case_data import catalog, load_case_route
from charts import route_map, study_plot
from study_evidence import read_study_file
from data_input import TEMPLATE, make_dataset, read_matrix, read_stops
from evidence import model_fingerprint, save_live_comparison, save_trial
from experiments import CONDITIONS, STAGES, experiment
from model_integration import run_experiment, predict
from neural_policy import models
from result_import import import_comparison
from storage import load_reviews, load_runs, remove_synthetic_runs, review_run, save_run, read_artifact, cloud_mode, storage_ready, StorageError, save_model_batch, load_model_batches

st.set_page_config(page_title="RouteCheck", page_icon="📍", layout="centered")
try:
    for setting in ("ROUTECHECK_DATABASE_URL", "ROUTECHECK_MODEL"):
        if setting in st.secrets:
            os.environ[setting] = st.secrets[setting]
except FileNotFoundError:
    pass
try:
    remove_synthetic_runs()
except StorageError as error:
    st.error(str(error))
    st.stop()
if not storage_ready():
    st.info("Study preview: add your Neon connection string in Streamlit Secrets to enable saved comparisons and reviews.")
if st.session_state.get("dataset", {}).get("source") == "synthetic_demo":
    st.session_state.pop("dataset", None)

st.title("RouteCheck")
st.write("Compare delivery routes when travel-time data is outdated or unreliable.")


def remember_dataset(data):
    st.session_state.dataset = data
    st.session_state.pop("comparison", None)
    st.success(f'Loaded {len(data["stops"])} stops. Choose the experiment settings below.')


def comparison_table(result):
    return pd.DataFrame([
        {"Plan": label, "Estimated minutes": result[key]["duration_min"],
         "On time (%)": result[key]["on_time_pct"], "Stops scheduled": result[key]["completed"]}
        for label, key in [(result.get("before_label", "Without countermeasure"), "before"), (result.get("after_label", "Current settings"), "after")]
    ])


def run_comparison(data, condition, stage, seed, enabled):
    if data.get("travel_matrix"):
        external = bool(os.environ.get("ROUTECHECK_MODEL"))
        networks = None if external else models()
        solver = predict if external else None
        before_row, before_pred, changes, before = experiment(data, condition, stage, seed, False, networks, solver=solver)
        after_row, after_pred, _, after = experiment(data, condition, stage, seed, enabled, networks, solver=solver)
        result = dict(source="case_json_simulated_defects", route_id=data["route_id"], dataset_hash=data["dataset_hash"],
                      scenario=condition, severity=stage, seed=seed, countermeasures=enabled,
                      model_name=after_pred["model_name"], total_stops=len(data["stops"]),
                      stop_ids=[s["stop_id"] for s in data["stops"]], issues=data["issues"],
                      before=before, after=after, before_prediction=before_pred, after_prediction=after_pred,
                      experiment_diagnostics=after_row, modifications=changes)
        directory = Path(os.environ.get("ROUTECHECK_EVIDENCE_DIR", str(RESULTS_DIR / "interactive_trials")))
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("interactive_%Y%m%dT%H%M%S_%fZ")
        log_path = directory / (stamp + "_changes.json")
        log_path.write_text(json.dumps(changes, indent=2))
        result["trial_files"] = []
        for suffix, row, pred, metrics in [("before", before_row, before_pred, before), ("current", after_row, after_pred, after)]:
            save_trial(directory, stamp + "_" + suffix, row, pred, metrics, data["dataset_hash"], ("External callable: " + os.environ["ROUTECHECK_MODEL"] if external else model_fingerprint()), log_path.name)
            result["trial_files"].append(str(directory / (stamp + "_" + suffix + ".json")))
    else:
        if condition != "Reference":
            raise ValueError("Load case JSON or a travel matrix to test damaged travel-time data.")
        result = run_experiment(data if os.environ.get("ROUTECHECK_MODEL") else dict(data, planner="baseline"), enabled)
        result["trial_files"] = save_live_comparison(data, result)
    result["review_id"] = save_run(result)
    return result


if cloud_mode():
    from hosted_data import bundled_routes, load_bundled_route
    entries = bundled_routes()
    st.write("Choose a real route from the held-out test set.")
    route = st.selectbox("Route", [r["route_id"] for r in entries])
    st.caption("Four historical routes bundled from the Amazon research dataset. Input defects are simulated; performance is an offline estimate.")
    if st.button("Load route"):
        remember_dataset(load_bundled_route(route))
else:
    st.write("Choose a route from your original case files.")
    folder = st.text_input("Data folder", str(RAW_DIR))
    if st.button("Find routes"):
        try:
            st.session_state.routes = catalog(folder)
            st.session_state.folder = str(Path(folder).expanduser().resolve())
        except (ValueError, OSError, KeyError) as error:
            st.error(str(error))
    entries = st.session_state.get("routes", [])
    if entries and str(Path(folder).expanduser().resolve()) == st.session_state.get("folder"):
        st.caption(f'{len(entries):,} routes found')
        route = st.selectbox("Route", [r["route_id"] for r in entries])
        info = next(r for r in entries if r["route_id"] == route)
        st.caption(f'{info["station"]} · {info["date"]} · {info["stops"]} stops')
        if st.button("Load route"):
            try:
                with st.spinner("Reading this route…"):
                    remember_dataset(load_case_route(st.session_state.folder, route))
            except (ValueError, OSError, KeyError, TypeError) as error:
                st.error(str(error))
with st.expander("Use a CSV instead"):
    st.download_button("Blank stop template", TEMPLATE, "routecheck_stops.csv", "text/csv")
    st.caption("Windows are minutes from departure; blank windows mean no deadline. Coordinates use decimal degrees. Service times are minutes.")
    uploaded = st.file_uploader("Stops CSV", type="csv")
    if uploaded:
        try:
            raw = uploaded.getvalue()
            frame = read_stops(raw)
            route = st.selectbox("CSV route", list(frame.route_id.unique()))
            lat = st.number_input("Depot latitude", -90., 90., value=None, format="%.6f")
            lon = st.number_input("Depot longitude", -180., 180., value=None, format="%.6f")
            speed = st.number_input("Estimated speed (km/h)", 1., 130., 25.)
            matrix_file = st.file_uploader("Optional matrix CSV", type="csv")
            st.caption("Matrix headers: from_id,to_id,travel_min,distance_km. Include all directed pairs and __DEPOT__. Without a matrix, the CSV uses the baseline heuristic and straight-line estimates.")
            if st.button("Load CSV route"):
                if lat is None or lon is None:
                    st.error("Enter both depot coordinates.")
                else:
                    ids = set(frame[frame.route_id == route].stop_id)
                    matrix = read_matrix(matrix_file.getvalue(), ids) if matrix_file else None
                    remember_dataset(make_dataset(frame, route, lat, lon, speed, raw, matrix))
        except (ValueError, OSError, pd.errors.ParserError) as error:
            st.error(str(error))
if st.session_state.get("dataset"):
    data = st.session_state.dataset
    with st.expander("Check the loaded data"):
        st.dataframe(pd.DataFrame(data["stops"]), hide_index=True, width="stretch")
        st.write("Source-data flags:", len(data["issues"]))
        if data["issues"]:
            st.dataframe(pd.DataFrame(data["issues"]), hide_index=True)
        st.caption("Checks: missing coordinates, duplicate records, reversed windows, and zero or unusually long service durations. No source errors are injected here.")
with st.expander("Files and model details"):
    st.write("The case importer uses route_data.json, package_data.json and travel_times.json. actual_sequences.json supplies historical driver order for training/comparison, never live planner inputs.")
    st.write("The default is a small neural route policy trained by driver imitation. Full setup, architecture, units and limitations are in the README.")
    st.download_button("Download README", Path(__file__).with_name("README.md").read_text(), "RouteCheck_README.md")


st.subheader("Compare routes")
data = st.session_state.get("dataset")
if not data:
    st.info("Load a route above to start.")
elif data["issues"]:
    st.warning("Correct the source-data flags before injecting controlled errors. Open Check the loaded data above.")
else:
    st.write("**Loaded route**")
    route_info = {"Route": data["route_id"], "Station": data.get("station", "CSV import"),
                  "Date": data.get("route_date", "Not supplied"), "Delivery stops": len(data["stops"]),
                  "Packages": data.get("package_count", "Not supplied"),
                  "Travel data": "Supplied travel-time matrix" if data["travel_matrix"] else "Straight-line estimates"}
    selected_path = RESULTS_DIR / "selected_routes.csv"
    if selected_path.is_file():
        selected = pd.read_csv(selected_path)
        match = selected[selected.route_id == data["route_id"]]
        route_info["Study split"] = match.iloc[0]["split"] if not match.empty else "Outside the 24-route pilot"
    st.dataframe(pd.DataFrame(route_info.items(), columns=["Detail", "Value"]).astype(str), hide_index=True, width="stretch")
    st.caption("The study evaluates later test dates separately. A comparison on a training route is a demonstration, not held-out evidence.")
    with st.expander("Load results from your teammate's model"):
        st.write("Upload both baseline and current stop orders for this route. RouteCheck calculates the displayed metrics from the loaded historical data and saves both trials for review.")
        st.caption("This imports results; it does not run training or recreate the model's input defects. Use Run comparison below to execute the configured live model.")
        export = st.file_uploader("Model results JSON", type="json", key="model_results")
        if st.button("Import and save model results", disabled=not storage_ready()):
            try:
                if export is None:
                    raise ValueError("Choose a model results JSON first.")
                imported = import_comparison(export.getvalue(), data)
                imported["trial_files"] = save_live_comparison(data, imported)
                imported["review_id"] = save_run(imported)
                st.session_state.comparison = imported
                st.session_state.comparison_request = None
            except (ValueError, OSError, StorageError) as error:
                st.error(str(error))
        st.download_button("Model integration guide", Path(__file__).with_name("MODEL_INTEGRATION.md").read_text(), "MODEL_INTEGRATION.md")
    st.write("**Experiment settings**")
    condition = st.selectbox("Data condition", CONDITIONS)
    descriptions = {
        "Reference": "Use the original supplied travel times. This checks whether the countermeasure changes an ordinary route.",
        "Stale only": "Simulate an older feed with cheaper travel assumptions that miss a congestion increase. The original matrix still supplies the evaluation reference.",
        "Low quality only": "Replace some travel-time observations with missing or extreme values. These changes represent unreliable input records.",
        "Combined": "Apply both stale and low-quality observations. Their edge selections can overlap, so the unique affected share can exceed the slider value."
    }
    st.caption(descriptions[condition])
    stage = st.select_slider("Simulated affected share (%)", STAGES, value=0 if condition == "Reference" else 20, disabled=condition == "Reference")
    enabled = st.toggle("Use countermeasure", value=True)
    st.caption("When enabled, the filter flags missing times, old feed entries and unusually large prediction residuals, then replaces them with learned estimates. It never reads the hidden reference values or defect labels.")
    with st.expander("Experiment settings"):
        seed = st.selectbox("Random seed", [7, 19, 43])
        st.caption("The neural model stays frozen. Stale values simulate an old feed missing a congestion increase. Low-quality values are missing or extreme. These are controlled simulations, not documented Amazon incidents.")
    effective_stage = 0 if condition == "Reference" else stage
    request = (data["dataset_hash"], condition, effective_stage, seed, enabled)
    if st.button("Run comparison", type="primary", disabled=not storage_ready()):
        try:
            with st.spinner("Planning and comparing…"):
                st.session_state.comparison = run_comparison(data, condition, effective_stage, seed, enabled)
                st.session_state.comparison_request = request
        except Exception as error:
            st.error(f'Comparison failed: {error}')
    result = st.session_state.get("comparison")
    if result:
        if not result.get("external_results") and request != st.session_state.get("comparison_request"):
            st.info("Settings changed. Click Run comparison to update the result below.")
        if result.get("external_results"):
            st.info(result["provenance_note"])
        st.caption(f'Last run: {result["scenario"]} · {result.get("severity", 0)}% · Countermeasure {"on" if result["countermeasures"] else "off"}')
        st.write("**Comparison result**")
        st.dataframe(comparison_table(result), hide_index=True, width="stretch")
        chart_data=comparison_table(result).set_index("Plan")[["Estimated minutes"]]
        if chart_data["Estimated minutes"].notna().any():
            st.bar_chart(chart_data,height=220,color="#4D8B68")
        st.caption("Lower minutes is better. On-time percentage covers stops with specified windows. Both plans replay on the untouched historical travel matrix and planned service times, not fresh delivery ground truth.")
        before, after = result["before"], result["after"]
        st.write("**What this result means**")
        if before["duration_min"] is not None and after["duration_min"] is not None:
            change = after["duration_min"] - before["duration_min"]
            direction = "longer" if change > 0 else "shorter" if change < 0 else "unchanged"
            if change == 0:
                st.write(f'Both plans take an estimated {after["duration_min"]:.1f} minutes under the same reference replay.')
            else:
                st.write(f'The current plan is {abs(change):.1f} minutes {direction} under the same reference replay.')
        st.write(f'It schedules {after["completed"]} of {result["total_stops"]} deliveries, compared with {before["completed"]} in the baseline plan.')
        if before["on_time_pct"] is not None and after["on_time_pct"] is not None:
            delta = after["on_time_pct"] - before["on_time_pct"]
            st.write(f'Estimated on-time performance is {after["on_time_pct"]:.1f}%, compared with {before["on_time_pct"]:.1f}% in the baseline plan ({delta:+.1f} percentage points).')
        st.caption("These are offline estimates using historical travel times and planned service durations. The original data is a consistent reference, not verified current traffic or an optimal route.")
        diagnostics = result.get("experiment_diagnostics")
        if diagnostics:
            st.write(f'{diagnostics["changed_edges"]:,} of {diagnostics["total_edges"]:,} directed travel records were changed by the simulation. The filter repaired {diagnostics["repairs"]:,}. Mean simulated record age: {diagnostics["mean_age_days"]:.1f} days.')
        st.success(f'Saved comparison {result["review_id"]}. Open the review section below to leave a decision.')
        with st.expander("Files saved for this run"):
            st.write("Each trial has its own JSON and readable summary. The modification log records the exact original and changed values.")
            for index, filename in enumerate(result.get("trial_files", [])):
                label = result.get("before_label", "Without countermeasure") if index == 0 else result.get("after_label", "Current settings")
                try:
                    st.download_button(label + " JSON", read_artifact(filename), Path(filename).name, "application/json")
                    summary = filename.rsplit(".", 1)[0] + ".md"
                    st.download_button(label + " readable summary", read_artifact(summary), Path(summary).name, "text/markdown")
                except (OSError, StorageError) as error:
                    st.error(str(error))
            st.download_button("Full modification log", json.dumps(result.get("modifications", []), indent=2), "modifications.json", "application/json")
        st.download_button("Download this comparison", json.dumps(result, indent=2), "routecheck_comparison.json", "application/json")
        with st.expander("See the route"):
            st.plotly_chart(route_map(result, data))
            st.caption("Lines connect coordinates; the JSON does not provide road distances or road geometry.")
        with st.expander("What changed in the data?"):
            diagnostics = result.get("experiment_diagnostics")
            if diagnostics:
                st.write(f'Repaired edges: {diagnostics["repairs"]}. False positives: {diagnostics["fp"]}. Missed defects: {diagnostics["fn"]}.')
                st.caption(f'Compute time: {diagnostics["runtime_seconds"]*1000:.1f} ms. Human review time is recorded separately.')
                st.dataframe(pd.DataFrame(result.get("modifications", [])).head(100), hide_index=True)
                st.caption("Showing the first 100 changes. The downloaded comparison and trial evidence retain the full log.")


with st.expander("Teammate model: pilot results"):
    from pilot_results import read_pilot, pilot_time_summary
    st.write("Load the actual CatBoost experiment output from BearTyree/caseathon. Use metrics.csv or a ZIP of outputs/pilot; the GitHub source ZIP is not a measured result.")
    pilot_file = st.file_uploader("Pilot output ZIP or metrics CSV", type=["zip", "csv"])
    if st.button("Import pilot results", disabled=not storage_ready()):
        try:
            if pilot_file is None: raise ValueError("Choose the pilot output file first.")
            batch = read_pilot(pilot_file.getvalue(), pilot_file.name)
            st.session_state.pilot_batch = save_model_batch(batch)
            st.success("Saved every trial and the source results archive.")
        except (ValueError, OSError, StorageError) as error:
            st.error(str(error))
    batches = load_model_batches()
    if batches:
        batch_id = st.selectbox("Imported experiment", [b["id"] for b in batches])
        batch = next(b for b in batches if b["id"] == batch_id)
        st.caption(batch["notice"])
        frame = pd.DataFrame(batch["rows"])
        st.write(f'{len(frame):,} trial rows on {batch["route_count"]} independent routes. Model methods: raw input, rule repair, CatBoost.')
        st.write(f'Invalid output rows: {int((~frame.valid).sum())}. These remain in the saved evidence and are excluded from the duration chart.')
        scenario = st.selectbox("Pilot scenario", sorted(frame.scenario.unique()))
        condition = st.selectbox("Pilot condition", sorted(frame.condition.unique()))
        summary = pilot_time_summary(batch["rows"], scenario, condition)
        if not summary.empty:
            st.line_chart(summary.pivot(index="time_point",columns="method",values="duration_min"), y_label="Reported duration (minutes)")
        st.caption("Time points follow the teammate simulator (one step = 60 minutes). Means first average seeds within each route. This chart does not combine these measurements with the older neural-policy study.")
        st.dataframe(frame[(frame.scenario == scenario) & (frame.condition == condition)], hide_index=True)
        st.download_button("Download original result files", read_artifact("artifact://" + batch_id + "/pilot_results.zip"), "pilot_results.zip")
        number = st.number_input("Pilot trial number", min_value=1, max_value=len(frame), value=1, step=1)
        prefix = "artifact://" + batch_id + f"/trial_{number:05d}"
        st.download_button("Download pilot trial JSON", read_artifact(prefix + ".json"), f"trial_{number:05d}.json")
        st.download_button("Download pilot trial summary", read_artifact(prefix + ".md"), f"trial_{number:05d}.md")
        if batch["manifest"]:
            st.download_button("Pilot provenance manifest", json.dumps(batch["manifest"], indent=2), "pilot_manifest.json")
    else:
        st.caption("No measured teammate results imported yet. The source repository's mock API fixtures are not pilot findings.")


with st.expander("Previous experiments and trial files"):
    view = st.radio("Show", ["Study results", "My comparisons"], horizontal=True)
    if view == "My comparisons":
        runs = load_runs()
        if not runs:
            st.info("Run a comparison to add it here.")
        else:
            table = pd.DataFrame([dict(run=r["id"], time_utc=r["created_at"], condition=r["result"].get("scenario"), stage=r["result"].get("severity"), minutes=r["result"]["after"]["duration_min"], status=r["status"]) for r in runs])
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button("Download my results", table.to_csv(index=False), "my_comparisons.csv")
            selected_run = st.selectbox("Saved evidence", [r["id"] for r in runs])
            record = next(r["result"] for r in runs if r["id"] == selected_run)
            st.download_button("Download saved comparison", json.dumps(record, indent=2), selected_run + ".json")
            for number, reference in enumerate(record.get("trial_files", [])):
                st.download_button("Trial " + str(number + 1) + " JSON", read_artifact(reference), Path(reference).name, key="saved_trial_" + str(number))
                summary = reference.rsplit(".", 1)[0] + ".md"
                st.download_button("Trial " + str(number + 1) + " summary", read_artifact(summary), Path(summary).name, key="saved_summary_" + str(number))
    elif not (RESULTS_DIR / "results.csv").is_file():
        st.info("Run the training/study script to create measured results.")
    else:
        frame = pd.read_csv(RESULTS_DIR / "results.csv")
        manifest = json.loads((RESULTS_DIR / "manifest.json").read_text())
        counts = manifest["split_counts"]
        st.write(f'{manifest["experiment_rows"]} measured trials on {counts["test"]} independent test routes. The model trained on {counts["train"]} earlier routes and used {counts["validation"]} validation routes.')
        condition = st.selectbox("Condition", CONDITIONS, index=3)
        st.plotly_chart(study_plot(frame, condition))
        st.caption("Simulated progression, averaged over held-out routes and three seeds. More repeats do not increase the number of independent routes.")
        stage = st.select_slider("Results at stage (%)", STAGES, value=50)
        summary = pd.read_csv(RESULTS_DIR / "summary.csv")
        table = summary[summary.stage_pct == stage][["condition", "countermeasures", "mean_duration_min", "mean_on_time_pct"]].rename(columns={"condition":"Condition", "countermeasures":"Fix enabled", "mean_duration_min":"Mean minutes", "mean_on_time_pct":"Mean on time (%)"})
        st.dataframe(table, hide_index=True, width="stretch")
        with st.expander("Failures, uncertainty and detector details"):
            pairs = pd.read_csv(RESULTS_DIR / "paired_comparisons.csv")
            st.dataframe(pairs.sort_values("recovery_min").head(5), hide_index=True)
            st.caption("Negative recovery means the countermeasure made a route worse.")
            ci = pd.read_csv(RESULTS_DIR / "uncertainty.csv")
            st.dataframe(ci[ci.stage_pct == stage], hide_index=True)
            detail = frame[(frame.condition == condition) & (frame.stage_pct == stage)]
            st.dataframe(detail[["route_id", "seed", "countermeasures", "fp", "fn", "rare_false_positives", "stops_scheduled", "runtime_seconds"]], hide_index=True)
        with st.expander("Open an individual trial"):
            trial_id = st.selectbox("Trial", frame.trial_id.tolist())
            trial = frame[frame.trial_id == trial_id].iloc[0]
            relative = Path(trial.evidence_file)
            try:
                st.markdown(read_study_file(relative.with_suffix(".md").as_posix()).decode())
                st.download_button("Download trial JSON", read_study_file(relative.as_posix()), relative.name, "application/json")
            except FileNotFoundError:
                st.warning("This study trial file is unavailable in the deployment package.")
        with st.expander("Download the evidence"):
            for filename in ["results.csv", "manifest.json", "selected_routes.csv", "modification_log.jsonl.gz", "REPORT.md"]:
                st.download_button(filename, (RESULTS_DIR / filename).read_bytes(), filename)

with st.expander("Review a saved comparison"):
    runs = load_runs()
    if not runs:
        st.info("Run a comparison first, then return here to review it.")
    else:
        run_id = st.selectbox("Comparison", [r["id"] for r in runs])
        saved = next(r for r in runs if r["id"] == run_id)
        result = saved["result"]
        st.caption(f'{result["route_id"]} · {result.get("scenario")} · {saved["status"]}')
        st.dataframe(comparison_table(result), hide_index=True, width="stretch")
        with st.expander("Check flagged records"):
            issues = result.get("issues", []) + result["after_prediction"]["issues"]
            st.write(f'{len(issues)} flags reported. Showing the first 50.')
            if issues:
                st.dataframe(pd.DataFrame(issues).head(50), hide_index=True)
            st.caption("Check questionable travel times against a trusted source. Correct the source and reload to rerun. A learned replacement can make a route worse.")
        with st.form("review"):
            reviewer = st.text_input("Your name")
            decision = st.selectbox("Decision", ["Needs revision", "Approved", "Rejected"])
            note = st.text_area("What did you check or correct?")
            with st.expander("Optional override and review time"):
                override = st.text_area("Proposed stop order (comma-separated IDs)")
                minutes = st.number_input("Minutes spent reviewing", min_value=0., max_value=1440., value=None)
                st.caption("Include every delivery once, excluding depot. An override is saved as a proposed order; the model metrics above remain unchanged. Time is self-reported human effort.")
            acknowledge = st.checkbox("I checked the plan and its flagged inputs.")
            if st.form_submit_button("Save review"):
                if not acknowledge:
                    st.error("Check the plan and acknowledge the review first.")
                else:
                    try:
                        review_run(run_id, reviewer, decision, note, [x.strip() for x in override.split(",")] if override.strip() else None, minutes*60 if minutes is not None else None)
                        st.rerun()
                    except (ValueError, StorageError) as error:
                        st.error(str(error))
        audit = [r for r in load_reviews() if r["run_id"] == run_id]
        if audit:
            st.write("Review history")
            st.dataframe(pd.DataFrame(audit), hide_index=True)
            if audit[0].get("override_order_json"):
                st.download_button("Download proposed order", audit[0]["override_order_json"], "reviewed_order.json")

st.caption("RouteCheck is an offline prototype. It does not dispatch deliveries or claim optimal routes.")
