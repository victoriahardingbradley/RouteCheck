"""Routing adapter for uploaded data. No Streamlit dependency or invented references."""
import copy
import importlib
import math
import os
import time
from statistics import median
from data_input import inspect_stops


def distance(a, b):
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    h = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(b["lon"]-a["lon"])/2)**2
    return 6371*2*math.asin(min(1,math.sqrt(h)))


def leg(a_id, b_id, a, b, dataset):
    if a_id == b_id: return 0., 0.
    matrix = dataset.get("travel_matrix", [])
    if matrix:
        # Lookup is built once per planner/evaluator call, never serialized.
        edge = dataset["_matrix_lookup"][a_id,b_id]
        return edge["distance_km"],edge["travel_min"]
    km = distance(a,b)
    return km,km/dataset["speed_kmh"]*60


def with_lookup(dataset):
    data = dict(dataset)
    data["_matrix_lookup"] = {(r["from_id"],r["to_id"]):r for r in data.get("travel_matrix",[])}
    return data


def baseline_solve(payload, countermeasures):
    data = with_lookup(payload)
    stops = copy.deepcopy(payload["stops"])
    issues = inspect_stops(stops)
    valid_service = [s["service_min"] for s in stops if 0 < s["service_min"] <= 30]
    replacement = median(valid_service) if valid_service else None
    conflicts = {i["stop_id"] for i in issues if i["reason"].startswith("Conflicting")}
    usable,seen = [],set()
    for s in stops:
        sid = s["stop_id"]
        if sid in conflicts: continue
        if countermeasures and sid in seen: continue
        seen.add(sid)
        if s["lat"] is None or s["lon"] is None: continue
        if s["window_end"] is not None and s["window_end"] < s["window_start"]: continue
        if countermeasures and not 0 < s["service_min"] <= 30:
            if replacement is None: continue
            s["service_min"] = replacement
        if s["service_min"] <= 0: continue
        usable.append(s)
    point,point_id,elapsed,order = payload["depot"],"__DEPOT__",0.,[]
    while usable:
        def cost(s):
            _,travel = leg(point_id,s["stop_id"],point,s,data)
            return travel+.35*max(0,elapsed+travel+s["service_min"]-(s["window_end"] if s["window_end"] is not None else math.inf))
        stop = min(usable,key=cost)
        _,travel = leg(point_id,stop["stop_id"],point,stop,data)
        elapsed = max(elapsed+travel,stop["window_start"])+stop["service_min"]
        order.append(stop["stop_id"])
        point,point_id = stop,stop["stop_id"]
        usable.remove(stop)
    _,travel = leg(point_id,"__DEPOT__",point,payload["depot"],data)
    return dict(order=order,planned_duration_min=elapsed+travel,issues=issues,
                model_name="Built-in baseline heuristic")


def validate_result(result, payload):
    if not isinstance(result, dict): raise ValueError("Model must return a dictionary")
    ids = [s["stop_id"] for s in payload["stops"]]
    order = result.get("order")
    if not isinstance(order, list) or any(not isinstance(x, str) or x not in ids for x in order):
        raise ValueError("Model order must contain known stop IDs")
    # Duplicate input may deliberately cause repeat baseline visits, but never invented visits.
    from collections import Counter
    if Counter(order) - Counter(ids): raise ValueError("Model returned more visits than input records")
    value = result.get("planned_duration_min")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("planned_duration_min must be a finite, nonnegative number")
    if not isinstance(result.get("model_name"), str) or not result["model_name"].strip():
        raise ValueError("Model must provide model_name")
    if not isinstance(result.get("issues"), list): raise ValueError("Model must provide an issues list")
    for issue in result["issues"]:
        if not isinstance(issue, dict) or issue.get("stop_id") not in ids or not isinstance(issue.get("reason"), str):
            raise ValueError("Each issue needs a known stop_id and reason string")
    return result


def predict(dataset, countermeasures):
    keys = ("schema_version","route_id","depot","stops","speed_kmh","travel_matrix")
    payload = {k:copy.deepcopy(dataset[k]) for k in keys}
    spec = os.environ.get("ROUTECHECK_MODEL", "")
    if dataset.get("planner") == "baseline":
        result = baseline_solve(payload,countermeasures)
    elif spec:
        module,name = spec.split(":",1)
        result = getattr(importlib.import_module(module),name)(copy.deepcopy(payload),countermeasures)
    else:
        from project_paths import MODEL_DIR
        if (MODEL_DIR/"neural_policy.npz").is_file():
            from neural_policy import solve
            result = solve(copy.deepcopy(payload),countermeasures)
        else:
            result = baseline_solve(payload,countermeasures)
    return validate_result(result,payload)


def evaluate(dataset, prediction):
    """Estimated replay from uploaded inputs, never actual delivery outcomes.

    Full-route performance is unavailable if source quality remains uncertain.
    Scheduled visit counts are still known from a model's order.
    """
    data = with_lookup(dataset)
    reference = {s["stop_id"]:s for s in dataset["stops"]}
    visited = set(prediction["order"])
    issues = dataset["issues"]
    uncertain = any(i["category"] != "duplicate_stops" or i["reason"].startswith("Conflicting") for i in issues)
    point,point_id,elapsed,km,on_time = dataset["depot"],"__DEPOT__",0.,0.,set()
    distance_known = True
    trace = [dict(sequence=0,stop_id="Depot",**point,arrival_min=0)]
    for i,sid in enumerate(prediction["order"],1):
        s = reference[sid]
        if s["lat"] is None or s["lon"] is None:
            uncertain = True
            continue
        d,t = leg(point_id,sid,point,s,data)
        if d is None: distance_known = False
        else: km += d
        arrival = max(elapsed+t,s["window_start"])
        if s["window_end"] is not None and arrival <= s["window_end"]: on_time.add(sid)
        elapsed = arrival+max(0,s["service_min"])
        trace.append(dict(sequence=i,stop_id=sid,lat=s["lat"],lon=s["lon"],arrival_min=None if uncertain else round(arrival,1)))
        point,point_id = s,sid
    d,t = leg(point_id,"__DEPOT__",point,dataset["depot"],data)
    if d is None: distance_known = False
    else: km += d
    elapsed += t
    trace.append(dict(sequence=len(trace),stop_id="Depot",**dataset["depot"],arrival_min=None if uncertain else round(elapsed,1)))
    windowed = sum(s["window_end"] is not None for s in reference.values())
    return dict(distance_km=None if uncertain or not distance_known else round(km,2),duration_min=None if uncertain else round(elapsed,1),
                on_time_pct=None if uncertain or not windowed else round(100*len(on_time)/windowed,1),
                completed=len(visited),missed=len(reference)-len(visited),
                repeat_visits=len(prediction["order"])-len(visited),planning_error_min=None,trace=trace)


def run_experiment(dataset,countermeasures):
    started=time.perf_counter()
    before = predict(dataset,False)
    before_runtime=time.perf_counter()-started
    after_started=time.perf_counter()
    after = predict(dataset,True) if countermeasures else copy.deepcopy(before)
    after_runtime=time.perf_counter()-after_started
    result = dict(source=dataset["source"],dataset_hash=dataset["dataset_hash"],route_id=dataset["route_id"],
                scenario="uploaded_data",severity=None,countermeasures=countermeasures,compute_seconds=time.perf_counter()-started,before_runtime_seconds=before_runtime,after_runtime_seconds=after_runtime,
                total_stops=len({s["stop_id"] for s in dataset["stops"]}),stop_ids=sorted({s["stop_id"] for s in dataset["stops"]}),issues=dataset["issues"],
                depot=dataset["depot"],speed_kmh=dataset["speed_kmh"],
                travel_basis=("Case travel-time matrix (no road distances)" if dataset["source"] == "case_json" else "Uploaded road matrix") if dataset["travel_matrix"] else "Straight-line estimate",
                model_name=after["model_name"],before=evaluate(dataset,before),after=evaluate(dataset,after),
                before_prediction=before,after_prediction=after)

    if dataset.get("actual_order") is not None:
        actual = dataset["actual_order"]
        actual_edges = set(zip(actual,actual[1:]))
        proposed_edges = set(zip(after["order"],after["order"][1:]))
        result["historical"] = dict(order=actual,replay=evaluate(dataset,dict(order=actual,planned_duration_min=0)),
            adjacency_match_pct=round(100*len(actual_edges & proposed_edges)/len(actual_edges),1) if actual_edges else None)
    return result
