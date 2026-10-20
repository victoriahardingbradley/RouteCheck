"""CSV validation and route loading. No fabricated delivery records."""
import hashlib
import io
import math
import csv

import pandas as pd

COLUMNS = ["route_id", "stop_id", "lat", "lon", "window_start", "window_end", "service_min"]
TEMPLATE = ",".join(COLUMNS) + "\n"
SCENARIOS = {
    "missing_coordinates": ("Missing coordinates", "Stops without a location stay unresolved; upload corrected coordinates before routing them."),
    "duplicate_stops": ("Duplicate stops", "Exact repeated records can be removed. Conflicting records require a source-data correction."),
    "invalid_windows": ("Invalid time windows", "Reversed windows require a source-data correction; the planner skips these stops."),
    "service_time_outliers": ("Service-time outliers", "The optional countermeasure uses the median of durations above zero and up to 30 minutes. Review the assumption."),
}


def inspect_stops(stops):
    issues, seen = [], {}
    for s in stops:
        sid = s["stop_id"]
        if sid in seen:
            same = all(s[k] == seen[sid][k] for k in s)
            issues.append(dict(stop_id=sid, category="duplicate_stops", reason="Exact duplicate record" if same else "Conflicting duplicate; correct the source data"))
        else: seen[sid] = s
        if s["lat"] is None or s["lon"] is None:
            issues.append(dict(stop_id=sid, category="missing_coordinates", reason="Missing coordinates; location cannot be inferred reliably"))
        if s["window_end"] is not None and s["window_end"] < s["window_start"]:
            issues.append(dict(stop_id=sid, category="invalid_windows", reason="Window closes before it opens"))
        if not 0 < s["service_min"] <= 30:
            issues.append(dict(stop_id=sid, category="service_time_outliers", reason="Service duration is zero or above 30 min; verify this value"))
    return issues


def read_stops(raw):
    if len(raw) > 5_000_000: raise ValueError("Stop CSV must be 5 MB or smaller")
    try:
        headers = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        if len({h.strip() for h in headers}) != len(headers):
            raise ValueError("Duplicate column names")
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as error: raise ValueError(f"Cannot read CSV: {error}") from error
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any(): raise ValueError("Duplicate column names")
    missing = set(COLUMNS)-set(frame.columns)
    if missing: raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty: raise ValueError("CSV has headers but no delivery rows")
    if len(frame) > 1000: raise ValueError("This local prototype supports up to 1,000 input rows")
    frame = frame[COLUMNS].copy()
    for col in ["route_id", "stop_id"]:
        frame[col] = frame[col].str.strip()
        if (frame[col] == "").any(): raise ValueError(f"{col} cannot be blank")
        if (frame[col].str.len() > 100).any(): raise ValueError(f"{col} must be at most 100 characters")
    if (frame.stop_id == "__DEPOT__").any(): raise ValueError("__DEPOT__ is reserved for the depot")
    for col in COLUMNS[2:]:
        values = []
        for row, val in enumerate(frame[col], 2):
            if not val.strip() and col in ("lat", "lon", "window_end"):
                values.append(None)
                continue
            if not val.strip() and col == "window_start":
                values.append(0.)
                continue
            try: number = float(val)
            except ValueError: raise ValueError(f"Row {row}: {col} must be numeric") from None
            if not math.isfinite(number): raise ValueError(f"Row {row}: {col} must be finite")
            if col == "lat" and not -90 <= number <= 90: raise ValueError(f"Row {row}: latitude outside -90 to 90")
            if col == "lon" and not -180 <= number <= 180: raise ValueError(f"Row {row}: longitude outside -180 to 180")
            if col in ("window_start", "window_end") and not -10080 <= number <= 10080:
                raise ValueError(f"Row {row}: windows must be minutes from departure, from -10080 to 10080")
            if col == "service_min" and not 0 <= number <= 1440:
                raise ValueError(f"Row {row}: service_min must be from 0 to 1440")
            values.append(number)
        frame[col] = pd.Series(values, dtype=object)
    for route in frame.route_id.unique():
        if frame[frame.route_id == route].stop_id.nunique() > 250:
            raise ValueError("At most 250 unique stops per route are supported")
    return frame


def read_matrix(raw, stop_ids):
    """Optional complete directed road-distance/travel-time matrix for one route."""
    if len(raw) > 10_000_000: raise ValueError("Travel matrix must be 10 MB or smaller")
    frame = pd.read_csv(io.BytesIO(raw),dtype=str,keep_default_na=False)
    required = {"from_id", "to_id", "travel_min", "distance_km"}
    if not required <= set(frame.columns): raise ValueError("Matrix columns: from_id,to_id,travel_min,distance_km")
    ids = set(stop_ids) | {"__DEPOT__"}
    matrix = {}
    for r in frame.to_dict("records"):
        a, b = r["from_id"].strip(), r["to_id"].strip()
        if a not in ids or b not in ids: raise ValueError("Matrix contains an unknown stop ID")
        if (a,b) in matrix: raise ValueError("Duplicate matrix edge")
        try: t,d = float(r["travel_min"]),float(r["distance_km"])
        except ValueError: raise ValueError("Matrix times and distances must be numeric") from None
        if not all(math.isfinite(x) and x >= 0 for x in (t,d)): raise ValueError("Matrix values must be finite and nonnegative")
        if a != b and (t == 0 or d == 0): raise ValueError("Non-diagonal travel times and distances must be positive")
        matrix[a,b] = dict(travel_min=t,distance_km=d)
    needed = {(a,b) for a in ids for b in ids if a != b}
    if not needed <= matrix.keys(): raise ValueError(f"Matrix is missing {len(needed-matrix.keys())} directed edges, including depot legs")
    return [dict(from_id=a,to_id=b,**v) for (a,b),v in matrix.items()]


def make_dataset(frame, route_id, depot_lat, depot_lon, speed_kmh, raw, matrix=None):
    if not -90 <= depot_lat <= 90 or not -180 <= depot_lon <= 180: raise ValueError("Invalid depot coordinates")
    if not 1 <= speed_kmh <= 130: raise ValueError("Speed must be 1–130 km/h")
    stops = frame[frame.route_id == route_id].drop(columns="route_id").to_dict("records")
    if not stops: raise ValueError("Selected route has no stops")
    # Pandas can normalize None to NaN; normalize missing locations for the adapter.
    for s in stops:
        for key in ("lat", "lon"):
            if s[key] is None or pd.isna(s[key]): s[key] = None
    return dict(schema_version=1,source="uploaded_csv",dataset_hash=hashlib.sha256(raw).hexdigest(),
                route_id=route_id,depot=dict(lat=depot_lat,lon=depot_lon),speed_kmh=speed_kmh,
                stops=stops,issues=inspect_stops(stops),travel_matrix=matrix or [])
