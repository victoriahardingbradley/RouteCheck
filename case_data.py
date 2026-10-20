"""Read the original case JSON files locally, one route at a time.

The incremental top-level decoder supports the dataset's NaN values without
loading the multi-gigabyte travel-time file into memory.
"""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from data_input import inspect_stops

FILES = ('route_data.json','package_data.json','travel_times.json','actual_sequences.json')


def iter_routes(path, chunk_size=65536):
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8-sig') as stream:
        buffer, ended = '', False
        def fill():
            nonlocal buffer,ended
            chunk = stream.read(chunk_size)
            buffer += chunk
            ended = not chunk
            if len(buffer) > 50_000_000: raise ValueError('A single JSON route exceeds the 50 MB limit')
        def token():
            nonlocal buffer
            while True:
                buffer = buffer.lstrip()
                if buffer: return buffer[0]
                if ended: raise ValueError('Unexpected end of JSON file')
                fill()
        def consume(char):
            nonlocal buffer
            if token() != char: raise ValueError(f'Expected {char} in {path}')
            buffer = buffer[1:]
        def value():
            nonlocal buffer
            token()
            while True:
                try: item,end = decoder.raw_decode(buffer)
                except json.JSONDecodeError as error:
                    if ended: raise ValueError(f'Malformed JSON in {path}: {error}') from error
                    fill()
                else:
                    buffer = buffer[end:]
                    return item
        consume('{')
        if token() == '}': return
        while True:
            key = value()
            if not isinstance(key,str): raise ValueError('Route keys must be strings')
            consume(':')
            item = value()
            if not isinstance(item,dict): raise ValueError('Route entries must be JSON objects')
            yield key,item
            if token() == '}': return
            consume(',')


def folder_signature(folder):
    root = Path(folder).expanduser().resolve()
    for name in FILES[:3]:
        if not (root/name).is_file(): raise ValueError(f'Missing file: {name}')
    return tuple((name,(root/name).stat().st_size,(root/name).stat().st_mtime_ns) for name in FILES if (root/name).is_file())


def catalog(folder):
    folder_signature(folder)
    return [dict(route_id=key,station=r.get('station_code',''),date=r.get('date_YYYY_MM_DD',''),
                 stops=sum(s.get('type')=='Dropoff' for s in r.get('stops',{}).values()))
            for key,r in iter_routes(Path(folder)/'route_data.json')]


def find_route(path, route_id):
    for key,item in iter_routes(path):
        if key == route_id: return item
    raise ValueError(f'Route {route_id} not found in {Path(path).name}')


def number(value, label, allow_missing=False):
    if value is None or (isinstance(value,float) and math.isnan(value)):
        if allow_missing: return None
        raise ValueError(f'Missing {label}')
    if isinstance(value,bool): raise ValueError(f'Invalid {label}')
    try: value = float(value)
    except (ValueError,TypeError): raise ValueError(f'Invalid {label}') from None
    if not math.isfinite(value): raise ValueError(f'Nonfinite {label}')
    return value


def timestamp(value):
    if value is None or (isinstance(value,float) and math.isnan(value)): return None
    return datetime.fromisoformat(value)


def load_case_route(folder, route_id):
    root = Path(folder).expanduser().resolve()
    signature = folder_signature(root)
    route = find_route(root/'route_data.json',route_id)
    packages = find_route(root/'package_data.json',route_id)
    travel = find_route(root/'travel_times.json',route_id)
    departure = datetime.fromisoformat(route['date_YYYY_MM_DD']+' '+route['departure_time_utc'])
    stations = [(sid,s) for sid,s in route['stops'].items() if s.get('type')=='Station']
    if len(stations) != 1: raise ValueError('Route must have exactly one Station stop')
    depot_id,depot_stop = stations[0]
    depot = dict(lat=number(depot_stop.get('lat'),'depot latitude'),lon=number(depot_stop.get('lng'),'depot longitude'))
    if not -90 <= depot['lat'] <= 90 or not -180 <= depot['lon'] <= 180: raise ValueError('Depot coordinates out of range')
    stops,package_count = [],0
    for sid,s in route['stops'].items():
        if s.get('type') != 'Dropoff': continue
        rows = packages.get(sid,{})
        if not rows: raise ValueError(f'No package data for stop {sid}')
        times,starts,ends = [],[],[]
        for package in rows.values():
            duration = number(package.get('planned_service_time_seconds'),f'service duration at {sid}')
            if duration < 0: raise ValueError(f'Negative service duration at {sid}')
            times.append(duration/60)
            window = package.get('time_window',{})
            start,end = timestamp(window.get('start_time_utc')),timestamp(window.get('end_time_utc'))
            if (start is None) != (end is None): raise ValueError(f'Incomplete time window at {sid}')
            if start is not None:
                starts.append((start-departure).total_seconds()/60)
                ends.append((end-departure).total_seconds()/60)
        lat,lon = number(s.get('lat'),f'latitude at {sid}',True),number(s.get('lng'),f'longitude at {sid}',True)
        if lat is not None and not -90 <= lat <= 90: raise ValueError(f'Invalid latitude at {sid}')
        if lon is not None and not -180 <= lon <= 180: raise ValueError(f'Invalid longitude at {sid}')
        # Conservative stop-level intersection: a visit must satisfy every package window.
        stops.append(dict(stop_id=sid,lat=lat,lon=lon,window_start=max(starts) if starts else 0.,
                          window_end=min(ends) if ends else None,service_min=sum(times),
                          zone_id=s.get('zone_id') if isinstance(s.get('zone_id'),str) else None,
                          package_count=len(rows)))
        package_count += len(rows)
    if not stops: raise ValueError('Route has no delivery stops')
    ids = [depot_id]+[s['stop_id'] for s in stops]
    matrix = []
    for a in ids:
        for b in ids:
            if a == b: continue
            try: seconds = number(travel[a][b],f'travel time {a} → {b}')
            except KeyError: raise ValueError(f'Missing travel edge {a} → {b}') from None
            if seconds < 0: raise ValueError('Negative travel time')
            matrix.append(dict(from_id='__DEPOT__' if a==depot_id else a,to_id='__DEPOT__' if b==depot_id else b,
                               travel_min=seconds/60,distance_km=None))
    actual_order = None
    if (root/'actual_sequences.json').is_file():
        actual = find_route(root/'actual_sequences.json',route_id).get('actual',{})
        if set(actual) != set(ids) or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in actual.values()) or set(actual.values()) != set(range(len(ids))):
            raise ValueError('Actual sequence must give every stop a unique contiguous nonnegative position')
        if actual[depot_id] != 0: raise ValueError('Actual sequence must start at the Station stop')
        actual_order = [sid for sid in sorted(actual,key=actual.get) if sid != depot_id]
    observed = dict(route_id=route_id,depot=depot,stops=stops,travel_matrix=matrix)
    return dict(schema_version=1,source='case_json',dataset_hash=hashlib.sha256(json.dumps(observed,sort_keys=True).encode()).hexdigest(),
                route_id=route_id,depot=depot,speed_kmh=25.,stops=stops,issues=inspect_stops(stops),travel_matrix=matrix,
                actual_order=actual_order,station=route.get('station_code'),route_date=route.get('date_YYYY_MM_DD'),
                package_count=package_count,source_signature=signature,
                notes=['Package service seconds are summed per stop and converted to minutes.',
                       'No package time window means unconstrained, not missing data.',
                       'Multiple package windows use a conservative stop-level intersection.',
                       'Travel times are supplied seconds converted to minutes; road distance is not provided.',
                       'Route quality labels, scan outcomes and actual sequences are excluded from planner inputs.'])
