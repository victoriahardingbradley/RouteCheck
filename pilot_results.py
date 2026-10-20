"""Read BearTyree/caseathon's actual pilot outputs; no model execution or mock API."""
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
from zipfile import ZipFile, BadZipFile, ZIP_DEFLATED
import pandas as pd

KEYS = ['route_id','scenario','seed','time_point','condition','method']
ALLOWED = {'metrics.csv','manifest.json','selected_routes.json','resolved_config.json',
           'route_examples.json','model_info.json','paired_effects.csv','aggregate_metrics.csv',
           'timepoint_metrics.csv','summary.md','injection_logs.csv','reference_metrics.csv'}


def read_pilot(raw, filename):
    if len(raw)>50_000_000: raise ValueError('Results upload must be 50 MB or smaller.')
    files={}
    if filename.lower().endswith('.zip'):
        try:
            with ZipFile(io.BytesIO(raw)) as archive:
                members=[x for x in archive.infolist() if not x.is_dir() and PurePosixPath(x.filename).name in ALLOWED]
                if sum(x.file_size for x in members)>150_000_000:
                    raise ValueError('Expanded result files exceed 150 MB.')
                for member in members:
                    name=PurePosixPath(member.filename).name
                    if name in files: raise ValueError('Upload one output folder at a time; duplicate '+name)
                    files[name]=archive.read(member)
        except BadZipFile:
            raise ValueError('Cannot read the ZIP file.') from None
    elif filename.lower().endswith('.csv'):
        files['metrics.csv']=raw
    else:
        raise ValueError('Upload the pilot metrics.csv or a ZIP of the pilot output folder.')
    if 'metrics.csv' not in files:
        raise ValueError('No metrics.csv found. This may be the source-code ZIP. Run the teammate experiment first, then upload its outputs/pilot folder as a ZIP.')
    try:
        frame=pd.read_csv(io.BytesIO(files['metrics.csv']),dtype={'route_id':str})
    except (ValueError,pd.errors.ParserError,UnicodeDecodeError):
        raise ValueError('Cannot read metrics.csv.') from None
    required=set(KEYS+['date','valid','duration_seconds','drive_seconds','wait_seconds','service_seconds','late_stops','late_packages','package_lateness_seconds','stop_count','package_count'])
    if not required<=set(frame): raise ValueError('Missing pilot columns: '+', '.join(sorted(required-set(frame))))
    if frame.empty or len(frame)>25000: raise ValueError('Supply between 1 and 25,000 measured rows.')
    if frame[KEYS+['date']].isna().any().any(): raise ValueError('Trial identifiers cannot be blank.')
    if frame.duplicated(KEYS).any(): raise ValueError('Duplicate trial keys in metrics.csv.')
    if not set(frame.method)<= {'raw','rule','catboost'}: raise ValueError('Unknown method; expected raw, rule and catboost.')
    if not set(frame.condition)<= {'normal','stale','inaccurate','combined'}: raise ValueError('Unknown pilot condition.')
    flags=frame['valid'].astype(str).str.lower().map({'true':True,'false':False})
    if flags.isna().any(): raise ValueError('valid must contain true or false.')
    frame['valid']=flags
    for column in ['seed','time_point','stop_count','package_count','duration_seconds','drive_seconds','wait_seconds','service_seconds','late_stops','late_packages','package_lateness_seconds']:
        try: frame[column]=pd.to_numeric(frame[column],errors='raise')
        except (ValueError,TypeError): raise ValueError(column+' must be numeric.') from None
        values=frame[column]
        # Invalid route outcomes may be null; their validity remains visible.
        required_rows=frame.valid if column.endswith('_seconds') or column.startswith('late_') else pd.Series(True,index=frame.index)
        if any(not math.isfinite(v) or v<0 for v in values[required_rows]):
            raise ValueError(column+' must be finite and nonnegative for valid trials.')
    for _,block in frame.groupby(KEYS[:-1],dropna=False):
        if set(block.method)!= {'raw','rule','catboost'}:
            raise ValueError('Each paired route/scenario/seed/time/condition needs raw, rule and catboost rows.')
        if 'reference_planner_duration_seconds' in block and block.reference_planner_duration_seconds.nunique(dropna=False)>1:
            raise ValueError('Paired methods disagree on the supplied reference duration.')
    manifest={}
    if 'manifest.json' in files:
        try: manifest=json.loads(files['manifest.json'])
        except ValueError: raise ValueError('Invalid manifest.json.') from None
        if not isinstance(manifest,dict): raise ValueError('manifest.json must contain an object.')
        if manifest.get('result_rows',len(frame))!=len(frame): raise ValueError('Manifest result_rows does not match metrics.csv.')
    # Preserve only recognized result files, never arbitrary code or uploaded secrets.
    out=io.BytesIO()
    with ZipFile(out,'w',ZIP_DEFLATED) as archive:
        for name,content in files.items(): archive.writestr(name,content)
    rows=json.loads(frame.to_json(orient='records'))
    return dict(rows=rows,manifest=manifest,source_sha256=hashlib.sha256(raw).hexdigest(),
                archive=out.getvalue(),route_count=int(frame.route_id.nunique()),
                notice='Imported CatBoost pilot measurements. Current traffic, reports, faults and time points are simulated. Values are reported by the teammate experiment, not independently recomputed by this upload. This upload does not execute a model or verify its checkpoint.')


def pilot_time_summary(rows, scenario, condition):
    frame=pd.DataFrame(rows)
    part=frame[(frame.scenario==scenario)&(frame.condition==condition)&frame.valid].copy()
    if part.empty: return pd.DataFrame()
    part['duration_min']=part.duration_seconds/60
    # First average repeated seeds within each independent route.
    route=part.groupby(['time_point','method','route_id'],as_index=False).duration_min.mean()
    return route.groupby(['time_point','method'],as_index=False).duration_min.mean()
