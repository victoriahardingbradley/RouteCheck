"""Copy this standard-library-only helper next to your model inference code."""
import json

def write_comparison(path, route_id, before_order, after_order, model_name, countermeasures=True, metadata=None):
    """Call this after inference to export one web-uploadable JSON; no Streamlit dependency."""
    from pathlib import Path
    document=dict(schema_version=1, route_id=route_id, countermeasures=countermeasures,
                  before=dict(order=list(before_order), model_name=model_name),
                  after=dict(order=list(after_order), model_name=model_name), metadata=metadata or {})
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False))
    return Path(path)
