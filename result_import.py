"""Import route orders without importing or executing a teammate's training code."""
import hashlib
import json
import math
from model_integration import evaluate, validate_result


def import_comparison(raw, dataset):
    """Read paired inference results. Replay metrics always use loaded reference data."""
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ValueError('Upload valid JSON exported by the model.') from None
    if not isinstance(document, dict) or document.get('route_id') != dataset['route_id']:
        raise ValueError('The results route_id must match the route loaded above.')
    if not isinstance(document.get('countermeasures'), bool):
        raise ValueError('countermeasures must be true or false, describing the exported current plan.')
    if dataset['issues']:
        raise ValueError('Correct source-data flags before comparing imported orders.')
    expected = [s['stop_id'] for s in dataset['stops']]
    predictions = []
    for key in ('before', 'after'):
        candidate = document.get(key)
        if not isinstance(candidate, dict):
            raise ValueError('Include before and after objects with order and model_name.')
        order = candidate.get('order')
        if (not isinstance(order, list) or any(not isinstance(x, str) for x in order)
                or len(order) != len(expected) or set(order) != set(expected)):
            raise ValueError(f'{key}.order must contain every loaded delivery stop exactly once, excluding the depot.')
        name = candidate.get('model_name')
        if not isinstance(name, str) or not name.strip() or name.startswith('REPLACE'):
            raise ValueError(f'{key}.model_name must identify the actual model.')
        metric = evaluate(dataset, {'order': order})
        prediction = dict(order=order, model_name=name, issues=[],
                          planned_duration_min=metric['duration_min'])
        validate_result(prediction, dataset)
        predictions.append((prediction, metric))
    before, after = predictions
    return dict(source='imported_model_results', route_id=dataset['route_id'],
                dataset_hash=dataset['dataset_hash'], total_stops=len(expected), stop_ids=expected,
                scenario='Imported model comparison', severity=0,
                countermeasures=document['countermeasures'], model_name=after[0]['model_name'],
                before_prediction=before[0], after_prediction=after[0], before=before[1], after=after[1],
                before_label='Imported baseline plan', after_label='Imported current plan',
                issues=[], modifications=[], import_sha256=hashlib.sha256(raw).hexdigest(),
                imported_document=document, external_results=True,
                provenance_note='Model orders and countermeasure setting supplied by the exporter. Metrics independently replayed on the loaded historical route; no model inference or defect simulation was executed by this import.')


from model_export import write_comparison
