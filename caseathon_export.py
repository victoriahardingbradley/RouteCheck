"""Export BearTyree/caseathon's integer-indexed orders; standard library only.

Copy into src/caseathon/routecheck_export.py. It does not train or modify a model.
"""
import json
from pathlib import Path


def export_trial(output_dir, route, orders, outcomes, scenario, condition, seed, time_point):
    """Call after all three methods have populated orders/outcomes in evaluate_holdout."""
    def delivery_ids(method):
        order=orders[method]
        if (len(order)!=len(route.stop_ids)+1 or order[0]!=route.depot or order[-1]!=route.depot
                or set(order[:-1])!=set(range(len(route.stop_ids)))):
            raise ValueError('Invalid model order: expected every stop and depot return')
        return [route.stop_ids[int(index)] for index in order[1:-1]]
    metadata=dict(source='BearTyree/caseathon',scenario=scenario,condition=condition,
                  seed=int(seed),time_point=int(time_point),
                  notice='Reported outcomes use the teammate synthetic current reference, not historical replay.',
                  reported_outcomes=outcomes,rule_order=delivery_ids('rule'))
    document=dict(schema_version=1,route_id=route.route_id,countermeasures=True,
                  before=dict(model_name='Teammate fixed planner / raw reports',order=delivery_ids('raw')),
                  after=dict(model_name='Teammate fixed planner / CatBoost repair',order=delivery_ids('catboost')),
                  metadata=metadata)
    directory=Path(output_dir)/'routecheck_trials';directory.mkdir(parents=True,exist_ok=True)
    # Hash route/scenario identifiers rather than treating them as file paths.
    import hashlib
    key=hashlib.sha256(json.dumps([route.route_id,scenario,condition,int(seed),int(time_point)]).encode()).hexdigest()[:20]
    path=directory/('comparison_'+key+'.json')
    path.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    return path
