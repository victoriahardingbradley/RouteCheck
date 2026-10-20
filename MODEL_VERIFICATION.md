# Model integration verification — October 20, 2026

Source: user-supplied caseathon-main.zip for private BearTyree/caseathon. No private repository was published or pushed.

## Changes

- Native metrics.csv / pilot-output ZIP import, preserving route/scenario/seed/time/condition/method keys, seconds, invalid outputs and provenance.
- Route-level averages before time-series means. All three methods shown, separately from the original neural-policy study.
- Every imported pilot trial has a JSON and readable Markdown artifact. Source results archive and manifest remain downloadable after restart in SQLite or PostgreSQL.
- One-file standard-library exporter for generic orders; exact integer-index-to-ID exporter added to a copy of the teammate's evaluate_holdout loop. No training, planner, reference-world or hyperparameter changes.
- Explicit paired-order import with route/stop validation, independent historical replay, source hash and original export retained; usable in existing human review.
- Live callable hook repaired for matrix routes and accepted from server-side Streamlit Secrets. Observed payloads exclude clean targets, actual sequences and defect labels.

## Tests actually run

78 website tests passed with a disposable PostgreSQL 16 database, including durable trial/archive retrieval, atomic rollback, CSV/ZIP validation, imported-result rendering, and a configured live callable receiving both toggle states. The same suite works locally with PostgreSQL-specific tests skipped unless ROUTECHECK_TEST_DATABASE_URL is configured.

10 teammate handoff tests passed (9 original + 1 added). The added test invokes the actual evaluate_holdout loop on a tiny synthetic route with an explicitly fake predictor, verifies 72 metric rows and 24 exported paired comparisons, and checks depot removal and string IDs. This is a code-integration test, not trained-model evidence.

No checkpoint or measured pilot outputs were included in the source ZIP. No new CatBoost training, full pilot, model performance claim, production Neon connection or public deployment was performed. The supplied repository is CatBoost repair plus a fixed heuristic; it is not neural combinatorial optimization. Existing neural-study results were preserved and remain separately labeled.

## Files to send your teammate

Use Teammate-model-integration.zip. Its ROUTECHECK_HANDOFF.md lists the two code changes and the existing command to run. Once the teammate sends outputs/pilot/metrics.csv (or a results ZIP), upload that directly in RouteCheck. No model inference package is required in the website for this path.
