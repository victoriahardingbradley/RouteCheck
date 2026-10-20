# Cloud verification — 2026-10-20

- 78 automated tests passed, including real PostgreSQL 16 integration tests.
- Verified SQLite compatibility, PostgreSQL run/review persistence, full paired JSON/Markdown artifacts and the exact modification log after deleting temporary local files.
- Verified failed artifact writes roll back the entire database transaction; invalid reviews leave status and audit history unchanged; connection errors omit credentials.
- Streamlit AppTest: read-only preview without credentials, real 141-stop bundled route load, neural comparison with PostgreSQL, fresh-session saved-evidence downloads, no duplicate trial from rerendering.
- Browser: cloud preview loads the real route, hides Desktop folder selection, and disables saving without credentials.
- All four bundle hashes match their manifest; normalized dataset hashes match the original selected_routes.csv test split.
- Original 576 historical trial records and readable summaries retained; no retraining or new result claims.

The database tests used a disposable local PostgreSQL server, not Neon. Neon account setup, TLS connection with the owner's real credentials, and a live Streamlit deployment remain deployment acceptance checks in DEPLOYMENT.md.

Reproduce unit and storage tests:

    python -m pip install -r requirements-dev.txt
    python -m pytest tests -q

PostgreSQL-specific tests skip unless ROUTECHECK_TEST_DATABASE_URL is set. Set it only to an explicitly disposable database: integration fixtures clear the routecheck test tables. Never point this test setting at the team's production Neon database.
