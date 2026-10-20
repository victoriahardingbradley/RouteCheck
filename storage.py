"""SQLite locally; transactional PostgreSQL storage when a database URL is configured."""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from project_paths import DB_PATH
from contextlib import contextmanager
import re


class StorageError(RuntimeError):
    pass


def cloud_mode():
    setting = os.environ.get("ROUTECHECK_CLOUD")
    return setting == "1" if setting is not None else Path(__file__).with_name("cloud_bundle.json").is_file()


def storage_ready():
    return bool(os.environ.get("ROUTECHECK_DATABASE_URL")) or not cloud_mode()


class PostgresAdapter:
    def __init__(self, db): self.db = db
    def execute(self, sql, parameters=()):
        for table in ("runs", "reviews", "artifacts", "model_batches"):
            sql = re.sub(r"\b" + table + r"\b", "routecheck_" + table, sql)
        return self.db.execute(sql.replace("?", "%s"), parameters)


@contextmanager
def connect():
    url = os.environ.get("ROUTECHECK_DATABASE_URL")
    if not storage_ready():
        raise StorageError("Add ROUTECHECK_DATABASE_URL in Streamlit Secrets to save comparisons and reviews.")
    if url:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError:
            raise StorageError("Install the project's requirements to enable PostgreSQL storage.") from None
        try:
            with psycopg.connect(url, row_factory=dict_row, connect_timeout=10) as raw:
                db = PostgresAdapter(raw)
                # Serialize initial schema creation across simultaneous app sessions.
                db.execute("SELECT pg_advisory_xact_lock(742039105)")
                db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT, result TEXT, status TEXT)")
                db.execute("CREATE TABLE IF NOT EXISTS reviews (id BIGSERIAL PRIMARY KEY, run_id TEXT REFERENCES routecheck_runs(id), created_at TEXT, reviewer TEXT, decision TEXT, note TEXT, override_order_json TEXT, human_review_seconds DOUBLE PRECISION)")
                db.execute("CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, content BYTEA NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS model_batches (id TEXT PRIMARY KEY, created_at TEXT, document TEXT)")
                yield db
        except psycopg.Error:
            raise StorageError("Database unavailable. Check the Neon connection string and database status. Nothing was saved.") from None
    else:
        db = _sqlite_connect()
        try:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, content BLOB NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS model_batches (id TEXT PRIMARY KEY, created_at TEXT, document TEXT)")
                yield db
        finally:
            db.close()


def read_artifact(reference):
    if reference.startswith("artifact://"):
        with connect() as db:
            row = db.execute("SELECT content FROM artifacts WHERE id=?", (reference[11:],)).fetchone()
            if row is None: raise StorageError("Trial file not found in the database.")
            return bytes(row["content"])
    return Path(reference).read_bytes()



def _sqlite_connect():
    path = Path(os.environ.get("ROUTECHECK_DB", DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT, result TEXT, status TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, run_id TEXT, created_at TEXT, reviewer TEXT, decision TEXT, note TEXT)")
    columns = {r["name"] for r in db.execute("PRAGMA table_info(reviews)")}
    for name, kind in [("override_order_json", "TEXT"), ("human_review_seconds", "REAL")]:
        if name not in columns:
            db.execute(f"ALTER TABLE reviews ADD COLUMN {name} {kind}")
    return db


def save_run(result):
    run_id = uuid.uuid4().hex[:12]
    with connect() as db:
        # The paired trial JSON, summaries, change log and run commit together.
        if os.environ.get("ROUTECHECK_DATABASE_URL"):
            references = []
            for filename in result.get("trial_files", []):
                path = Path(filename)
                for artifact in (path, path.with_suffix(".md")):
                    key = run_id + "/" + artifact.name
                    db.execute("INSERT INTO artifacts VALUES (?,?)", (key, artifact.read_bytes()))
                references.append("artifact://" + run_id + "/" + path.name)
                record = json.loads(path.read_text())
                log = record.get("modification_log")
                if log and Path(log).name == log and (path.parent / log).is_file():
                    key = run_id + "/" + log
                    db.execute("INSERT INTO artifacts VALUES (?,?) ON CONFLICT (id) DO NOTHING", (key, (path.parent / log).read_bytes()))
            result["trial_files"] = references
        db.execute("INSERT INTO runs VALUES (?,?,?,?)", (run_id, datetime.now(timezone.utc).isoformat(), json.dumps(result), "Pending"))
    return run_id


def load_runs():
    if not storage_ready(): return []
    with connect() as db:
        return [dict(r, result=json.loads(r["result"])) for r in db.execute("SELECT * FROM runs ORDER BY created_at DESC")]


def review_run(run_id, reviewer, decision, note, override_order=None, human_review_seconds=None):
    if decision not in ("Approved", "Needs revision", "Rejected"):
        raise ValueError("Invalid review decision")
    if not reviewer.strip() or not note.strip(): raise ValueError("Reviewer and rationale are required")
    if human_review_seconds is not None:
        import math
        if not isinstance(human_review_seconds, (int, float)) or not math.isfinite(human_review_seconds) or human_review_seconds < 0:
            raise ValueError("Review duration must be finite and nonnegative")
    with connect() as db:
        saved = db.execute("SELECT result FROM runs WHERE id=?", (run_id,)).fetchone()
        if saved is None: raise ValueError("Run not found")
        if override_order is not None:
            result = json.loads(saved["result"])
            expected = set(result.get("stop_ids", result.get("after_prediction", {}).get("order", [])))
            if not isinstance(override_order, list) or len(override_order) != len(expected) or set(override_order) != expected:
                raise ValueError("An override must contain every route stop exactly once; exclude the depot")
        cursor = db.execute("UPDATE runs SET status=? WHERE id=?", (decision, run_id))
        if cursor.rowcount != 1: raise ValueError("Run not found")
        db.execute("INSERT INTO reviews(run_id,created_at,reviewer,decision,note,override_order_json,human_review_seconds) VALUES (?,?,?,?,?,?,?)",
                   (run_id, datetime.now(timezone.utc).isoformat(), reviewer.strip(), decision, note.strip(),
                    json.dumps(override_order) if override_order is not None else None, human_review_seconds))


def load_reviews():
    if not storage_ready(): return []
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM reviews ORDER BY id DESC")]


def remove_synthetic_runs():
    """Retire only old generated demo runs and their linked reviews."""
    if not storage_ready(): return
    with connect() as db:
        demo_ids = [r["id"] for r in db.execute("SELECT id,result FROM runs") if json.loads(r["result"]).get("source") == "synthetic_demo"]
        for run_id in demo_ids:
            db.execute("DELETE FROM reviews WHERE run_id=?", (run_id,))
            db.execute("DELETE FROM runs WHERE id=?", (run_id,))


def save_model_batch(batch):
    """Persist the original measured rows and an individual file for every trial."""
    batch_id=uuid.uuid4().hex[:12]
    created=datetime.now(timezone.utc).isoformat()
    document={k:v for k,v in batch.items() if k!='archive'}
    artifacts=[(batch_id+'/pilot_results.zip',batch['archive'])]
    for number,row in enumerate(batch['rows'],1):
        name=f'trial_{number:05d}'
        artifacts.append((batch_id+'/'+name+'.json',json.dumps(dict(source='teammate_pilot_reported_metrics',trial=row,source_sha256=batch['source_sha256'],notice=batch['notice']),indent=2).encode()))
        summary=(f"# {name}\n\nRoute: {row['route_id']}\n\nCondition: {row['condition']}; scenario: {row['scenario']}; method: {row['method']}\n\n"
                 f"Seed: {row['seed']}; time point: {row['time_point']}; valid: {row['valid']}\n\n"
                 f"Reported duration: {row['duration_seconds']} seconds.\n\n"+batch['notice'])
        artifacts.append((batch_id+'/'+name+'.md',summary.encode()))
    with connect() as db:
        if isinstance(db,PostgresAdapter):
            db.db.cursor().executemany('INSERT INTO routecheck_artifacts VALUES (%s,%s)',artifacts)
        else:
            db.executemany('INSERT INTO artifacts VALUES (?,?)',artifacts)
        db.execute('INSERT INTO model_batches VALUES (?,?,?)',(batch_id,created,json.dumps(document)))
    return batch_id


def load_model_batches():
    if not storage_ready(): return []
    with connect() as db:
        return [dict(id=r['id'],created_at=r['created_at'],**json.loads(r['document'])) for r in db.execute('SELECT * FROM model_batches ORDER BY created_at DESC')]
