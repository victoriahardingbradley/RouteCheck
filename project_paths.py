"""Support both a standalone archive and the Desktop caseathon folder layout."""
import os
from pathlib import Path
APP_DIR=Path(__file__).resolve().parent
CASE_ROOT=APP_DIR.parents[1] if APP_DIR.parent.name=='website' else None
TRAINING_DIR=CASE_ROOT/'training'/'routecheck' if CASE_ROOT else APP_DIR
MODEL_DIR=Path(os.environ.get('ROUTECHECK_MODEL_DIR',str(TRAINING_DIR/'models')))
RESULTS_DIR=Path(os.environ.get('ROUTECHECK_RESULTS_DIR',str(CASE_ROOT/'results'/'routecheck' if CASE_ROOT else APP_DIR/'experiments')))
RAW_DIR=Path(os.environ.get('ROUTECHECK_DATA_DIR',str(CASE_ROOT/'data'/'raw' if CASE_ROOT else Path.home()/'Desktop'/'caseathon'/'data'/'raw')))
DB_PATH=CASE_ROOT/'data'/'processed'/'routecheck'/'routecheck.sqlite3' if CASE_ROOT else APP_DIR/'data'/'routecheck.sqlite3'
