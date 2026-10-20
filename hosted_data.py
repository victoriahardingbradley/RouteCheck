"""Read the small, real-data deployment excerpt without Desktop file paths."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def bundled_routes():
    return json.loads((ROOT / 'cloud_bundle.json').read_text())['routes']


def load_bundled_route(route_id):
    entry = next((r for r in bundled_routes() if r['route_id'] == route_id), None)
    if entry is None:
        raise ValueError('Route is not in the deployment excerpt')
    content = (ROOT / 'bundled_data' / entry['file']).read_bytes()
    if hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise ValueError('Bundled route checksum failed')
    return json.loads(content)
