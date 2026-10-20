"""Read original study artifacts from files or the compact deployment archive."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from project_paths import RESULTS_DIR


def read_study_file(relative):
    key=PurePosixPath(str(relative))
    if key.is_absolute() or '..' in key.parts:
        raise ValueError('Study evidence path must stay inside experiments.')
    root=Path(RESULTS_DIR)
    path=root/str(key)
    if path.is_file(): return path.read_bytes()
    archive=root/'study_evidence.zip'
    if not archive.is_file(): raise FileNotFoundError(str(key))
    try:
        with ZipFile(archive) as z: return z.read(str(key))
    except KeyError:
        raise FileNotFoundError(str(key)) from None
