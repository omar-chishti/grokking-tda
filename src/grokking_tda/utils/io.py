from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    # Atomic: write a temp file then rename, so readers never see partial JSON.
    file_path = Path(path)
    tmp = file_path.with_name(file_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, file_path)

