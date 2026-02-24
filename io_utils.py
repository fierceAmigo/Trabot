"""io_utils.py

Production utilities:
- atomic writes (JSON, text, CSV append)
- schema-stable CSV writing with header enforcement
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_write_bytes(path: str, data: bytes) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def atomic_write_text(path: str, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str, obj: Any, *, indent: int = 2, sort_keys: bool = True) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys).encode("utf-8")
    atomic_write_bytes(path, data)


def append_csv_row(path: str, row: Dict[str, Any], columns: List[str]) -> None:
    """Append a single row to a schema-stable CSV.

    - Writes header if file is new/empty.
    - Ensures output order matches `columns`.
    - Missing fields become empty string.
    """
    ensure_dir(os.path.dirname(path) or ".")
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if need_header:
            w.writeheader()
        out = {c: row.get(c, "") for c in columns}
        w.writerow(out)
