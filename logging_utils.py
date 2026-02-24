"""logging_utils.py

Structured logging (JSON lines) for production runs.
Falls back to plain logs if JSON formatting fails.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        # user extras
        for k, v in getattr(record, "__dict__", {}).items():
            if k.startswith("_") or k in ("msg","args","levelname","levelno","name","pathname","filename","module",
                                           "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
                                           "relativeCreated","thread","threadName","processName","process"):
                continue
            # keep scalars small
            try:
                json.dumps({k: v})
                base[k] = v
            except Exception:
                base[k] = str(v)
        try:
            return json.dumps(base, ensure_ascii=False)
        except Exception:
            return f"{base['ts']} {base['level']} {base['name']}: {base['msg']}"


def init_logging() -> None:
    level = os.getenv("TRABOT_LOG_LEVEL", "INFO").upper().strip() or "INFO"
    json_mode = os.getenv("TRABOT_LOG_JSON", "1").strip() in ("1","true","yes")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    # clear handlers to avoid duplicates when imported in notebooks
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    if json_mode:
        h.setFormatter(JsonFormatter())
    else:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(h)
