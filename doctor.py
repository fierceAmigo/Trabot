"""doctor.py

Quick environment and schema sanity checks.
Usage:
  python doctor.py
"""

from __future__ import annotations

import os
import sys
import importlib
from trabot_schema import RECO_SCHEMA_VERSION, RECO_COLUMNS

def main():
    print("Trabot Doctor\n----------------")
    print("Python:", sys.version.split()[0])
    print("Schema version:", RECO_SCHEMA_VERSION)
    print("Reco columns:", len(RECO_COLUMNS))
    required_env = ["KITE_API_KEY", "KITE_ACCESS_TOKEN"]
    for k in required_env:
        print(f"Env {k}:", "SET" if os.getenv(k) else "MISSING")
    # Try imports
    mods = ["kite_client","market_data","scan_options_v22","reco_analyzer_v22"]
    for m in mods:
        try:
            importlib.import_module(m)
            print("Import OK:", m)
        except Exception as e:
            print("Import FAIL:", m, "->", e)

if __name__ == "__main__":
    main()
