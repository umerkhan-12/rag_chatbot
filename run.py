"""Start VeriDesk:  python run.py   then open http://127.0.0.1:8000

Optional: python run.py --ingest path/to/manual.pdf   (index a PDF before the server starts)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", help="PDF to index and activate before starting")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.ingest:
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "ingest.py"), a.ingest])
        if rc != 0:
            sys.exit(rc)
    from app import config

    import uvicorn

    url = f"http://{'127.0.0.1' if config.HOST in ('0.0.0.0', '') else config.HOST}:{config.PORT}"
    print(f"\n  {config.APP_NAME} running at {url}\n")
    if not a.no_browser and os.environ.get("NO_BROWSER") != "1":
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    uvicorn.run("app.server:app", host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
