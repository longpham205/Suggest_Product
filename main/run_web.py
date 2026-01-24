# backend/run_web.py

import uvicorn
import logging
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

HOST = "0.0.0.0"
PORT = 8000
RELOAD = True   # dev mode


def run_backend():
    """
    Entry point to start FastAPI backend
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.getLogger("run_web").info(
        f"Starting backend at http://{HOST}:{PORT}"
    )

    uvicorn.run(
        "web.backend.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level="info",
    )


if __name__ == "__main__":
    run_backend()
