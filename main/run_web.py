# web/backend/run_web.py
import uvicorn
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

HOST = "127.0.0.1"
PORT = 8000
RELOAD = True   # dev mode


def run_backend():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.info(f"Starting backend at http://{HOST}:{PORT}")
    logging.info(f"Swagger docs at http://{HOST}:{PORT}/docs")

    uvicorn.run(
        "web.backend.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level="info",
    )


if __name__ == "__main__":
    run_backend()
