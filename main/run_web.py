# backend/run_web.py

import uvicorn
import logging

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
        "main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level="info",
    )


if __name__ == "__main__":
    run_backend()
