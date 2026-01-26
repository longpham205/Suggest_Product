# backend/main.py

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.api.recommend import router as recommend_router
from web.backend.api.cart import router as cart_router
from web.backend.api.user import router as user_router
from web.backend.api.products import router as products_router
from web.backend.api.evaluate import router as evaluate_router
from web.backend.core.exception_handlers import register_exception_handlers


# =====================================================
# Logging config
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend")


# =====================================================
# Lifespan (modern FastAPI)
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info(" Backend starting up...")
    yield
    logger.info(" Backend shutting down...")


# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(
    title="Hybrid Recommendation Backend",
    description="Demo backend for Hybrid Recommendation System with FE integration",
    version="2.0.0",
    lifespan=lifespan,
)


# =====================================================
# Exception Handlers
# =====================================================
register_exception_handlers(app)


# =====================================================
# CORS (configured for FE development)
# =====================================================
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://localhost:5500",
    "https://740b939da229.ngrok-free.app",  # Ngrok backend
    "https://4da1c0a36db9.ngrok-free.app",  # Ngrok frontend
    "*",  # Allow all for ngrok (development only)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Routers
# =====================================================
app.include_router(recommend_router)
app.include_router(cart_router)
app.include_router(user_router)
app.include_router(products_router)
app.include_router(evaluate_router)


# =====================================================
# Health check
# =====================================================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "hybrid-recommendation-backend",
        "version": "2.0.0",
    }


# =====================================================
# Serve Frontend Static Files (for single ngrok tunnel)
# =====================================================
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Get frontend path relative to backend
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"

# Mount static files
app.mount("/scripts", StaticFiles(directory=FRONTEND_PATH / "scripts"), name="scripts")
app.mount("/statics", StaticFiles(directory=FRONTEND_PATH / "statics"), name="statics")
app.mount("/templates", StaticFiles(directory=FRONTEND_PATH / "templates", html=True), name="templates")

# Redirect root to index.html
from fastapi.responses import RedirectResponse

@app.get("/")
def redirect_to_frontend():
    return RedirectResponse(url="/templates/index.html")

