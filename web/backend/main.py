# backend/main.py

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.recommend import router as recommend_router
from api.cart import router as cart_router
from api.user import router as user_router


# =====================================================
# Logging config
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend")


# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(
    title="Hybrid Recommendation Backend",
    description="Demo backend for Hybrid Recommendation System",
    version="1.0.0",
)


# =====================================================
# CORS (cho FE demo)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only, prod thì fix domain
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


# =====================================================
# Health check
# =====================================================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "hybrid-recommendation-backend",
    }


# =====================================================
# Startup / Shutdown hooks
# =====================================================
@app.on_event("startup")
def on_startup():
    logger.info("🚀 Backend started successfully")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("🛑 Backend shutting down")
