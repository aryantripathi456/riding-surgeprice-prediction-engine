"""FastAPI application for the Dynamic Pricing Engine.

Provides REST API endpoints for price predictions, health checks, and model info.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RideRequest,
)
from src.models.predict import PricePredictor
from src.utils import load_config

logger = logging.getLogger(__name__)

# Load config eagerly
_config = load_config()

# Global state
_predictor: PricePredictor = None


def get_predictor() -> PricePredictor:
    """Get or initialize the predictor singleton."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = PricePredictor(config=_config)
            logger.info("Model loaded successfully from %s", _config["paths"]["model_output"])
        except Exception as e:
            logger.warning("Could not load model: %s", e)
            _predictor = None
    return _predictor


def reset_predictor():
    """Reset the predictor (useful for testing)."""
    global _predictor
    _predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    get_predictor()
    yield


app = FastAPI(
    title=_config["api"]["title"],
    version=_config["api"]["version"],
    description="Real-time ride price prediction API using machine learning",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version=_config["api"]["version"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: RideRequest):
    """Predict ride price multiplier based on current conditions.

    Accepts ride request data and returns the predicted surge multiplier,
    estimated final price, and confidence level.
    """
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        data = request.model_dump()
        result = predictor.predict(data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return information about the loaded model."""
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    # Load metrics
    metrics_path = Path(_config["paths"]["metrics"])
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return ModelInfoResponse(
        model_name=metrics.get("best_model", "unknown"),
        model_type=type(predictor.model).__name__,
        features=predictor.feature_names or [],
        metrics=metrics,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=_config["api"]["host"],
        port=_config["api"]["port"],
        reload=True,
    )
