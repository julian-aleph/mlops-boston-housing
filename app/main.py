"""API FastAPI de inferencia: expone /predict, /health y /metrics.

La API sirve únicamente predicciones y no depende de MLflow ni DVC en
runtime: el modelo productivo se persiste como joblib autocontenido.
"""

import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from app.model_loader import ModelArtifacts, load_model_artifacts
from app.monitoring import (
    MODEL_INFO,
    MODEL_LOADED,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUESTS_TOTAL,
    PREDICTION_VALUE,
    render_metrics,
)
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Boston Housing Model API")
artifacts: ModelArtifacts = load_model_artifacts()

MODEL_LOADED.set(1)
MODEL_INFO.labels(
    model_name=artifacts.model_name,
    model_stage=artifacts.model_stage,
    target=artifacts.target,
).set(1)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=artifacts.model_name,
        model_stage=artifacts.model_stage,
    )


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Genera una predicción individual usando el modelo productivo cargado."""
    PREDICTION_REQUESTS_TOTAL.inc()
    start = time.perf_counter()

    try:
        payload = request.model_dump()
        # El orden de columnas se toma de metadata["features"] para que el
        # DataFrame coincida con el orden visto durante training.
        dataframe = pd.DataFrame(
            [{feature: payload[feature] for feature in artifacts.features}]
        )
        prediction = float(artifacts.model.predict(dataframe)[0])
    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        PREDICTION_LATENCY_SECONDS.observe(time.perf_counter() - start)

    PREDICTION_VALUE.observe(prediction)
    logger.info("Prediction generated")

    return PredictionResponse(
        prediction=prediction,
        model_name=artifacts.model_name,
        model_stage=artifacts.model_stage,
        target=artifacts.target,
        features_used=artifacts.features,
    )
