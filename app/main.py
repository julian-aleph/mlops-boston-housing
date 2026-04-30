import logging

import pandas as pd
from fastapi import FastAPI

from app.model_loader import ModelArtifacts, load_model_artifacts
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Boston Housing Model API")
artifacts: ModelArtifacts = load_model_artifacts()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=artifacts.model_name,
        model_stage=artifacts.model_stage,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    payload = request.model_dump()
    dataframe = pd.DataFrame(
        [{feature: payload[feature] for feature in artifacts.features}]
    )
    prediction = float(artifacts.model.predict(dataframe)[0])
    logger.info("Prediction generated")

    return PredictionResponse(
        prediction=prediction,
        model_name=artifacts.model_name,
        model_stage=artifacts.model_stage,
        target=artifacts.target,
        features_used=artifacts.features,
    )
