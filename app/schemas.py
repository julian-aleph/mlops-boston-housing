from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crim: float
    zn: float
    indus: float
    chas: float
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    b: float
    lstat: float


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_stage: str
    target: str
    features_used: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_stage: str
