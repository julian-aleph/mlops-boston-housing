"""Carga el modelo productivo y su metadata para serving en FastAPI."""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import yaml

logger = logging.getLogger(__name__)

REQUIRED_METADATA_FIELDS = {"features", "target", "model_name", "model_stage"}


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    metadata: dict[str, Any]
    model_path: Path
    metadata_path: Path
    features: list[str]
    target: str
    model_name: str
    model_stage: str


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}.")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must be a dictionary.")

    return config


def resolve_artifact_paths() -> tuple[Path, Path]:
    env_model_path = os.getenv("MODEL_PATH")
    env_metadata_path = os.getenv("MODEL_METADATA_PATH")
    if env_model_path and env_metadata_path:
        return Path(env_model_path), Path(env_metadata_path)

    config_path = Path(os.getenv("CONFIG_PATH", "configs/params.yaml"))
    config = read_config(config_path)
    serving_config = config.get("serving")

    if not isinstance(serving_config, dict):
        raise ValueError("Config file must include a serving section.")

    model_path = Path(env_model_path or serving_config["model_path"])
    metadata_path = Path(env_metadata_path or serving_config["metadata_path"])
    return model_path, metadata_path


def validate_metadata(metadata: dict[str, Any]) -> tuple[list[str], str, str, str]:
    missing_fields = REQUIRED_METADATA_FIELDS - set(metadata)
    if missing_fields:
        raise ValueError(f"Missing metadata fields: {sorted(missing_fields)}.")

    features = metadata["features"]
    target = metadata["target"]
    model_name = metadata["model_name"]
    model_stage = metadata["model_stage"]

    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError("Metadata field features must be a list of strings.")
    if not all(isinstance(value, str) for value in (target, model_name, model_stage)):
        raise ValueError(
            "Metadata target, model_name, and model_stage must be strings."
        )

    return features, target, model_name, model_stage


def ensure_src_import_path() -> None:
    # joblib reconstruye QuantileClipper desde src/preprocessing.py; el
    # path debe estar disponible antes del joblib.load.
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def load_model_artifacts() -> ModelArtifacts:
    """Carga el modelo productivo y su metadata desde models/production/."""
    model_path, metadata_path = resolve_artifact_paths()
    if not model_path.exists():
        raise FileNotFoundError(f"Required model file not found: {model_path}.")

    metadata = read_json(metadata_path)
    features, target, model_name, model_stage = validate_metadata(metadata)
    ensure_src_import_path()
    model = joblib.load(model_path)
    logger.info("Production model artifacts loaded")

    return ModelArtifacts(
        model=model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
        features=features,
        target=target,
        model_name=model_name,
        model_stage=model_stage,
    )
