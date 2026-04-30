import json
from pathlib import Path

from train import REQUIRED_METADATA_FIELDS, build_candidates, validate_metadata_fields


def test_feature_schema_contains_features_and_target() -> None:
    schema = json.loads(
        Path("data/processed/feature_schema.json").read_text(encoding="utf-8")
    )

    assert schema["features"]
    assert schema["target"] == "medv"


def test_model_candidates_and_pipeline_steps() -> None:
    candidates = build_candidates(
        preprocessing_config={"lower_quantile": 0.01, "upper_quantile": 0.99},
        random_state=42,
        n_jobs=1,
    )

    assert set(candidates) == {
        "ridge",
        "elastic_net",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert list(candidates["ridge"][0].named_steps) == [
        "imputer",
        "clipper",
        "scaler",
        "model",
    ]
    assert list(candidates["elastic_net"][0].named_steps) == [
        "imputer",
        "clipper",
        "scaler",
        "model",
    ]
    assert list(candidates["random_forest"][0].named_steps) == [
        "imputer",
        "clipper",
        "model",
    ]
    assert list(candidates["hist_gradient_boosting"][0].named_steps) == [
        "imputer",
        "clipper",
        "model",
    ]


def test_metadata_required_fields_are_validated() -> None:
    metadata = {field: "value" for field in REQUIRED_METADATA_FIELDS}

    validate_metadata_fields(metadata)
