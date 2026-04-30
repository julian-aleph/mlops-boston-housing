import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate staging regression model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must be a dictionary.")

    for section in (
        "features",
        "models",
        "evaluation",
        "feature_importance",
        "training",
    ):
        if section not in config:
            raise ValueError(f"Config file must include a {section} section.")

    return config


def load_evaluation_data(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_config = config["features"]
    processed_dir = Path(feature_config["processed_dir"])
    dataset_path = processed_dir / feature_config["processed_filename"]
    schema_path = processed_dir / feature_config["schema_filename"]
    indices_path = (
        Path(config["evaluation"]["metrics_dir"])
        / config["evaluation"]["test_indices_filename"]
    )

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}.")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}.")
    if not indices_path.exists():
        raise FileNotFoundError(f"Test indices not found: {indices_path}.")

    data = pd.read_parquet(dataset_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    test_indices = json.loads(indices_path.read_text(encoding="utf-8"))[
        "test_indices"
    ]
    features = list(schema["features"])
    target = schema["target"]

    return data.loc[test_indices, features], data.loc[test_indices, target], features


def compute_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "rmse": float(root_mean_squared_error(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    model_path = (
        Path(config["models"]["staging_dir"]) / config["models"]["model_filename"]
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Staging model not found: {model_path}.")

    model = joblib.load(model_path)
    x_test, y_test, features = load_evaluation_data(config)
    metrics = compute_metrics(model, x_test, y_test)
    logger.info("Evaluation metrics: %s", metrics)

    importance_config = config["feature_importance"]
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=int(importance_config["n_repeats"]),
        random_state=int(importance_config["random_state"]),
        scoring=config["training"]["scoring"],
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance["rank"] = range(1, len(importance) + 1)

    output_dir = Path(importance_config["output_dir"])
    importance_path = output_dir / importance_config["filename"]
    selected_path = output_dir / importance_config["selected_features_filename"]
    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(importance_path, index=False)

    threshold = float(importance_config["threshold"])
    selected_features = importance.loc[
        importance["importance_mean"] > threshold,
        "feature",
    ].tolist()
    rejected_features = [
        feature for feature in features if feature not in selected_features
    ]
    write_json(
        {
            "selection_method": "permutation_importance",
            "threshold": threshold,
            "selected_features": selected_features,
            "rejected_features": rejected_features,
        },
        selected_path,
    )
    logger.info("Feature importance saved")

    return importance_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
