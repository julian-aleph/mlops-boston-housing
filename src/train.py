import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from preprocessing import QuantileClipper
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

REQUIRED_METADATA_FIELDS = {
    "model_name",
    "model_stage",
    "target",
    "features",
    "training_timestamp",
    "test_size",
    "random_state",
    "cv_folds",
    "search_n_iter",
    "best_params",
    "cv_rmse",
    "test_rmse",
    "test_mae",
    "test_r2",
}


@dataclass
class CandidateResult:
    model_name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    cv_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    selected_model: bool = False
    mlflow_run_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train staging regression model.")
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
        "preprocessing",
        "training",
        "models",
        "evaluation",
    ):
        if section not in config:
            raise ValueError(f"Config file must include a {section} section.")

    return config


def load_feature_artifacts(
    feature_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed_dir = Path(feature_config["processed_dir"])
    dataset_path = processed_dir / feature_config["processed_filename"]
    schema_path = processed_dir / feature_config["schema_filename"]

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}.")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}.")

    data = pd.read_parquet(dataset_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return data, schema


def build_candidates(
    preprocessing_config: dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    clipper = QuantileClipper(
        lower_quantile=float(preprocessing_config["lower_quantile"]),
        upper_quantile=float(preprocessing_config["upper_quantile"]),
    )
    linear_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", clipper),
        ("scaler", RobustScaler()),
    ]
    tree_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", clipper),
    ]

    return {
        "ridge": (
            Pipeline([*linear_steps, ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "elastic_net": (
            Pipeline(
                [
                    *linear_steps,
                    ("model", ElasticNet(random_state=random_state, max_iter=10000)),
                ]
            ),
            {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
        ),
        "random_forest": (
            Pipeline(
                [
                    *tree_steps,
                    (
                        "model",
                        RandomForestRegressor(
                            random_state=random_state,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 1.0],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    *tree_steps,
                    (
                        "model",
                        HistGradientBoostingRegressor(random_state=random_state),
                    ),
                ]
            ),
            {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [100, 200],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 0.1],
            },
        ),
    }


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "test_rmse": float(root_mean_squared_error(target, predictions)),
        "test_mae": float(mean_absolute_error(target, predictions)),
        "test_r2": float(r2_score(target, predictions)),
    }


def get_search_iterations(
    param_space: dict[str, list[Any]],
    configured_n_iter: int,
) -> int:
    search_space_size = 1
    for values in param_space.values():
        search_space_size *= len(values)

    return min(configured_n_iter, search_space_size)


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def validate_metadata_fields(metadata: dict[str, Any]) -> None:
    missing_fields = REQUIRED_METADATA_FIELDS - set(metadata)
    if missing_fields:
        raise ValueError(f"Missing metadata fields: {sorted(missing_fields)}.")


def mlflow_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("mlflow", {}).get("enabled", False))


def configure_mlflow(mlflow_config: dict[str, Any]) -> Any:
    import mlflow

    tracking_uri = str(mlflow_config["tracking_uri"])
    experiment_name = str(mlflow_config["experiment_name"])

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=str(mlflow_config["artifact_location"]),
        )
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_candidate_run(
    mlflow: Any,
    result: CandidateResult,
    training_config: dict[str, Any],
    target: str,
    number_of_features: int,
) -> str:
    with mlflow.start_run(run_name=result.model_name) as run:
        mlflow.log_params(
            {
                "model_name": result.model_name,
                "target": target,
                "number_of_features": number_of_features,
                "test_size": float(training_config["test_size"]),
                "random_state": int(training_config["random_state"]),
                "cv_folds": int(training_config["cv_folds"]),
                "search_n_iter": int(training_config["search_n_iter"]),
                "scoring": training_config["scoring"],
                "best_params": json.dumps(result.best_params, sort_keys=True),
            }
        )
        mlflow.log_metrics(
            {
                "cv_rmse": result.cv_rmse,
                "test_rmse": result.test_rmse,
                "test_mae": result.test_mae,
                "test_r2": result.test_r2,
            }
        )
        mlflow.set_tags(
            {
                "selected_model": str(result.selected_model).lower(),
                "model_stage": "staging",
            }
        )
        return run.info.run_id


def log_training_artifacts(
    mlflow: Any,
    result: CandidateResult,
    artifact_paths: list[Path],
    model_path: Path,
    log_model_artifact: bool,
) -> None:
    if result.mlflow_run_id is None:
        return

    with mlflow.start_run(run_id=result.mlflow_run_id):
        mlflow.set_tag("selected_model", str(result.selected_model).lower())
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        if log_model_artifact:
            mlflow.log_artifact(str(model_path))


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    data, schema = load_feature_artifacts(config["features"])
    features = list(schema["features"])
    target = schema["target"]
    training_config = config["training"]
    mlflow_config = config.get("mlflow", {})
    mlflow_client = configure_mlflow(mlflow_config) if mlflow_enabled(config) else None
    random_state = int(training_config["random_state"])

    train_indices, test_indices = train_test_split(
        list(data.index),
        test_size=float(training_config["test_size"]),
        random_state=random_state,
    )
    write_json(
        {"test_indices": [int(index) for index in test_indices]},
        Path(config["evaluation"]["metrics_dir"])
        / config["evaluation"]["test_indices_filename"],
    )

    x_train = data.loc[train_indices, features]
    y_train = data.loc[train_indices, target]
    x_test = data.loc[test_indices, features]
    y_test = data.loc[test_indices, target]

    candidates = build_candidates(
        preprocessing_config=config["preprocessing"],
        random_state=random_state,
        n_jobs=int(training_config["n_jobs"]),
    )
    comparison_rows: list[dict[str, Any]] = []
    best_result: CandidateResult | None = None
    configured_n_iter = int(training_config["search_n_iter"])

    for model_name, (pipeline, param_space) in candidates.items():
        logger.info("Training candidate model: %s", model_name)
        n_iter = get_search_iterations(param_space, configured_n_iter)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            scoring=training_config["scoring"],
            cv=int(training_config["cv_folds"]),
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=int(training_config["n_jobs"]),
        )
        search.fit(x_train, y_train)
        cv_rmse = float(-search.best_score_)
        test_metrics = evaluate_model(search.best_estimator_, x_test, y_test)
        result = CandidateResult(
            model_name=model_name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            cv_rmse=cv_rmse,
            test_rmse=test_metrics["test_rmse"],
            test_mae=test_metrics["test_mae"],
            test_r2=test_metrics["test_r2"],
        )
        if mlflow_client is not None:
            result.mlflow_run_id = log_candidate_run(
                mlflow=mlflow_client,
                result=result,
                training_config=training_config,
                target=target,
                number_of_features=len(features),
            )

        row = {
            "model_name": model_name,
            "cv_rmse": cv_rmse,
            **test_metrics,
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        }
        comparison_rows.append(row)

        if best_result is None or cv_rmse < best_result.cv_rmse:
            best_result = result

    if best_result is None:
        raise RuntimeError("No model candidates were trained.")
    best_result.selected_model = True

    staging_dir = Path(config["models"]["staging_dir"])
    model_path = staging_dir / config["models"]["model_filename"]
    metadata_path = staging_dir / config["models"]["metadata_filename"]
    metrics_path = (
        Path(config["evaluation"]["metrics_dir"])
        / config["evaluation"]["staging_metrics_filename"]
    )
    comparison_path = (
        Path(config["evaluation"]["metrics_dir"])
        / config["evaluation"]["model_comparison_filename"]
    )

    staging_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_result.best_estimator, model_path)
    logger.info("Staging model saved")

    metadata = {
        "model_name": best_result.model_name,
        "model_stage": "staging",
        "target": target,
        "features": features,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "test_size": float(training_config["test_size"]),
        "random_state": random_state,
        "cv_folds": int(training_config["cv_folds"]),
        "search_n_iter": int(training_config["search_n_iter"]),
        "best_params": best_result.best_params,
        "cv_rmse": best_result.cv_rmse,
        "test_rmse": best_result.test_rmse,
        "test_mae": best_result.test_mae,
        "test_r2": best_result.test_r2,
    }
    validate_metadata_fields(metadata)
    write_json(metadata, metadata_path)
    write_json(metadata, metrics_path)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    logger.info("Training metrics saved")

    if mlflow_client is not None:
        log_training_artifacts(
            mlflow=mlflow_client,
            result=best_result,
            artifact_paths=[comparison_path, metrics_path, metadata_path],
            model_path=model_path,
            log_model_artifact=bool(mlflow_config["log_model_artifact"]),
        )
        logger.info("MLflow training runs logged")

    return model_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
