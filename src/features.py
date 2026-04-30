import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare feature dataset.")
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

    for section in ("data", "features"):
        if section not in config:
            raise ValueError(f"Config file must include a {section} section.")

    return config


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [column.lower() for column in normalized.columns]
    return normalized


def validate_features(data: pd.DataFrame, target_column: str) -> list[str]:
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}.")

    feature_columns = [column for column in data.columns if column != target_column]
    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric_features:
        raise ValueError(f"Non-numeric feature columns: {non_numeric_features}.")

    return feature_columns


def build_feature_schema(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> dict[str, Any]:
    return {
        "features": feature_columns,
        "target": target_column,
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "number_of_rows": len(data),
        "number_of_features": len(feature_columns),
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    data_config = config["data"]
    feature_config = config["features"]

    raw_path = Path(data_config["raw_dir"]) / data_config["raw_filename"]
    processed_dir = Path(feature_config["processed_dir"])
    processed_path = processed_dir / feature_config["processed_filename"]
    schema_path = processed_dir / feature_config["schema_filename"]
    registry_path = processed_dir / feature_config["registry_filename"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}.")

    data = normalize_columns(pd.read_csv(raw_path))
    target_column = data_config["target_column"].lower()
    feature_columns = validate_features(data, target_column)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(processed_path, index=False)
    logger.info("Processed feature dataset saved")

    schema = build_feature_schema(data, feature_columns, target_column)
    write_json(schema, schema_path)
    logger.info("Feature schema saved")

    write_json(
        {
            "feature_set_name": feature_config["feature_set_name"],
            "version": feature_config["version"],
            "source_dataset": raw_path.as_posix(),
            "processed_dataset": processed_path.as_posix(),
            "feature_schema": schema_path.as_posix(),
            "features": feature_columns,
            "target": target_column,
        },
        registry_path,
    )
    logger.info("Feature registry saved")

    return processed_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
