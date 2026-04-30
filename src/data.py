import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and profile raw data.")
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

    if not isinstance(config, dict) or "data" not in config:
        raise ValueError("Config file must include a data section.")

    return config["data"]


def download_dataset(dataset_name: str) -> Path:
    logger.info("Downloading dataset from KaggleHub")
    import kagglehub

    return Path(kagglehub.dataset_download(dataset_name))


def find_csv_file(dataset_dir: Path) -> Path:
    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {dataset_dir}.")

    if len(csv_files) > 1:
        logger.info("Multiple CSV files found; using %s", csv_files[0])

    return csv_files[0]


def validate_raw_data(
    data: pd.DataFrame,
    required_columns: list[str],
    target_column: str,
    min_rows: int,
) -> None:
    normalized_columns = {column.upper(): column for column in data.columns}
    missing_columns = sorted(
        {column.upper() for column in required_columns} - set(normalized_columns)
    )

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}.")

    normalized_target = target_column.upper()
    if normalized_target not in normalized_columns:
        raise ValueError(f"Target column not found: {target_column}.")

    if len(data) < min_rows:
        raise ValueError(f"Dataset has {len(data)} rows; expected at least {min_rows}.")

    real_target_column = normalized_columns[normalized_target]
    if not pd.api.types.is_numeric_dtype(data[real_target_column]):
        raise ValueError(f"Target column must be numeric: {target_column}.")


def build_data_profile(data: pd.DataFrame, target_column: str) -> dict[str, Any]:
    numeric_data = data.select_dtypes(include="number")
    numeric_columns = list(numeric_data.columns)
    non_numeric_columns = [
        column for column in data.columns if column not in numeric_columns
    ]
    numeric_summary = numeric_data.describe(percentiles=[0.25, 0.5, 0.75]).T
    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1
    lower_bounds = q1 - 1.5 * iqr
    upper_bounds = q3 + 1.5 * iqr
    outlier_mask = numeric_data.lt(lower_bounds) | numeric_data.gt(upper_bounds)
    outlier_counts = outlier_mask.sum().astype(int)

    return {
        "number_of_rows": len(data),
        "number_of_columns": len(data.columns),
        "target_column": target_column,
        "columns": list(data.columns),
        "missing_values_by_column": data.isna().sum().astype(int).to_dict(),
        "missing_rate_by_column": data.isna().mean().astype(float).to_dict(),
        "numeric_columns": numeric_columns,
        "non_numeric_columns": non_numeric_columns,
        "duplicate_rows": int(data.duplicated().sum()),
        "basic_numeric_summary": {
            column: {
                "mean": float(row["mean"]),
                "std": None if pd.isna(row["std"]) else float(row["std"]),
                "min": float(row["min"]),
                "p25": float(row["25%"]),
                "median": float(row["50%"]),
                "p75": float(row["75%"]),
                "max": float(row["max"]),
            }
            for column, row in numeric_summary.iterrows()
        },
        "potential_outliers_iqr_count_by_column": outlier_counts.to_dict(),
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run(config_path: Path) -> Path:
    params = load_config(config_path)
    raw_dir = Path(params["raw_dir"])
    output_csv = raw_dir / params["raw_filename"]
    metadata_path = raw_dir / params["metadata_filename"]
    profile_path = Path(params["profile_dir"]) / params["profile_filename"]

    dataset_dir = download_dataset(params["kaggle_dataset"])
    source_csv = find_csv_file(dataset_dir)
    data = pd.read_csv(source_csv)

    validate_raw_data(
        data=data,
        required_columns=list(params["required_columns"]),
        target_column=params["target_column"],
        min_rows=int(params["min_rows"]),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, output_csv)
    logger.info("Raw dataset saved")

    write_json(
        {
            "dataset": params["kaggle_dataset"],
            "output_csv": output_csv.as_posix(),
            "rows": len(data),
            "columns": list(data.columns),
            "target_column": params["target_column"],
        },
        metadata_path,
    )
    logger.info("Dataset metadata saved")

    write_json(build_data_profile(data, params["target_column"]), profile_path)
    logger.info("Data profile saved")

    return output_csv


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
