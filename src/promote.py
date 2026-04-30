"""Promueve el modelo staging a producción si supera el umbral de mejora."""

import argparse
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote staging model.")
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

    for section in ("models", "evaluation", "promotion"):
        if section not in config:
            raise ValueError(f"Config file must include a {section} section.")

    return config


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def should_promote(
    staging_rmse: float,
    production_rmse: float | None,
    min_rmse_improvement: float,
) -> tuple[bool, str]:
    """Decide si staging supera a producción por al menos el umbral configurado."""
    if production_rmse is None:
        return True, "No production model exists."

    required_rmse = production_rmse * (1 - min_rmse_improvement)
    if staging_rmse <= required_rmse:
        return True, "Staging RMSE meets required improvement."

    return False, "Staging RMSE does not meet required improvement."


def copy_model_artifacts(config: dict[str, Any]) -> bool:
    model_config = config["models"]
    evaluation_config = config["evaluation"]
    staging_dir = Path(model_config["staging_dir"])
    production_dir = Path(model_config["production_dir"])
    production_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        staging_dir / model_config["model_filename"],
        production_dir / model_config["model_filename"],
    )
    shutil.copy2(
        Path(evaluation_config["metrics_dir"])
        / evaluation_config["staging_metrics_filename"],
        production_dir / model_config["metrics_filename"],
    )
    metadata = read_json(staging_dir / model_config["metadata_filename"])
    metadata_stage_corrected = metadata.get("model_stage") != "production"
    metadata["model_stage"] = "production"
    write_json(metadata, production_dir / model_config["metadata_filename"])
    return metadata_stage_corrected


def correct_production_metadata(metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False

    metadata = read_json(metadata_path)
    if metadata.get("model_stage") == "production":
        return False

    metadata["model_stage"] = "production"
    write_json(metadata, metadata_path)
    return True


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run(config_path: Path) -> Path:
    """Aplica la regla de promoción y copia los artefactos a producción."""
    config = load_config(config_path)
    model_config = config["models"]
    evaluation_config = config["evaluation"]
    promotion_config = config["promotion"]

    staging_metrics_path = (
        Path(evaluation_config["metrics_dir"])
        / evaluation_config["staging_metrics_filename"]
    )
    production_model_path = (
        Path(model_config["production_dir"]) / model_config["model_filename"]
    )
    production_metrics_path = (
        Path(model_config["production_dir"]) / model_config["metrics_filename"]
    )
    production_metadata_path = (
        Path(model_config["production_dir"]) / model_config["metadata_filename"]
    )
    report_path = (
        Path(evaluation_config["metrics_dir"]) / promotion_config["report_filename"]
    )

    staging_metrics = read_json(staging_metrics_path)
    staging_rmse = float(staging_metrics["test_rmse"])
    production_rmse = None
    if production_model_path.exists():
        production_metrics = read_json(production_metrics_path)
        production_rmse = float(production_metrics["test_rmse"])

    min_improvement = float(promotion_config["min_rmse_improvement"])
    promoted, reason = should_promote(
        staging_rmse=staging_rmse,
        production_rmse=production_rmse,
        min_rmse_improvement=min_improvement,
    )

    if promoted:
        metadata_stage_corrected = copy_model_artifacts(config)
        logger.info("Model promoted to production")
    else:
        metadata_stage_corrected = correct_production_metadata(production_metadata_path)
        if metadata_stage_corrected:
            logger.info("Production metadata stage corrected")
        logger.info("Model promotion skipped")

    report = {
        "promoted": promoted,
        "reason": reason,
        "metadata_stage_corrected": metadata_stage_corrected,
        "staging_rmse": staging_rmse,
        "production_rmse": production_rmse,
        "min_rmse_improvement": min_improvement,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    write_json(report, report_path)
    logger.info("Promotion report saved")

    return report_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
