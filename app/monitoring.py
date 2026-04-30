from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received.",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of prediction requests that failed.",
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests in seconds.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_VALUE = Histogram(
    "prediction_value",
    "Distribution of predicted target values.",
    buckets=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Indicates whether the production model is loaded (1) or not (0).",
)

MODEL_INFO = Gauge(
    "model_info",
    "Static labels describing the served model.",
    ["model_name", "model_stage", "target"],
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
