PYTHON ?= python3.11
PORT ?= 8000

.PHONY: setup data features train evaluate promote pipeline retrain serve kill-port serve-clean api-check mlflow-ui dvc-repro dvc-status dvc-metrics test lint format ci docker-build docker-up docker-down docker-logs

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/data.py --config configs/params.yaml

features:
	$(PYTHON) src/features.py --config configs/params.yaml

train:
	$(PYTHON) src/train.py --config configs/params.yaml

evaluate:
	$(PYTHON) src/evaluate.py --config configs/params.yaml

promote:
	$(PYTHON) src/promote.py --config configs/params.yaml

pipeline: data features train evaluate promote

retrain: pipeline

serve:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

kill-port:
	@PID=$$(lsof -ti tcp:$(PORT)); \
	if [ -n "$$PID" ]; then \
		kill $$PID; \
		echo "Killed process $$PID on port $(PORT)"; \
	else \
		echo "No process found on port $(PORT)"; \
	fi

serve-clean: kill-port serve

api-check:
	$(PYTHON) -c "from app.main import app; print('API loaded')"

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

dvc-repro:
	dvc repro

dvc-status:
	dvc status

dvc-metrics:
	dvc metrics show

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m compileall src tests

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check . --fix

ci: lint test pipeline
