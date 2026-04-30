PYTHON ?= python3.11

.PHONY: setup data features train evaluate promote pipeline retrain test lint format ci

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

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m compileall src tests

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check . --fix

ci: lint test pipeline
