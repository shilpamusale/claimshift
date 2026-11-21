PYTHON ?= python3
PY := .venv/bin/python
ROWS ?= 5000000
SEED ?= 7
MONTH ?= 2026-10

.PHONY: setup data data-small contract contract-demo test lint clean

setup:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m data_gen.generate_claims --rows $(ROWS) --seed $(SEED) --out data

data-small:
	$(MAKE) data ROWS=500000

contract:                ## data contract for one month; exit 2 = retrain, 3 = block
	$(PY) -m pipeline.run_pipeline contract --eval-month $(MONTH)

contract-demo:           ## September passes, October triggers a retrain
	$(PY) -m pipeline.run_pipeline contract --eval-month 2026-09
	-$(PY) -m pipeline.run_pipeline contract --eval-month 2026-10

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

clean:
	rm -rf data mlruns spark-events