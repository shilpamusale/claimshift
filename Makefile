PYTHON ?= python3
PY := .venv/bin/python
ROWS ?= 5000000
SEED ?= 7

.PHONY: setup data data-small test lint clean

setup:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m data_gen.generate_claims --rows $(ROWS) --seed $(SEED) --out data

data-small:
	$(MAKE) data ROWS=500000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

clean:
	rm -rf data mlruns spark-events
