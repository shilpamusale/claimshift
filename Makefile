PYTHON ?= python3
PY := .venv/bin/python
ROWS ?= 5000000
SEED ?= 7
MONTH ?= 2026-10
CLUSTER ?= claimshift
IMAGE ?= claimshift-scorer
BUNDLE ?= v1
FEATURE_ORDER ?= manifest
SKLEARN_VERSION = $(shell $(PY) -c "import sklearn; print(sklearn.__version__)")
DOCKER_BUILD = docker build -f serving/Dockerfile --build-arg SKLEARN_VERSION=$(SKLEARN_VERSION)

.PHONY: setup data data-small contract contract-demo features skew-bench train mlflow-ui test lint clean \
        serve images cluster load-images deploy-v1 rollout-v2-bad rollout-v2.1 watch traffic analysis-logs cluster-delete

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

features:                ## as-of features for every month after the first
	$(PY) -m pipeline.run_pipeline features

skew-bench:              ## incident 2: baseline vs AQE vs broadcast vs salting
	$(PY) -m pipeline.run_pipeline skew-bench --month $(MONTH)

train:                   ## train v1-flat, v1, v2; log to MLflow; write bundles to models/
	$(PY) -m training.train --all

mlflow-ui:               ## browse runs at http://localhost:5001 (5000 is taken by AirPlay on macOS)
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

serve:                   ## one scorer on localhost:8080 (BUNDLE=v1, FEATURE_ORDER=manifest)
	MODEL_DIR=models/$(BUNDLE) FEATURE_ORDER=$(FEATURE_ORDER) BUILD=local $(PY) -m uvicorn serving.app:app --port 8080

images:                  ## scorer images: v1, v2-bad (incident 3 bug), v2.1 (fixed)
	$(DOCKER_BUILD) --build-arg BUNDLE=v1 --build-arg BUILD=v1 -t $(IMAGE):v1 .
	$(DOCKER_BUILD) --build-arg BUNDLE=v2 --build-arg BUILD=v2-bad --build-arg FEATURE_ORDER=hardcoded -t $(IMAGE):v2-bad .
	$(DOCKER_BUILD) --build-arg BUNDLE=v2 --build-arg BUILD=v2.1 -t $(IMAGE):v2.1 .

cluster:                 ## kind cluster + Argo Rollouts controller
	kind create cluster --config k8s/kind-config.yaml
	kubectl create namespace argo-rollouts
	kubectl apply --server-side --force-conflicts -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
	kubectl -n argo-rollouts rollout status deployment/argo-rollouts --timeout=180s

load-images:             ## copy the scorer images into the kind cluster
	kind load docker-image $(IMAGE):v1 $(IMAGE):v2-bad $(IMAGE):v2.1 --name $(CLUSTER)

deploy-v1:               ## services, the canary check, and v1 as the stable version
	kubectl apply -f k8s/services.yaml -f rollout/analysis-template.yaml -f rollout/rollout.yaml
	kubectl argo rollouts status denial-scorer --timeout 180s

rollout-v2-bad:          ## incident 3: v2 with the feature-order bug; expect an automatic rollback
	kubectl argo rollouts set image denial-scorer scorer=$(IMAGE):v2-bad
	-kubectl argo rollouts status denial-scorer --timeout 300s

rollout-v2.1:            ## the fixed v2; expect promotion to 100%
	kubectl argo rollouts set image denial-scorer scorer=$(IMAGE):v2.1
	kubectl argo rollouts status denial-scorer --timeout 300s

watch:                   ## live view of the rollout (Ctrl+C to stop)
	kubectl argo rollouts get rollout denial-scorer --watch

traffic:                 ## which builds answer on localhost:8080
	$(PY) -m rollout.traffic --url http://localhost:8080 --requests 200

analysis-logs:           ## output of the most recent canary check
	kubectl logs $$(kubectl get pods -l job-name --sort-by=.metadata.creationTimestamp -o name | tail -1)

cluster-delete:          ## remove the kind cluster
	kind delete cluster --name $(CLUSTER)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

clean:
	rm -rf data mlruns spark-events models mlartifacts mlflow.db