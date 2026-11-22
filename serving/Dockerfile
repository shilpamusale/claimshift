# Denial-scoring image: the serving code plus one model bundle.
# Build from the repo root (see `make images`):
#   docker build -f serving/Dockerfile --build-arg SKLEARN_VERSION=1.9.1 \
#     --build-arg BUNDLE=v1 --build-arg BUILD=v1 -t claimshift-scorer:v1 .
FROM python:3.12-slim

ARG SKLEARN_VERSION
ARG BUNDLE=v1
ARG FEATURE_ORDER=manifest
ARG BUILD=dev

WORKDIR /app
RUN test -n "$SKLEARN_VERSION" || (echo "pass --build-arg SKLEARN_VERSION (must match training)" && false) \
 && pip install --no-cache-dir "scikit-learn==${SKLEARN_VERSION}" "pandas>=2.1,<3" "pyarrow>=15" \
      "fastapi>=0.110" "uvicorn>=0.29" "requests>=2.31" "pyyaml>=6" joblib numpy

COPY training/__init__.py training/model_io.py training/
COPY serving/ serving/
COPY rollout/__init__.py rollout/canary_check.py rollout/traffic.py rollout/
COPY configs/thresholds.yaml configs/
COPY models/${BUNDLE}/ /model/
RUN chmod -R a+rX /app /model        # host file permissions carry over; make code and model readable

ENV MODEL_DIR=/model FEATURE_ORDER=${FEATURE_ORDER} BUILD=${BUILD} PYTHONUNBUFFERED=1
RUN useradd --create-home scorer
USER scorer
EXPOSE 8080
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]