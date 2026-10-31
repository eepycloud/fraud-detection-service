FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

# LightGBM needs the OpenMP runtime on Debian.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN groupadd --system app && useradd --system --gid app app
COPY --chown=app:app src/ ./src/
# Bundle the model trained on public simulated data for reproducible deployments.
COPY --chown=app:app models/baseline.joblib ./models/baseline.joblib

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

