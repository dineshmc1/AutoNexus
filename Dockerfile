FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUTONEXUS_WEB_WORKSPACE=/data/studio-runs \
    AUTONEXUS_WEB_DB=/data/autonexus.sqlite3

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN python -m pip install --no-cache-dir ".[serve,auth,cloud,boosting,explain,vision,memory]"

RUN mkdir -p /data/studio-runs

EXPOSE 8080
CMD ["sh", "-c", "uvicorn autonexus.railway:app --host 0.0.0.0 --port ${PORT:-8080}"]
