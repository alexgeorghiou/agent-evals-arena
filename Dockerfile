# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[dashboard,anthropic,openai]"

COPY suites ./suites
COPY baselines ./baselines
COPY dashboard ./dashboard

RUN useradd --create-home --uid 10001 arena && mkdir -p /data && chown arena /data
USER arena

VOLUME ["/data"]
EXPOSE 8501

# Default: serve the dashboard. Override the command to run the CLI, e.g.
#   docker compose run --rm arena arena run suites --store /data
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--", "--store", "/data"]
