# syntax=docker/dockerfile:1
#
# Targets:
#   - runtime (default): image for users
#   - dev: adds developer tooling + installs ".[dev]" extras
#
# Build runtime (users):
#   docker build -t dsanmartim/jira-issues-dashboard:latest .
#
# Build dev image (contributors):
#   docker build --target dev -t jira-issues-dashboard:dev .

FROM python:3.13.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Create the mount point for secrets
RUN mkdir -p /app/.streamlit

# Non-root user (better for Kubernetes)
RUN useradd -m -u 10001 appuser

# Copy only what we need for install + runtime
COPY pyproject.toml /app/pyproject.toml
COPY jira_app /app/jira_app
COPY run_dashboard.py /app/run_dashboard.py

# Install runtime deps from pyproject.toml :contentReference[oaicite:1]{index=1}
RUN pip install --no-cache-dir .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1


# ---- Dev target (optional) ----
FROM base AS dev

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Allow non-root builds to write egg-info during editable/dev installs
RUN chown -R appuser:appuser /app

USER appuser
# Installs your dev extras (pytest/pre-commit/ruff/black) :contentReference[oaicite:2]{index=2}
RUN pip install --no-cache-dir ".[dev]"


# ---- Runtime target (default) ----
FROM base AS runtime

USER appuser
CMD ["streamlit", "run", "run_dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
