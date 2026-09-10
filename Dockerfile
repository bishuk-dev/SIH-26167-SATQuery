FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build
COPY pyproject.toml ./
COPY satquery ./satquery
ARG SATQUERY_INSTALL_TARGET=.
RUN pip install "$SATQUERY_INSTALL_TARGET"


FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/data \
    MODEL_ROOT=/models \
    PORT=8000

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system satquery \
    && useradd --system --gid satquery --home-dir /app satquery \
    && mkdir -p /app /data /models \
    && chown -R satquery:satquery /app /data /models

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=satquery:satquery apps ./apps
COPY --chown=satquery:satquery satquery ./satquery
COPY --chown=satquery:satquery models/registry.yaml ./models/registry.yaml
COPY --chown=satquery:satquery experiments/phase5_backend/backend_contract.yaml \
    ./experiments/phase5_backend/backend_contract.yaml

USER satquery
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + '${PORT:-8000}' + '/health/ready', timeout=3)" || exit 1

CMD ["sh", "-c", "exec uvicorn apps.api.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
