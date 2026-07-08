FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

FROM python:3.11-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY app/ ./app/
COPY config/ ./config/
COPY docs/ ./docs/
COPY pdf_ragpromax.py ./
COPY config.json ./

RUN mkdir -p /app/vector_store

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]