# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* requirements.txt* /app/

RUN set -eux; \
    pip install --upgrade pip; \
    if [ -f requirements.txt ]; then \
      pip install -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
      pip install "uv" && uv pip install --system .; \
    else \
      pip install fastapi uvicorn[standard] python-dotenv litellm pydantic; \
    fi

COPY . /app

EXPOSE 8082

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082", "--log-level", "error"]
