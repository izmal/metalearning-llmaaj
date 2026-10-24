FROM python:3.12-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /root/.cache
COPY judge/ judge/

# API responses are cached only inside the container (/tmp), so every adapt/eval run starts clean.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LLM_CACHE_DIR=/tmp/llm-cache

ENTRYPOINT ["python", "/app/judge/main.py"]
