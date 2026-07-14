# syntax=docker/dockerfile:1.7
# Single-process image: FastAPI (uvicorn) only. The engine is a stateless API
# client — no frontend, no database — so there is no Caddy / supervisord / Postgres.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    # Decouple the manifests dir from the installed package location (see config.py).
    MANIFESTS_DIR=/app/manifests

WORKDIR /app

# Pinned deps first (see requirements.txt, which already includes the `llm`
# extra so USE_LLM=1 works without rebuilding the image) for a reproducible,
# cache-friendly layer. The `--no-deps` install right after only registers
# this local package — its dependencies are already satisfied above.
COPY requirements.txt pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps .

# Runtime data (manifests) are copied separately and referenced via MANIFESTS_DIR.
COPY manifests ./manifests

# Non-root, matching the sibling gyms' convention (defense in depth — Cloud Run
# also sandboxes the container regardless). Fixed uid/gid keeps it deterministic
# across rebuilds.
RUN groupadd --gid 1001 app \
    && useradd --uid 1001 --gid app --no-create-home --no-log-init app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# Stdlib-only healthcheck (keeps the image slim — no curl).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request,sys; \
u='http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/health'; \
sys.exit(0 if urllib.request.urlopen(u,timeout=3).status==200 else 1)" || exit 1

# exec via sh so ${PORT} is expanded (Cloud Run injects PORT) but signals still
# reach uvicorn for graceful shutdown.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
