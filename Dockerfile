# Stage 1: Build
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY README.md ./

RUN uv sync --no-dev --no-editable --frozen

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# System deps for docling/easyocr (OpenCV headless, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 noidrag && \
    useradd --uid 1001 --gid noidrag --shell /bin/bash --create-home noidrag

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER noidrag

ENTRYPOINT ["noid-rag"]
CMD ["--help"]
