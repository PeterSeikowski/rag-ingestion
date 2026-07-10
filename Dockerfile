# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # requirements.txt's "# --- Testing ---" section (pytest, pytest-mock)
    # is only needed to run the test suite, never at runtime — prune it
    # from the venv before it gets copied into the runtime stage below,
    # so the production image doesn't carry test tooling as dead weight.
    && pip uninstall -y pytest pytest-mock

# Pre-warm the tiktoken BPE cache at build time so containers never need
# outbound network access just to tokenize text for chunk sizing.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"


FROM python:3.11-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

# docling's dependency chain (docling-ibm-models / easyocr -> torch,
# torchvision, opencv-python-headless) expects shared libraries that
# python:3.11-slim doesn't ship by default. Missing them surfaces as an
# ImportError the first time a real PDF ingestion job runs in the worker
# (DoclingPdfParser imports docling lazily — see adapters/parsers/
# docling_pdf_parser.py) rather than at build or startup time, so install
# them here proactively instead of discovering it at 2am.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken_cache /opt/tiktoken_cache
ENV PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY src ./src

RUN mkdir -p /tmp/rag_ingestion_uploads && chown -R app:app /app /tmp/rag_ingestion_uploads
USER app

EXPOSE 8000

# docker-compose overrides this command for the worker service.
CMD ["uvicorn", "rag_ingestion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
