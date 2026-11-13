FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CORPUS_DIR=/app/corpus CACHE_DIR=/app/.cache/embeddings \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 researcher
COPY requirements.txt requirements-lock.txt ./
RUN pip install -r requirements-lock.txt && pip check
COPY pharma_agent ./pharma_agent
COPY corpus ./corpus
RUN mkdir -p /app/.cache/embeddings && chown -R researcher:researcher /app
USER researcher
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=150s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "pharma_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
