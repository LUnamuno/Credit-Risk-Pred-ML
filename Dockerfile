FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Editable install keeps the /app/{data,models} layout that credit_risk.data/train expect.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[api]"

# The image contains no model and no data: train inside the container (same library versions
# as the API) into mounted volumes, then serve from them. See the README.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/data/raw \
    && chown -R app:app /app/models /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
