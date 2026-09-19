FROM python:3.12-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Only what the API needs at runtime — models, data, api/ + src/ packages
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY outputs/metrics/ ./outputs/metrics/

EXPOSE 8000

# $PORT is injected by most PaaS hosts (Render, Railway, Fly) at runtime;
# falls back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
