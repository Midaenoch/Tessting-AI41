FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at runtime — models, data, app/ + src/ packages
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY data/raw/ ./data/raw/
COPY outputs/metrics/ ./outputs/metrics/

EXPOSE 8000

# $PORT is injected by most PaaS hosts (Render, Railway, Fly) at runtime;
# falls back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]