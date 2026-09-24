#!/bin/bash
# AI4Lassa API — local Docker build & test script
#
# Run this from inside the AI-lassa/ project folder:
#   chmod +x test_docker.sh
#   ./test_docker.sh
#
# Requires Docker Desktop (or Docker Engine) installed and running.

set -e  # stop immediately if any step fails

IMAGE_NAME="ai4lassa-api"
CONTAINER_NAME="ai4lassa-api-test"
PORT=8000

echo "=== 1. Building the Docker image ==="
docker build -t "$IMAGE_NAME" .

echo ""
echo "=== 2. Removing any old container with the same name (if present) ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "=== 3. Starting the container ==="
docker run -d --name "$CONTAINER_NAME" -p ${PORT}:8000 "$IMAGE_NAME"

echo ""
echo "=== 4. Waiting for the API to finish loading models (~10-15s) ==="
sleep 15

echo ""
echo "=== 5. Testing /health ==="
curl -s "http://localhost:${PORT}/health"
echo ""

echo ""
echo "=== 6. Testing /forecast/latest ==="
curl -s "http://localhost:${PORT}/forecast/latest?decision_threshold=0.15"
echo ""

echo ""
echo "=== 7. Testing /forecast/predict (explicit feature vector) ==="
curl -s -X POST "http://localhost:${PORT}/forecast/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "case_count": 180,
    "case_count_lag1": 169,
    "case_count_lag2": 216,
    "case_count_lag3": 316,
    "case_count_lag6": 231,
    "case_count_lag12": 400,
    "case_count_roll3_mean": 233.7,
    "case_count_roll6_mean": 246.8,
    "case_count_roll3_max": 316,
    "case_growth_lag1": -47,
    "positivity_rate_lag1": 0.09,
    "month_sin": -0.5,
    "month_cos": 0.87,
    "year": 2025,
    "decision_threshold": 0.2
  }'
echo ""

echo ""
echo "=== 8. Container logs (check for errors) ==="
docker logs "$CONTAINER_NAME" --tail 30

echo ""
echo "=== 9. Cleaning up ==="
docker rm -f "$CONTAINER_NAME"

echo ""
echo "=== Done ==="
echo "If all 3 endpoints above returned JSON (not empty/errors), the image is deployment-ready."
echo "Interactive API docs would be at: http://localhost:${PORT}/docs (while the container is running)"
