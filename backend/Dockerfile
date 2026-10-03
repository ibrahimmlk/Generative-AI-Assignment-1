# Build context: repository root (see docker-compose.yml)
FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/models SAMPLE_DIR=/srv/samples
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# only the corruption module of the training package is needed at inference time
COPY genai/__init__.py genai/corruptions.py genai/
COPY backend/app app
COPY backend/samples samples
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
