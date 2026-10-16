FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models in their own layer (before app code) so code edits do not re-download.
ENV MODEL_DIR=/app/models \
    LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx \
    PYTHONUNBUFFERED=1

COPY scripts/download_models.py scripts/download_liveness_model.py ./scripts/
RUN python scripts/download_models.py \
    && python scripts/download_liveness_model.py

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
