FROM python:3.11-slim

# System dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY config.py dataset.py models.py evaluate.py api.py ./

# Model checkpoint and threshold are mounted at runtime (see docker-compose.yml)
# so they are not baked into the image.

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]