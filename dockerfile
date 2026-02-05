# Use slim to keep the image smaller (TF will still be big, but this helps)
FROM python:3.11-slim

# Prevent Python from writing .pyc files + force logs straight to stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed by Pillow (image handling) and some TF builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Render sets PORT automatically; we read it in the CMD
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]