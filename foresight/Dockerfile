FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data and model directories
RUN mkdir -p data/raw \
    data/processed/cleaned \
    data/processed/features \
    data/processed/forecasts \
    data/processed/dashboard \
    models/baseline \
    models/lightgbm \
    models/lstm \
    reports

# Expose ports: Streamlit (8501) + FastAPI (8000)
EXPOSE 8501 8000

# Default: run Streamlit dashboard
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
