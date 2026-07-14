# Use an optimized, slim Debian-based Python image
FROM python:3.10-slim

# Install system dependencies required for system telemetry (psutil) and C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core system architecture files
COPY . .

# Expose the ports for both the Streamlit GUI (8501) and the FastAPI Engine (8000)
EXPOSE 8501
EXPOSE 8000

# Default command runs the production telemetry dashboard
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
