FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install system deps (important for pandas / numpy sometimes)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# HF required port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]