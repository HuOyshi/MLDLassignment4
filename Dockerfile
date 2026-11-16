FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# the container has no GPU, so use the CPU build of TensorFlow (about half the download)
RUN sed -i 's/^tensorflow==/tensorflow-cpu==/' requirements.txt \
    && pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements.txt

# only what the app needs at run time
COPY ocr_utils.py app.py labels.json ./
COPY models/model.keras models/model.keras

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
