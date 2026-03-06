FROM python:3.11-slim

# Build argument: backend specification
# Format: "onnx" or "keras-tf2.13" or "keras-tf2.14" (default: "keras-tf2.13")
ARG BACKEND=keras-tf2.13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BACKEND=${BACKEND}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo && rm -rf /var/lib/apt/lists/*

# Copy the appropriate requirements file based on TensorFlow version
COPY requirements-${BACKEND}.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy server and interpreter modules
COPY interpreter_keras.py interpreter_onnx.py server.py /app/
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
