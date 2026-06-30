FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
