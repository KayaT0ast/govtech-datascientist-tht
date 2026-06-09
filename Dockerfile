FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first so pip doesn't pull the ~2-3 GB CUDA build.
# The container runs CPU-only (no --gpus in docker-compose), so nothing is lost.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/knowledge-base
RUN chmod +x entrypoint.sh

EXPOSE 8000

CMD ["bash", "entrypoint.sh"]
