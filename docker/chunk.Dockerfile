# podman build -t job-image docker
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    postgresql libpq-dev \
    gcc \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN uv pip install --no-cache-dir "git+https://github.com/CoopHive/markdown-converter.git@main" --system

ENTRYPOINT ["python3", "-c", "import sys; from src.chunker import chunk_from_url; print(chunk_from_url(*sys.argv[1:]))"]