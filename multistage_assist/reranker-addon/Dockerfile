ARG BUILD_FROM
FROM python:3.11-slim-bookworm

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    jq && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    sentence-transformers>=2.2.0

# Copy application files
COPY app.py /app.py
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD [ "/run.sh" ]
