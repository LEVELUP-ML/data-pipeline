FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y git curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt