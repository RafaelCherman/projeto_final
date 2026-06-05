FROM python:3.10-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    faiss-cpu==1.7.4 \
    annoy==1.17.3 \
    hnswlib==0.7.0 \
    psutil==5.9.5

# Copia os scripts de experimento
COPY experiments/ /app/experiments/

# Diretórios de dados e resultados via volume
VOLUME ["/data", "/results"]