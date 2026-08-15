FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivo de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código-fonte e dados
COPY . .

# Treinar modelo inicial se necessário
RUN PYTHONPATH=src python3 src/train_model.py

# Expor portas para FastAPI (8000) e Streamlit (8501)
EXPOSE 8000 8501

# Comando padrão
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
