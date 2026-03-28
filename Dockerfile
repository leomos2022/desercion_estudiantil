# Dockerfile para Alerta Estudiantil Colombia
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p models data

# Variables de entorno
ENV PORT=8000
ENV HOST=0.0.0.0

# Exponer puerto
EXPOSE 8000

# Comando de inicio (Render usa $PORT automáticamente)
CMD ["/bin/sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
