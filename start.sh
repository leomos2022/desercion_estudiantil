#!/bin/bash

# ============================================
# Alerta Estudiantil Colombia - Script de Inicio
# ============================================

echo "🎓 Alerta Estudiantil Colombia"
echo "Sistema de Predicción de Deserción Estudiantil"
echo "=============================================="
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias
echo ""
echo "📦 Instalando dependencias..."
pip install -q -r requirements.txt

# Crear directorio de modelos
mkdir -p models
mkdir -p data

# Entrenar modelo si no existe
if [ ! -f "models/desercion_model.joblib" ]; then
    echo ""
    echo "🤖 Entrenando modelo de ML..."
    python train_model.py
fi

# Procesar datos SPADIES si existen
if [ -f "../DESERCION.xlsx" ] && [ ! -f "data/estadisticas_nacionales.json" ]; then
    echo ""
    echo "📊 Procesando datos SPADIES..."
    python process_spadies.py
fi

# Iniciar servidor
echo ""
echo "🚀 Iniciando servidor..."
echo ""
echo "=============================================="
echo "📡 API disponible en:      http://localhost:8000"
echo "📖 Documentación en:       http://localhost:8000/docs"
echo "🌐 Frontend en:            http://localhost:8000/app"
echo "=============================================="
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
