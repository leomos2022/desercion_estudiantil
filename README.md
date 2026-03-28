# 🎓 Alerta Estudiantil Colombia

## Plataforma de Predicción de Deserción Estudiantil con Machine Learning

> **Misión**: Reducir la deserción estudiantil en Colombia mediante inteligencia artificial y acceso público a información.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js/React)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Dashboard   │  │ Calculadora  │  │   Mapa Interactivo       │  │
│  │  Nacional    │  │ de Riesgo    │  │   por Departamento       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API REST (FastAPI - Python)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ /predict     │  │ /stats       │  │ /ies/{codigo}            │  │
│  │ Predicción   │  │ Estadísticas │  │ Info por Universidad     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CAPA DE DATOS                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Modelo ML    │  │ SPADIES      │  │ PostgreSQL/SQLite        │  │
│  │ (joblib)     │  │ (Excel/CSV)  │  │ (Usuarios/Logs)          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Funcionalidades Principales

### 1. Dashboard Público Nacional
- Tasa de deserción por nivel de formación (TyT vs Universitario)
- Ranking de IES por deserción/graduación
- Evolución histórica 2019-2023
- Filtros por departamento y tipo de institución

### 2. Calculadora de Riesgo Personal
- Formulario interactivo con factores de riesgo
- Predicción en tiempo real usando modelo ML
- Explicación de factores (SHAP values)
- Recomendaciones personalizadas

### 3. API Pública
- Endpoints documentados con Swagger
- Acceso gratuito para investigadores
- Datos anonimizados y agregados

---

## 📦 Stack Tecnológico

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **Backend** | FastAPI (Python) | Async, tipado, documentación automática |
| **ML** | scikit-learn + XGBoost | Estándar industria para datos tabulares |
| **Frontend** | HTML/CSS/JS + Chart.js | Simplicidad, sin dependencias pesadas |
| **Deploy Backend** | Railway / Render | Gratis para MVPs |
| **Deploy Frontend** | Vercel / GitHub Pages | CDN global, SSL gratis |
| **Datos** | SPADIES (MEN Colombia) | Fuente oficial del gobierno |

---

## 🏃 Inicio Rápido

### Requisitos
- Python 3.9+
- pip

### Instalación

```bash
# Clonar o descargar el proyecto
cd alerta_estudiantil

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo (primera vez)
python train_model.py

# Iniciar servidor
python -m uvicorn api.main:app --reload
```

### Acceder
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Frontend: Abrir `frontend/index.html` en navegador

---

## 📊 Datos Utilizados

### Fuente: SPADIES 3.0 - Ministerio de Educación Nacional

| Indicador | Descripción | Año |
|-----------|-------------|-----|
| TDA | Tasa de Deserción Anual | 2019-2023 |
| TAI | Tasa de Ausencia Intersemestral | 2019-2023 |
| TDCA | Tasa Deserción Cohorte Acumulada | 2017-2023 |
| TGA | Tasa de Graduación Acumulada | 2017-2023 |

### Estadísticas Clave (2023)
- **288 IES** registradas en Colombia
- **TDA promedio**: 15.28%
- **Deserción acumulada TyT**: 51.64%
- **Deserción acumulada Universitario**: 40.98%
- **Graduación TyT**: 31.31%
- **Graduación Universitario**: 44.84%

---

## 🤖 Modelo de Machine Learning

### Variables de Entrada (Features)
```python
features = {
    'promedio_academico': float,      # 0.0 - 5.0
    'asistencia': float,              # 0.0 - 100.0 (%)
    'creditos_aprobados': int,        # Créditos aprobados
    'creditos_totales': int,          # Créditos matriculados
    'estrato': int,                   # 1-6
    'tiene_beca': bool,               # True/False
    'trabaja': bool,                  # True/False
    'edad': int,                      # Años
    'semestre': int,                  # Semestre actual
    'uso_plataforma': float,          # Horas semanales
    'distancia_campus': float,        # km
    'nivel_formacion': str,           # 'TyT' / 'Universitario'
}
```

### Salida del Modelo
```json
{
    "riesgo_desercion": 0.73,
    "clasificacion": "Alto",
    "factores_principales": [
        {"factor": "promedio_academico", "impacto": -0.25},
        {"factor": "trabaja", "impacto": 0.15},
        {"factor": "asistencia", "impacto": -0.12}
    ],
    "recomendaciones": [
        "Buscar tutorías académicas",
        "Explorar opciones de beca o financiamiento",
        "Contactar al programa de bienestar estudiantil"
    ]
}
```

---

## 📈 Impacto Esperado

1. **Estudiantes**: Identificar riesgo temprano y acceder a recursos
2. **Instituciones**: Datos para diseñar programas de retención
3. **Gobierno**: Monitoreo agregado para políticas públicas
4. **Investigadores**: API pública para estudios académicos

---

## 🔒 Privacidad y Ética

- **Sin datos personales**: La calculadora no almacena información del usuario
- **Datos agregados**: Solo se publican estadísticas a nivel institucional
- **Código abierto**: Transparencia total en el algoritmo
- **Cumplimiento**: Alineado con Ley 1581 de 2012 (Habeas Data)

---

## 👥 Equipo

Proyecto desarrollado para el Foro de Machine Learning - UNIMINUTO 2025

---

## 📖 Referencias

- SPADIES 3.0 - Ministerio de Educación Nacional de Colombia
- Barrero Ortiz, G. (2020). Machine Learning: 50 Conceptos Clave para Entenderlo
- Véliz Capuñay, C. (2020). Aprendizaje automático: Introducción al aprendizaje profundo

---

## 📄 Licencia

MIT License - Uso libre con atribución
