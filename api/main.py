"""
Alerta Estudiantil Colombia - API Principal
Sistema de Predicción de Deserción Estudiantil con Machine Learning
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Configuración de la API
app = FastAPI(
    title="🎓 Alerta Estudiantil Colombia",
    description="""
    ## API de Predicción de Deserción Estudiantil
    
    Sistema de Machine Learning para predecir el riesgo de deserción 
    estudiantil en instituciones de educación superior colombianas.
    
    ### Funcionalidades:
    - **Predicción Individual**: Calcula el riesgo de deserción de un estudiante
    - **Estadísticas Nacionales**: Datos agregados del sistema SPADIES
    - **Consulta por IES**: Información específica por institución
    
    ### Fuente de Datos:
    SPADIES 3.0 - Ministerio de Educación Nacional de Colombia
    """,
    version="1.0.0",
    contact={
        "name": "Proyecto UNIMINUTO",
        "url": "https://github.com/tu-usuario/alerta-estudiantil",
    },
    license_info={
        "name": "MIT",
    }
)

# CORS - permitir acceso desde cualquier origen (ajustar en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de archivos
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "desercion_model.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
DATA_PATH = BASE_DIR.parent / "DESERCION.xlsx"

# Cargar modelo si existe
model = None
scaler = None

def load_model():
    global model, scaler
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return True
    return False

# Intentar cargar modelo al inicio
load_model()


# ============== Schemas ==============

class EstudianteInput(BaseModel):
    """Datos de entrada para predicción de deserción"""
    promedio_academico: float = Field(..., ge=0, le=5, description="Promedio académico (0.0 - 5.0)")
    asistencia: float = Field(..., ge=0, le=100, description="Porcentaje de asistencia (0-100)")
    creditos_aprobados: int = Field(..., ge=0, description="Créditos aprobados")
    creditos_totales: int = Field(..., ge=1, description="Créditos matriculados totales")
    estrato: int = Field(..., ge=1, le=6, description="Estrato socioeconómico (1-6)")
    tiene_beca: bool = Field(..., description="¿Tiene beca o apoyo financiero?")
    trabaja: bool = Field(..., description="¿Trabaja actualmente?")
    edad: int = Field(..., ge=15, le=80, description="Edad en años")
    semestre: int = Field(..., ge=1, le=20, description="Semestre actual")
    uso_plataforma: float = Field(..., ge=0, description="Horas semanales de uso de plataforma virtual")
    distancia_campus: float = Field(..., ge=0, description="Distancia al campus en km")
    nivel_formacion: str = Field(..., description="'TyT' o 'Universitario'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "promedio_academico": 3.2,
                "asistencia": 75.0,
                "creditos_aprobados": 45,
                "creditos_totales": 60,
                "estrato": 3,
                "tiene_beca": False,
                "trabaja": True,
                "edad": 22,
                "semestre": 4,
                "uso_plataforma": 5.0,
                "distancia_campus": 15.0,
                "nivel_formacion": "Universitario"
            }
        }


class FactorImpacto(BaseModel):
    factor: str
    impacto: float
    descripcion: str


class PrediccionResponse(BaseModel):
    riesgo_desercion: float = Field(..., description="Probabilidad de deserción (0-1)")
    porcentaje_riesgo: float = Field(..., description="Riesgo como porcentaje")
    clasificacion: str = Field(..., description="Alto/Medio/Bajo")
    factores_principales: List[FactorImpacto]
    recomendaciones: List[str]
    confianza_modelo: float


class EstadisticasNacionales(BaseModel):
    total_ies: int
    tda_promedio: float
    tda_mediana: float
    tdca_tyt: float
    tdca_universitario: float
    tga_tyt: float
    tga_universitario: float
    anio_datos: int


class IESInfo(BaseModel):
    codigo: int
    nombre: str
    tda_2023: Optional[float]
    nivel_formacion: Optional[str]
    clasificacion_desercion: str


# ============== Endpoints ==============

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "nombre": "Alerta Estudiantil Colombia",
        "version": "1.0.0",
        "descripcion": "API de predicción de deserción estudiantil con ML",
        "documentacion": "/docs",
        "estado": "activo",
        "modelo_cargado": model is not None
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Verificación de salud del servicio"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_available": DATA_PATH.exists()
    }


@app.post("/predict", response_model=PrediccionResponse, tags=["Predicción"])
async def predecir_desercion(estudiante: EstudianteInput):
    """
    Predice el riesgo de deserción de un estudiante.
    
    Retorna:
    - Probabilidad de deserción (0-1)
    - Clasificación de riesgo (Alto/Medio/Bajo)
    - Factores que más influyen en la predicción
    - Recomendaciones personalizadas
    """
    
    # Si no hay modelo, usar reglas heurísticas
    if model is None:
        return calcular_riesgo_heuristico(estudiante)
    
    # Preparar datos para el modelo
    features = preparar_features(estudiante)
    
    # Escalar features
    features_scaled = scaler.transform([features])
    
    # Predecir
    probabilidad = model.predict_proba(features_scaled)[0][1]
    
    # Clasificar
    if probabilidad >= 0.7:
        clasificacion = "Alto"
    elif probabilidad >= 0.4:
        clasificacion = "Medio"
    else:
        clasificacion = "Bajo"
    
    # Obtener factores de impacto (simplificado)
    factores = obtener_factores_impacto(estudiante, probabilidad)
    
    # Generar recomendaciones
    recomendaciones = generar_recomendaciones(estudiante, clasificacion)
    
    return PrediccionResponse(
        riesgo_desercion=round(probabilidad, 4),
        porcentaje_riesgo=round(probabilidad * 100, 2),
        clasificacion=clasificacion,
        factores_principales=factores,
        recomendaciones=recomendaciones,
        confianza_modelo=0.85
    )


@app.get("/stats", response_model=EstadisticasNacionales, tags=["Estadísticas"])
async def obtener_estadisticas():
    """
    Retorna estadísticas nacionales de deserción estudiantil.
    
    Datos basados en SPADIES 3.0 - Ministerio de Educación Nacional.
    """
    # Valores calculados del archivo DESERCION.xlsx
    return EstadisticasNacionales(
        total_ies=288,
        tda_promedio=15.28,
        tda_mediana=11.49,
        tdca_tyt=51.64,
        tdca_universitario=40.98,
        tga_tyt=31.31,
        tga_universitario=44.84,
        anio_datos=2023
    )


@app.get("/ies", tags=["Estadísticas"])
async def listar_ies(
    limite: int = 20,
    orden: str = "asc",
    nivel: Optional[str] = None
):
    """
    Lista instituciones de educación superior ordenadas por tasa de deserción.
    
    - **limite**: Número máximo de resultados (default: 20)
    - **orden**: 'asc' (menor deserción primero) o 'desc' (mayor deserción primero)
    - **nivel**: Filtrar por 'TyT' o 'Universitario'
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Datos no disponibles")
        
        # Leer datos
        df = pd.read_excel(DATA_PATH, sheet_name='TDA IES Padre Total', header=12)
        df.columns = ['CODIGO_IES', 'IES', '2019', '2020', '2021', '2022', '2023'] + list(df.columns[7:])
        
        # Filtrar y ordenar
        df_clean = df[['CODIGO_IES', 'IES', '2023']].dropna()
        
        ascending = orden == "asc"
        df_sorted = df_clean.sort_values('2023', ascending=ascending).head(limite)
        
        result = []
        for _, row in df_sorted.iterrows():
            tda = row['2023'] * 100
            if tda < 10:
                clasificacion = "Muy Bajo"
            elif tda < 20:
                clasificacion = "Bajo"
            elif tda < 35:
                clasificacion = "Medio"
            else:
                clasificacion = "Alto"
            
            result.append({
                "codigo": int(row['CODIGO_IES']),
                "nombre": str(row['IES']),
                "tda_2023": round(tda, 2),
                "clasificacion_desercion": clasificacion
            })
        
        return {
            "total": len(result),
            "orden": "menor_desercion" if ascending else "mayor_desercion",
            "instituciones": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ies/{codigo}", response_model=IESInfo, tags=["Estadísticas"])
async def obtener_ies(codigo: int):
    """Obtiene información de una institución específica por su código"""
    try:
        if not DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Datos no disponibles")
        
        df = pd.read_excel(DATA_PATH, sheet_name='TDA IES Padre Total', header=12)
        df.columns = ['CODIGO_IES', 'IES', '2019', '2020', '2021', '2022', '2023'] + list(df.columns[7:])
        
        ies = df[df['CODIGO_IES'] == codigo]
        
        if ies.empty:
            raise HTTPException(status_code=404, detail=f"IES con código {codigo} no encontrada")
        
        row = ies.iloc[0]
        tda = row['2023'] * 100 if pd.notna(row['2023']) else None
        
        if tda is None:
            clasificacion = "Sin datos"
        elif tda < 10:
            clasificacion = "Muy Bajo"
        elif tda < 20:
            clasificacion = "Bajo"
        elif tda < 35:
            clasificacion = "Medio"
        else:
            clasificacion = "Alto"
        
        return IESInfo(
            codigo=codigo,
            nombre=str(row['IES']),
            tda_2023=round(tda, 2) if tda else None,
            nivel_formacion=None,
            clasificacion_desercion=clasificacion
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Funciones Auxiliares ==============

def preparar_features(estudiante: EstudianteInput) -> list:
    """Convierte el input del estudiante en features para el modelo"""
    ratio_creditos = estudiante.creditos_aprobados / estudiante.creditos_totales
    nivel_num = 0 if estudiante.nivel_formacion == "TyT" else 1
    
    return [
        estudiante.promedio_academico,
        estudiante.asistencia,
        ratio_creditos,
        estudiante.estrato,
        int(estudiante.tiene_beca),
        int(estudiante.trabaja),
        estudiante.edad,
        estudiante.semestre,
        estudiante.uso_plataforma,
        estudiante.distancia_campus,
        nivel_num
    ]


def calcular_riesgo_heuristico(estudiante: EstudianteInput) -> PrediccionResponse:
    """Calcula riesgo usando reglas heurísticas cuando no hay modelo ML"""
    riesgo = 0.5  # Base
    
    # Factores académicos
    if estudiante.promedio_academico < 3.0:
        riesgo += 0.2
    elif estudiante.promedio_academico >= 4.0:
        riesgo -= 0.15
    
    if estudiante.asistencia < 70:
        riesgo += 0.15
    elif estudiante.asistencia >= 90:
        riesgo -= 0.1
    
    # Ratio de créditos
    ratio = estudiante.creditos_aprobados / estudiante.creditos_totales
    if ratio < 0.6:
        riesgo += 0.15
    elif ratio >= 0.9:
        riesgo -= 0.1
    
    # Factores socioeconómicos
    if estudiante.estrato <= 2 and not estudiante.tiene_beca:
        riesgo += 0.1
    
    if estudiante.tiene_beca:
        riesgo -= 0.1
    
    if estudiante.trabaja:
        riesgo += 0.1
    
    # Uso de plataforma
    if estudiante.uso_plataforma < 2:
        riesgo += 0.1
    elif estudiante.uso_plataforma >= 10:
        riesgo -= 0.05
    
    # Distancia
    if estudiante.distancia_campus > 30:
        riesgo += 0.05
    
    # Semestre crítico (1-3)
    if estudiante.semestre <= 3:
        riesgo += 0.1
    
    # Nivel de formación
    if estudiante.nivel_formacion == "TyT":
        riesgo += 0.05  # Mayor deserción histórica en TyT
    
    # Limitar entre 0 y 1
    riesgo = max(0.05, min(0.95, riesgo))
    
    # Clasificar
    if riesgo >= 0.7:
        clasificacion = "Alto"
    elif riesgo >= 0.4:
        clasificacion = "Medio"
    else:
        clasificacion = "Bajo"
    
    factores = obtener_factores_impacto(estudiante, riesgo)
    recomendaciones = generar_recomendaciones(estudiante, clasificacion)
    
    return PrediccionResponse(
        riesgo_desercion=round(riesgo, 4),
        porcentaje_riesgo=round(riesgo * 100, 2),
        clasificacion=clasificacion,
        factores_principales=factores,
        recomendaciones=recomendaciones,
        confianza_modelo=0.7  # Menor confianza para heurístico
    )


def obtener_factores_impacto(estudiante: EstudianteInput, riesgo: float) -> List[FactorImpacto]:
    """Identifica los factores que más impactan en el riesgo"""
    factores = []
    
    # Promedio académico
    if estudiante.promedio_academico < 3.0:
        factores.append(FactorImpacto(
            factor="promedio_academico",
            impacto=0.25,
            descripcion="Promedio académico bajo aumenta significativamente el riesgo"
        ))
    elif estudiante.promedio_academico >= 4.0:
        factores.append(FactorImpacto(
            factor="promedio_academico",
            impacto=-0.15,
            descripcion="Buen promedio académico reduce el riesgo"
        ))
    
    # Asistencia
    if estudiante.asistencia < 70:
        factores.append(FactorImpacto(
            factor="asistencia",
            impacto=0.15,
            descripcion="Baja asistencia es un indicador temprano de deserción"
        ))
    
    # Trabajo
    if estudiante.trabaja:
        factores.append(FactorImpacto(
            factor="trabaja",
            impacto=0.1,
            descripcion="Trabajar mientras se estudia puede dificultar el rendimiento"
        ))
    
    # Beca
    if estudiante.tiene_beca:
        factores.append(FactorImpacto(
            factor="tiene_beca",
            impacto=-0.1,
            descripcion="El apoyo financiero reduce la probabilidad de deserción"
        ))
    
    # Estrato
    if estudiante.estrato <= 2:
        factores.append(FactorImpacto(
            factor="estrato",
            impacto=0.08,
            descripcion="Estratos bajos tienen mayor vulnerabilidad económica"
        ))
    
    # Ordenar por impacto absoluto
    factores.sort(key=lambda x: abs(x.impacto), reverse=True)
    
    return factores[:5]


def generar_recomendaciones(estudiante: EstudianteInput, clasificacion: str) -> List[str]:
    """Genera recomendaciones personalizadas según el perfil"""
    recomendaciones = []
    
    if clasificacion in ["Alto", "Medio"]:
        recomendaciones.append("📞 Contactar al programa de bienestar estudiantil de tu institución")
    
    if estudiante.promedio_academico < 3.5:
        recomendaciones.append("📚 Solicitar tutorías académicas en las materias con menor rendimiento")
    
    if estudiante.asistencia < 80:
        recomendaciones.append("📅 Establecer un horario fijo de clases y comprometerse con la asistencia")
    
    if not estudiante.tiene_beca and estudiante.estrato <= 3:
        recomendaciones.append("💰 Explorar opciones de becas, créditos ICETEX o apoyos institucionales")
    
    if estudiante.trabaja:
        recomendaciones.append("⚖️ Evaluar opciones de horarios flexibles o modalidad virtual")
    
    if estudiante.uso_plataforma < 5:
        recomendaciones.append("💻 Aumentar el uso de recursos virtuales y material complementario")
    
    if estudiante.semestre <= 2:
        recomendaciones.append("👥 Participar en grupos de estudio y actividades de integración")
    
    if not recomendaciones:
        recomendaciones.append("✅ Mantener el buen desempeño actual y buscar oportunidades de crecimiento")
    
    return recomendaciones[:5]


# Servir archivos estáticos del frontend si existen
frontend_path = BASE_DIR / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    @app.get("/app", tags=["Frontend"])
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
