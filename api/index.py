"""
API de Predicción de Deserción Estudiantil
Vercel Serverless Function - Self-contained
"""

import os
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Pydantic model for input validation
# ──────────────────────────────────────────────

class StudentInput(BaseModel):
    promedio_notas: float = Field(default=3.0, ge=0, le=5, description="Promedio académico (0-5)")
    asistencia_porcentaje: float = Field(default=75.0, ge=0, le=100, description="Porcentaje asistencia (0-100)")
    ratio_creditos_aprobados: float = Field(default=0.75, ge=0, le=1, description="Ratio créditos aprobados (0-1)")
    materias_perdidas: int = Field(default=1, ge=0, description="Número de materias perdidas")
    estrato: int = Field(default=2, ge=1, le=6, description="Estrato socioeconómico (1-6)")
    financiamiento: str = Field(default="propio", description="Tipo de financiamiento")
    trabaja: int = Field(default=0, ge=0, le=1, description="¿Trabaja? (0=No, 1=Sí)")
    horas_trabajo_semana: float = Field(default=0.0, ge=0, le=80, description="Horas de trabajo por semana")
    horas_plataforma_semana: float = Field(default=5.0, ge=0, le=40, description="Horas en plataforma por semana")
    interacciones_tutorias: int = Field(default=2, ge=0, description="Interacciones con tutorías")
    actividades_extracurriculares: int = Field(default=2, ge=0, description="Actividades extracurriculares")
    edad: int = Field(default=21, ge=17, le=100, description="Edad del estudiante")
    genero: str = Field(default="F", description="Género (M/F/Otro)")
    distancia_campus_km: float = Field(default=10.0, ge=0, le=500, description="Distancia al campus (km)")
    semestre: int = Field(default=1, ge=1, le=20, description="Semestre actual")


# ──────────────────────────────────────────────
# Constants matching training distribution
# ──────────────────────────────────────────────

CAT_COLUMNS = ["financiamiento", "genero"]
FINANCIAMIENTO_CATS = ["beca_completa", "beca_parcial", "credito_icetex", "patrocinio", "propio"]
GENERO_CATS = ["F", "M", "Otro"]


# ──────────────────────────────────────────────
# Lightweight Preprocessor
# ──────────────────────────────────────────────

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.feature_names: List[str] = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
        X = df.drop(columns=[target, "estudiante_id"], errors="ignore").copy()
        y = df[target].values

        X = pd.get_dummies(X, columns=CAT_COLUMNS, drop_first=True)
        self.feature_names = list(X.columns)

        X_arr = self.imputer.fit_transform(X)
        X_arr = self.scaler.fit_transform(X_arr)
        self.fitted = True
        return X_arr, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        # Drop extra columns if present
        X = X.drop(columns=["estudiante_id", "desercion"], errors="ignore")
        X = pd.get_dummies(X, columns=CAT_COLUMNS, drop_first=True)

        # Align with training columns
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_arr = self.imputer.transform(X)
        return self.scaler.transform(X_arr)


# ──────────────────────────────────────────────
# Synthetic data generator (embedded, no src/)
# ──────────────────────────────────────────────

def generate_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    grades = np.concatenate([
        rng.normal(3.8, 0.4, int(n * 0.7)),
        rng.normal(2.5, 0.5, n - int(n * 0.7))
    ])
    rng.shuffle(grades)

    data = {
        "estudiante_id": range(1, n + 1),
        "promedio_notas": np.clip(grades, 0, 5).round(2),
        "asistencia_porcentaje": (rng.beta(5, 1.5, n) * 100).round(1),
        "ratio_creditos_aprobados": rng.beta(4, 1.5, n).round(2),
        "materias_perdidas": rng.geometric(0.6, n) - 1,
        "estrato": rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.05]),
        "financiamiento": rng.choice(FINANCIAMIENTO_CATS, n, p=[0.10, 0.15, 0.25, 0.10, 0.40]),
        "trabaja": rng.choice([0, 1], n, p=[0.45, 0.55]),
        "horas_trabajo_semana": rng.choice([0, 10, 20, 30, 40, 48], n, p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05]),
        "horas_plataforma_semana": np.clip(rng.exponential(5, n), 0, 40).round(1),
        "interacciones_tutorias": rng.poisson(3, n),
        "actividades_extracurriculares": rng.poisson(2, n),
        "edad": np.clip(rng.normal(21, 3, n), 17, 50).astype(int),
        "genero": rng.choice(GENERO_CATS, n, p=[0.50, 0.48, 0.02]),
        "distancia_campus_km": np.clip(rng.exponential(15, n), 0, 100).round(1),
        "semestre": rng.choice(range(1, 11), n, p=[0.20, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05]),
    }

    df = pd.DataFrame(data)

    # Risk model identical to original
    risk = (
        (5 - df["promedio_notas"]) * 0.15
        + (100 - df["asistencia_porcentaje"]) * 0.005
        + (1 - df["ratio_creditos_aprobados"]) * 0.1
        + df["materias_perdidas"] * 0.05
        + (7 - df["estrato"]) * 0.02
        + (df["financiamiento"] == "propio").astype(float) * 0.1
        + df["horas_trabajo_semana"] * 0.003
        + (20 - df["horas_plataforma_semana"].clip(0, 20)) * 0.01
        + (10 - df["interacciones_tutorias"].clip(0, 10)) * 0.02
        + (df["distancia_campus_km"] > 30).astype(float) * 0.05
    )
    prob = 1 / (1 + np.exp(-risk.values + 1))
    prob = np.clip(prob * (0.25 / prob.mean()), 0, 1)
    df["desercion"] = rng.binomial(1, prob)
    return df


# ──────────────────────────────────────────────
# Model training (cached per instance)
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_model() -> Tuple[Preprocessor, LogisticRegression]:
    df = generate_data(3000, 42)
    pre = Preprocessor()
    X, y = pre.fit_transform(df, "desercion")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return pre, model


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="API Deserción Estudiantil",
    description="Predice el riesgo de deserción de estudiantes universitarios.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "message": "API de Deserción Estudiantil v2.0"}


@app.post("/predict", tags=["Predicción"])
def predict(student: StudentInput) -> Dict[str, Any]:
    pre, model = get_model()
    df = pd.DataFrame([student.model_dump()])
    X = pre.transform(df)
    prob = float(model.predict_proba(X)[0, 1])

    if prob >= 0.65:
        nivel = "ALTO"
        color = "#e74c3c"
        recomendacion = "Intervención urgente requerida. Contactar al estudiante inmediatamente."
    elif prob >= 0.35:
        nivel = "MEDIO"
        color = "#f39c12"
        recomendacion = "Monitoreo frecuente recomendado. Ofrecer apoyo académico y psicológico."
    else:
        nivel = "BAJO"
        color = "#27ae60"
        recomendacion = "Estudiante en buen camino. Mantener acompañamiento regular."

    return {
        "probabilidad_desercion": round(prob, 4),
        "porcentaje_riesgo": round(prob * 100, 2),
        "nivel_riesgo": nivel,
        "color": color,
        "prediccion": int(prob >= 0.5),
        "recomendacion": recomendacion,
    }
