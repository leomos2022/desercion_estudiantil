"""API de predicción de deserción para Vercel - Self-contained."""

from functools import lru_cache
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import os


class StudentInput(BaseModel):
    promedio_notas: float = Field(default=3.0, ge=0, le=5)
    asistencia_porcentaje: float = Field(default=75, ge=0, le=100)
    ratio_creditos_aprobados: float = Field(default=0.75, ge=0, le=1)
    materias_perdidas: int = Field(default=1, ge=0)
    estrato: int = Field(default=2, ge=1, le=6)
    financiamiento: str = "propio"
    trabaja: int = Field(default=0, ge=0, le=1)
    horas_trabajo_semana: float = Field(default=0, ge=0, le=80)
    horas_plataforma_semana: float = Field(default=5, ge=0, le=40)
    interacciones_tutorias: int = Field(default=2, ge=0)
    actividades_extracurriculares: int = Field(default=2, ge=0)
    edad: int = Field(default=21, ge=17, le=100)
    genero: str = "F"
    distancia_campus_km: float = Field(default=10, ge=0, le=500)
    semestre: int = Field(default=1, ge=1, le=20)


# Feature engineering constants (must match training)
CAT_COLUMNS = ["financiamiento", "genero"]
NUM_COLUMNS = [
    "promedio_notas", "asistencia_porcentaje", "ratio_creditos_aprobados",
    "materias_perdidas", "estrato", "trabaja", "horas_trabajo_semana",
    "horas_plataforma_semana", "interacciones_tutorias", "actividades_extracurriculares",
    "edad", "distancia_campus_km", "semestre"
]

FINANCIAMIENTO_CATEGORIES = ["beca_completa", "beca_parcial", "credito_icetex", "patrocinio", "propio"]
GENERO_CATEGORIES = ["F", "M", "Otro"]


class SimplePreprocessor:
    """Lightweight preprocessor matching training pipeline."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.feature_names: List[str] = []
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> "SimplePreprocessor":
        # Prepare training data
        X = df.drop(columns=["desercion", "estudiante_id"], errors="ignore")
        y = df["desercion"]
        
        # One-hot encode categoricals
        X_encoded = pd.get_dummies(X, columns=CAT_COLUMNS, drop_first=True)
        self.feature_names = list(X_encoded.columns)
        
        # Fit imputer and scaler
        self.imputer.fit(X_encoded)
        X_imputed = self.imputer.transform(X_encoded)
        self.scaler.fit(X_imputed)
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Preprocessor not fitted")
        
        X = df.copy()
        
        # One-hot encode with same columns
        X_encoded = pd.get_dummies(X, columns=CAT_COLUMNS, drop_first=True)
        
        # Align columns with training
        for col in self.feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.feature_names]
        
        # Impute and scale
        X_imputed = self.imputer.transform(X_encoded)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled


def generate_training_data(n_students: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic training data matching original distribution."""
    np.random.seed(random_state)
    
    data = {
        "estudiante_id": range(1, n_students + 1),
        "promedio_notas": np.clip(
            np.concatenate([
                np.random.normal(3.8, 0.4, int(n_students * 0.7)),
                np.random.normal(2.5, 0.5, int(n_students * 0.3))
            ][:n_students]), 0, 5
        ).round(2),
        "asistencia_porcentaje": (np.random.beta(5, 1.5, n_students) * 100).round(1),
        "ratio_creditos_aprobados": np.random.beta(4, 1.5, n_students).round(2),
        "materias_perdidas": np.random.geometric(0.6, n_students) - 1,
        "estrato": np.random.choice([1, 2, 3, 4, 5, 6], n_students, p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.05]),
        "financiamiento": np.random.choice(
            FINANCIAMIENTO_CATEGORIES, n_students, p=[0.10, 0.15, 0.25, 0.10, 0.40]
        ),
        "trabaja": np.random.choice([0, 1], n_students, p=[0.45, 0.55]),
        "horas_trabajo_semana": np.random.choice([0, 10, 20, 30, 40, 48], n_students, p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05]),
        "horas_plataforma_semana": np.clip(np.random.exponential(5, n_students), 0, 40).round(1),
        "interacciones_tutorias": np.random.poisson(3, n_students),
        "actividades_extracurriculares": np.random.poisson(2, n_students),
        "edad": np.clip(np.random.normal(21, 3, n_students), 17, 50).astype(int),
        "genero": np.random.choice(GENERO_CATEGORIES, n_students, p=[0.50, 0.48, 0.02]),
        "distancia_campus_km": np.clip(np.random.exponential(15, n_students), 0, 100).round(1),
        "semestre": np.random.choice(range(1, 11), n_students, p=[0.20, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target using same logic as original
    risk = np.zeros(n_students)
    risk += (5 - df["promedio_notas"]) * 0.15
    risk += (100 - df["asistencia_porcentaje"]) * 0.005
    risk += (1 - df["ratio_creditos_aprobados"]) * 0.1
    risk += df["materias_perdidas"] * 0.05
    risk += (7 - df["estrato"]) * 0.02
    risk += (df["financiamiento"] == "propio").astype(float) * 0.1
    risk += df["horas_trabajo_semana"] * 0.003
    risk += (20 - df["horas_plataforma_semana"].clip(0, 20)) * 0.01
    risk += (10 - df["interacciones_tutorias"].clip(0, 10)) * 0.02
    risk += (df["distancia_campus_km"] > 30).astype(float) * 0.05
    
    prob = 1 / (1 + np.exp(-risk + 1))
    prob = np.clip(prob, 0.05, 0.95)
    prob = prob * (0.25 / prob.mean())
    prob = np.clip(prob, 0, 1)
    
    df["desercion"] = np.random.binomial(1, prob)
    return df


@lru_cache(maxsize=1)
def get_model() -> Tuple[SimplePreprocessor, LogisticRegression]:
    """Load or train model once per serverless instance."""
    model_path = "api/model.joblib"
    preprocessor_path = "api/preprocessor.joblib"
    
    # Try to load pre-trained artifacts
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return preprocessor, model
    
    # Fallback: train on-the-fly (for first deploy)
    print("Training model on first request...")
    df = generate_training_data(2000, 42)
    preprocessor = SimplePreprocessor().fit(df)
    
    X = preprocessor.transform(df.drop(columns=["desercion", "estudiante_id"]))
    y = df["desercion"].values
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    
    # Save for next time
    os.makedirs("api", exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(model, model_path)
    
    return preprocessor, model


app = FastAPI(title="API de Deserción Estudiantil", version="1.0.0")


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "message": "API de predicción disponible"}


@app.post("/predict")
def predict(student: StudentInput) -> Dict[str, Any]:
    try:
        preprocessor, model = get_model()
        student_data = pd.DataFrame([student.model_dump()])
        transformed = preprocessor.transform(student_data)
        prob = float(model.predict_proba(transformed)[0, 1])
        
        risk_level = "ALTO" if prob >= 0.65 else "MEDIO" if prob >= 0.35 else "BAJO"
        
        return {
            "probabilidad_desercion": round(prob, 4),
            "porcentaje_riesgo": round(prob * 100, 2),
            "nivel_riesgo": risk_level,
            "prediccion": int(prob >= 0.5),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e