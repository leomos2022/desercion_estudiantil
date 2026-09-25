"""API de prediccion de desercion para Vercel."""

from functools import lru_cache
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from src.data_generator import StudentDataGenerator
from src.preprocessing import DataPreprocessor


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


@lru_cache(maxsize=1)
def get_model() -> tuple[DataPreprocessor, LogisticRegression]:
    """Entrena una vez por instancia serverless y reutiliza el modelo."""
    dataset = StudentDataGenerator(
        n_students=1000,
        dropout_rate=0.25,
        random_state=42,
    ).generate()
    preprocessor = DataPreprocessor(
        scaling_method="standard",
        imputation_method="mean",
        balance_method=None,
    )
    x_train, _, y_train, _ = preprocessor.fit_transform(
        dataset,
        target_column="desercion",
        test_size=0.2,
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    return preprocessor, model


app = FastAPI(
    title="API de Desercion Estudiantil",
    version="1.0.0",
)


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "message": "API de prediccion disponible"}


@app.post("/predict")
def predict(student: StudentInput) -> Dict[str, Any]:
    try:
        preprocessor, model = get_model()
        student_data = pd.DataFrame([student.model_dump()])
        transformed_data = preprocessor.transform(student_data)
        probability = float(model.predict_proba(transformed_data)[0, 1])
        risk_level = (
            "ALTO" if probability >= 0.65 else
            "MEDIO" if probability >= 0.35 else
            "BAJO"
        )
        return {
            "probabilidad_desercion": round(probability, 4),
            "porcentaje_riesgo": round(probability * 100, 2),
            "nivel_riesgo": risk_level,
            "prediccion": int(probability >= 0.5),
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error