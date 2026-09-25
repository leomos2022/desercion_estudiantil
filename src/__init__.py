"""
Sistema de Predicción de Deserción Estudiantil
==============================================

Un sistema completo de Machine Learning para predecir la deserción estudiantil
en instituciones de educación superior.

Módulos:
    - data_generator: Generación de datos sintéticos para simulación
    - preprocessing: Limpieza y preparación de datos
    - models: Entrenamiento de modelos de ML
    - evaluation: Métricas y evaluación de modelos
    - alert_system: Sistema de alertas tempranas

Autor: Proyecto académico para Componentes de Machine Learning
"""

from .data_generator import StudentDataGenerator
from .preprocessing import DataPreprocessor
from .models import DropoutPredictor
from .evaluation import ModelEvaluator
from .alert_system import EarlyAlertSystem

__version__ = "1.0.0"
__all__ = [
    "StudentDataGenerator",
    "DataPreprocessor", 
    "DropoutPredictor",
    "ModelEvaluator",
    "EarlyAlertSystem"
]
