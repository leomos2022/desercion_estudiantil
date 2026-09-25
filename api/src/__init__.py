"""
Sistema de Predicción de Deserción Estudiantil - API Module
===========================================================

Lightweight module for Vercel serverless deployment.
"""

from .data_generator import StudentDataGenerator
from .preprocessing import DataPreprocessor

__version__ = "1.0.0"
__all__ = [
    "StudentDataGenerator",
    "DataPreprocessor",
]
