"""
Generador de Datos Sintéticos para Estudiantes
===============================================

Este módulo genera datos sintéticos realistas de estudiantes universitarios
para entrenar y validar modelos de predicción de deserción.

Componentes de ML demostrados:
- Generación de datos de entrada (features)
- Creación de variable objetivo (target)
- Simulación de patrones realistas de deserción
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

class StudentDataGenerator:
    """
    Genera datos sintéticos de estudiantes para simulación de deserción.
    
    Esta clase crea un dataset realista que simula las características
    de estudiantes universitarios colombianos, incluyendo factores
    académicos, socioeconómicos y de comportamiento.
    
    Attributes:
        n_students (int): Número de estudiantes a generar
        dropout_rate (float): Tasa base de deserción (0.0 a 1.0)
        random_state (int): Semilla para reproducibilidad
    
    Example:
        >>> generator = StudentDataGenerator(n_students=1000, dropout_rate=0.25)
        >>> df = generator.generate()
        >>> print(df.shape)
        (1000, 16)
    """
    
    def __init__(
        self, 
        n_students: int = 1000, 
        dropout_rate: float = 0.25,
        random_state: int = 42
    ):
        """
        Inicializa el generador de datos.
        
        Args:
            n_students: Número de estudiantes a generar
            dropout_rate: Tasa base de deserción (25% por defecto)
            random_state: Semilla para reproducibilidad de resultados
        """
        self.n_students = n_students
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate(self) -> pd.DataFrame:
        """
        Genera el dataset completo de estudiantes.
        
        Returns:
            DataFrame con todas las características de los estudiantes
            
        El dataset incluye:
        - Variables académicas: notas, asistencia, créditos
        - Variables socioeconómicas: estrato, financiamiento, trabajo
        - Variables de comportamiento: uso de plataforma, tutorías
        - Variables demográficas: edad, género, distancia
        - Variable objetivo: desercion (0 o 1)
        """
        data = {
            # Identificador único
            'estudiante_id': range(1, self.n_students + 1),
            
            # ==========================================
            # DATOS ACADÉMICOS
            # ==========================================
            
            # Promedio de notas (0.0 a 5.0, escala colombiana)
            'promedio_notas': self._generate_grades(),
            
            # Porcentaje de asistencia (0 a 100)
            'asistencia_porcentaje': self._generate_attendance(),
            
            # Créditos aprobados vs matriculados (ratio 0 a 1)
            'ratio_creditos_aprobados': self._generate_credits_ratio(),
            
            # Número de materias perdidas
            'materias_perdidas': self._generate_failed_courses(),
            
            # ==========================================
            # DATOS SOCIOECONÓMICOS
            # ==========================================
            
            # Estrato socioeconómico (1 a 6, Colombia)
            'estrato': np.random.choice([1, 2, 3, 4, 5, 6], 
                                        size=self.n_students,
                                        p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.05]),
            
            # Tipo de financiamiento
            'financiamiento': np.random.choice(
                ['propio', 'credito_icetex', 'beca_completa', 'beca_parcial', 'patrocinio'],
                size=self.n_students,
                p=[0.40, 0.25, 0.10, 0.15, 0.10]
            ),
            
            # ¿Trabaja mientras estudia?
            'trabaja': np.random.choice([0, 1], size=self.n_students, p=[0.45, 0.55]),
            
            # Horas de trabajo por semana (si trabaja)
            'horas_trabajo_semana': self._generate_work_hours(),
            
            # ==========================================
            # DATOS DE COMPORTAMIENTO
            # ==========================================
            
            # Horas semanales en plataforma virtual
            'horas_plataforma_semana': np.clip(
                np.random.exponential(scale=5, size=self.n_students), 0, 40
            ).round(1),
            
            # Número de interacciones con tutorías
            'interacciones_tutorias': np.random.poisson(lam=3, size=self.n_students),
            
            # Participación en actividades extracurriculares (0-10 actividades)
            'actividades_extracurriculares': np.random.poisson(lam=2, size=self.n_students),
            
            # ==========================================
            # DATOS DEMOGRÁFICOS
            # ==========================================
            
            # Edad del estudiante
            'edad': np.clip(
                np.random.normal(loc=21, scale=3, size=self.n_students), 17, 50
            ).astype(int),
            
            # Género
            'genero': np.random.choice(['M', 'F', 'Otro'], 
                                       size=self.n_students,
                                       p=[0.48, 0.50, 0.02]),
            
            # Distancia al campus (km)
            'distancia_campus_km': np.clip(
                np.random.exponential(scale=15, size=self.n_students), 0, 100
            ).round(1),
            
            # Semestre actual
            'semestre': np.random.choice(range(1, 11), size=self.n_students,
                                         p=[0.20, 0.15, 0.12, 0.10, 0.10, 
                                            0.08, 0.08, 0.07, 0.05, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Generar variable objetivo (deserción) basada en factores de riesgo
        df['desercion'] = self._generate_dropout_target(df)
        
        return df
    
    def _generate_grades(self) -> np.ndarray:
        """Genera notas con distribución realista."""
        # Distribución bimodal: estudiantes buenos y con dificultades
        grades = np.concatenate([
            np.random.normal(loc=3.8, scale=0.4, size=int(self.n_students * 0.7)),
            np.random.normal(loc=2.5, scale=0.5, size=int(self.n_students * 0.3))
        ])
        np.random.shuffle(grades)
        return np.clip(grades[:self.n_students], 0, 5).round(2)
    
    def _generate_attendance(self) -> np.ndarray:
        """Genera porcentaje de asistencia."""
        attendance = np.random.beta(a=5, b=1.5, size=self.n_students) * 100
        return attendance.round(1)
    
    def _generate_credits_ratio(self) -> np.ndarray:
        """Genera ratio de créditos aprobados."""
        ratio = np.random.beta(a=4, b=1.5, size=self.n_students)
        return ratio.round(2)
    
    def _generate_failed_courses(self) -> np.ndarray:
        """Genera número de materias perdidas."""
        return np.random.geometric(p=0.6, size=self.n_students) - 1
    
    def _generate_work_hours(self) -> np.ndarray:
        """Genera horas de trabajo semanales."""
        # Simulamos que algunos no trabajan (0 horas)
        hours = np.random.choice(
            [0, 10, 20, 30, 40, 48],
            size=self.n_students,
            p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05]
        )
        return hours
    
    def _generate_dropout_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Genera la variable objetivo de deserción basada en factores de riesgo.
        
        El modelo de generación considera:
        - Bajo rendimiento académico aumenta probabilidad
        - Baja asistencia aumenta probabilidad
        - Trabajar muchas horas aumenta probabilidad
        - Bajo estrato con financiamiento propio aumenta probabilidad
        - Poca interacción con la universidad aumenta probabilidad
        """
        # Calcular score de riesgo para cada estudiante
        risk_score = np.zeros(self.n_students)
        
        # Factor académico (más peso)
        risk_score += (5 - df['promedio_notas']) * 0.15
        risk_score += (100 - df['asistencia_porcentaje']) * 0.005
        risk_score += (1 - df['ratio_creditos_aprobados']) * 0.1
        risk_score += df['materias_perdidas'] * 0.05
        
        # Factor socioeconómico
        risk_score += (7 - df['estrato']) * 0.02
        risk_score += (df['financiamiento'] == 'propio').astype(float) * 0.1
        risk_score += df['horas_trabajo_semana'] * 0.003
        
        # Factor comportamiento (engagement negativo = riesgo)
        risk_score += (20 - df['horas_plataforma_semana'].clip(0, 20)) * 0.01
        risk_score += (10 - df['interacciones_tutorias'].clip(0, 10)) * 0.02
        
        # Factor demográfico
        risk_score += (df['distancia_campus_km'] > 30).astype(float) * 0.05
        
        # Normalizar probabilidad y agregar ruido
        prob_dropout = 1 / (1 + np.exp(-risk_score + 1))  # Sigmoid
        prob_dropout = np.clip(prob_dropout, 0.05, 0.95)
        
        # Ajustar para tasa de deserción objetivo
        prob_dropout = prob_dropout * (self.dropout_rate / prob_dropout.mean())
        prob_dropout = np.clip(prob_dropout, 0, 1)
        
        # Generar deserciones
        dropout = np.random.binomial(1, prob_dropout)
        
        return dropout
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Retorna descripciones de todas las características.
        
        Returns:
            Diccionario con nombre y descripción de cada feature
        """
        return {
            'estudiante_id': 'Identificador único del estudiante',
            'promedio_notas': 'Promedio académico (escala 0-5)',
            'asistencia_porcentaje': 'Porcentaje de asistencia a clases (0-100%)',
            'ratio_creditos_aprobados': 'Ratio de créditos aprobados vs matriculados',
            'materias_perdidas': 'Número total de materias perdidas',
            'estrato': 'Estrato socioeconómico (1-6)',
            'financiamiento': 'Tipo de financiamiento de estudios',
            'trabaja': 'Indicador si trabaja mientras estudia',
            'horas_trabajo_semana': 'Horas de trabajo por semana',
            'horas_plataforma_semana': 'Horas semanales en plataforma virtual',
            'interacciones_tutorias': 'Número de interacciones con servicio de tutorías',
            'actividades_extracurriculares': 'Número de actividades extracurriculares',
            'edad': 'Edad del estudiante',
            'genero': 'Género del estudiante',
            'distancia_campus_km': 'Distancia al campus en kilómetros',
            'semestre': 'Semestre actual cursando',
            'desercion': 'Variable objetivo: 1=Desertó, 0=Continúa'
        }
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera estadísticas descriptivas del dataset.
        
        Args:
            df: DataFrame generado
            
        Returns:
            DataFrame con estadísticas resumidas
        """
        summary = df.describe(include='all').T
        summary['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
        return summary


# Función de conveniencia para uso rápido
def generate_student_data(
    n_students: int = 1000,
    dropout_rate: float = 0.25,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Función de conveniencia para generar datos de estudiantes.
    
    Args:
        n_students: Número de estudiantes
        dropout_rate: Tasa de deserción objetivo
        random_state: Semilla aleatoria
        
    Returns:
        DataFrame con datos de estudiantes
    """
    generator = StudentDataGenerator(n_students, dropout_rate, random_state)
    return generator.generate()


if __name__ == "__main__":
    # Ejemplo de uso
    print("=" * 60)
    print("GENERADOR DE DATOS - PREDICCIÓN DE DESERCIÓN ESTUDIANTIL")
    print("=" * 60)
    
    # Generar datos
    generator = StudentDataGenerator(n_students=1000, dropout_rate=0.25)
    df = generator.generate()
    
    print(f"\n📊 Dataset generado: {df.shape[0]} estudiantes, {df.shape[1]} características")
    print(f"\n📈 Tasa de deserción: {df['desercion'].mean()*100:.1f}%")
    
    print("\n📋 Primeras 5 filas:")
    print(df.head())
    
    print("\n📊 Estadísticas descriptivas:")
    print(df.describe())
