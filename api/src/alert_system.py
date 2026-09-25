"""
Sistema de Alertas Tempranas para Deserción Estudiantil
=======================================================

Este módulo implementa un sistema de alertas y recomendaciones
para intervención temprana de estudiantes en riesgo de deserción.

Componentes de ML demostrados:
- Salidas del modelo (scores de riesgo)
- Clasificación por niveles de riesgo
- Sistema de recomendaciones basado en reglas
- Generación de reportes de intervención

Este es el componente de SALIDA del sistema de Machine Learning,
transformando las predicciones del modelo en acciones concretas.

Referencias:
- Barrero Ortiz, G. (2020). Machine Learning: 50 Conceptos Clave (p. 70)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    """Niveles de riesgo de deserción."""
    BAJO = "BAJO"
    MEDIO = "MEDIO"
    ALTO = "ALTO"
    CRITICO = "CRÍTICO"


@dataclass
class StudentAlert:
    """
    Representa una alerta individual para un estudiante.
    
    Attributes:
        student_id: Identificador del estudiante
        risk_score: Probabilidad de deserción (0-1)
        risk_level: Nivel de riesgo categorizado
        risk_factors: Factores que contribuyen al riesgo
        recommendations: Acciones recomendadas
        priority: Prioridad de atención (1-5)
        created_at: Fecha de creación de la alerta
    """
    student_id: int
    risk_score: float
    risk_level: RiskLevel
    risk_factors: List[str]
    recommendations: List[str]
    priority: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la alerta a diccionario."""
        return {
            'student_id': self.student_id,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'risk_factors': self.risk_factors,
            'recommendations': self.recommendations,
            'priority': self.priority,
            'created_at': self.created_at.isoformat()
        }


class EarlyAlertSystem:
    """
    Sistema de Alertas Tempranas para Deserción Estudiantil.
    
    Este sistema toma las predicciones del modelo de ML y las convierte
    en alertas accionables para los equipos de bienestar estudiantil.
    
    Funcionalidades principales:
    1. Clasificación de estudiantes por nivel de riesgo
    2. Identificación de factores de riesgo principales
    3. Generación de recomendaciones personalizadas
    4. Priorización de intervenciones
    5. Reportes para diferentes audiencias
    
    Attributes:
        risk_thresholds: Umbrales para clasificación de riesgo
        alerts: Lista de alertas generadas
        statistics: Estadísticas de riesgo
        
    Example:
        >>> alert_system = EarlyAlertSystem()
        >>> alerts = alert_system.generate_alerts(df_students, predictions)
        >>> report = alert_system.generate_intervention_report()
    """
    
    # Umbrales de riesgo por defecto
    DEFAULT_THRESHOLDS = {
        'bajo': 0.25,      # 0-25%: Bajo riesgo
        'medio': 0.50,     # 25-50%: Riesgo medio
        'alto': 0.75,      # 50-75%: Alto riesgo
        'critico': 1.0     # 75-100%: Riesgo crítico
    }
    
    # Factores de riesgo y sus umbrales
    RISK_FACTORS_CONFIG = {
        'promedio_notas': {
            'threshold': 3.0,
            'direction': 'below',
            'message': 'Bajo rendimiento académico'
        },
        'asistencia_porcentaje': {
            'threshold': 70,
            'direction': 'below',
            'message': 'Asistencia irregular a clases'
        },
        'ratio_creditos_aprobados': {
            'threshold': 0.6,
            'direction': 'below',
            'message': 'Bajo ratio de créditos aprobados'
        },
        'materias_perdidas': {
            'threshold': 2,
            'direction': 'above',
            'message': 'Múltiples materias perdidas'
        },
        'horas_trabajo_semana': {
            'threshold': 30,
            'direction': 'above',
            'message': 'Carga laboral excesiva'
        },
        'horas_plataforma_semana': {
            'threshold': 3,
            'direction': 'below',
            'message': 'Bajo engagement con plataforma virtual'
        },
        'interacciones_tutorias': {
            'threshold': 1,
            'direction': 'below',
            'message': 'No usa servicios de tutoría'
        },
        'distancia_campus_km': {
            'threshold': 40,
            'direction': 'above',
            'message': 'Lejanía al campus dificulta asistencia'
        }
    }
    
    # Recomendaciones por tipo de factor de riesgo
    RECOMMENDATIONS = {
        'academico': [
            "Programar sesiones de tutoría académica personalizada",
            "Incluir en programa de nivelación para materias críticas",
            "Asignar mentor académico de semestres avanzados",
            "Evaluar carga académica y considerar reducción de créditos",
            "Aplicar estrategias de estudio personalizadas"
        ],
        'asistencia': [
            "Contacto telefónico para verificar situación",
            "Ofrecer modalidad híbrida/virtual si está disponible",
            "Identificar barreras de transporte",
            "Flexibilizar horarios si es posible",
            "Vincular con programa de becas de transporte"
        ],
        'socioeconomico': [
            "Vincular con oficina de bienestar universitario",
            "Evaluar opciones de financiamiento y becas",
            "Orientar sobre programas de apoyo económico",
            "Considerar reducción de carga para trabajadores",
            "Informar sobre programas de alimentación estudiantil"
        ],
        'engagement': [
            "Llamada de seguimiento por parte de coordinación",
            "Invitar a actividades de integración estudiantil",
            "Asignar a grupos de estudio colaborativo",
            "Orientar sobre uso efectivo de recursos virtuales",
            "Vincular con monitores y grupos de apoyo"
        ],
        'general': [
            "Entrevista con psicología de bienestar estudiantil",
            "Evaluación integral de situación personal",
            "Diseño de plan de acompañamiento personalizado",
            "Seguimiento semanal por parte de coordinación"
        ]
    }
    
    def __init__(
        self,
        risk_thresholds: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Inicializa el sistema de alertas.
        
        Args:
            risk_thresholds: Umbrales personalizados de riesgo
            feature_names: Nombres de las características
        """
        self.risk_thresholds = risk_thresholds or self.DEFAULT_THRESHOLDS
        self.feature_names = feature_names
        self.alerts: List[StudentAlert] = []
        self.statistics: Dict[str, Any] = {}
        
    def classify_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Clasifica un score de riesgo en nivel categórico.
        
        Args:
            risk_score: Probabilidad de deserción (0-1)
            
        Returns:
            Nivel de riesgo correspondiente
        """
        if risk_score < self.risk_thresholds['bajo']:
            return RiskLevel.BAJO
        elif risk_score < self.risk_thresholds['medio']:
            return RiskLevel.MEDIO
        elif risk_score < self.risk_thresholds['alto']:
            return RiskLevel.ALTO
        else:
            return RiskLevel.CRITICO
    
    def identify_risk_factors(
        self, 
        student_data: pd.Series
    ) -> List[str]:
        """
        Identifica los factores de riesgo presentes en un estudiante.
        
        Args:
            student_data: Datos del estudiante como Series
            
        Returns:
            Lista de factores de riesgo identificados
        """
        risk_factors = []
        
        for factor, config in self.RISK_FACTORS_CONFIG.items():
            if factor not in student_data.index:
                continue
            
            value = student_data[factor]
            
            if config['direction'] == 'below' and value < config['threshold']:
                risk_factors.append(config['message'])
            elif config['direction'] == 'above' and value > config['threshold']:
                risk_factors.append(config['message'])
        
        return risk_factors
    
    def generate_recommendations(
        self, 
        risk_factors: List[str],
        risk_level: RiskLevel
    ) -> List[str]:
        """
        Genera recomendaciones personalizadas basadas en factores de riesgo.
        
        Args:
            risk_factors: Lista de factores de riesgo identificados
            risk_level: Nivel de riesgo del estudiante
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Mapear factores a tipos de recomendación
        factor_types = set()
        
        for factor in risk_factors:
            if 'académico' in factor.lower() or 'notas' in factor.lower() or 'créditos' in factor.lower() or 'materias' in factor.lower():
                factor_types.add('academico')
            if 'asistencia' in factor.lower() or 'distancia' in factor.lower():
                factor_types.add('asistencia')
            if 'laboral' in factor.lower() or 'trabajo' in factor.lower():
                factor_types.add('socioeconomico')
            if 'engagement' in factor.lower() or 'plataforma' in factor.lower() or 'tutoría' in factor.lower():
                factor_types.add('engagement')
        
        # Obtener recomendaciones por tipo
        for factor_type in factor_types:
            recs = self.RECOMMENDATIONS.get(factor_type, [])
            # Agregar más recomendaciones según nivel de riesgo
            if risk_level in [RiskLevel.ALTO, RiskLevel.CRITICO]:
                recommendations.extend(recs[:3])  # Top 3 para alto riesgo
            else:
                recommendations.extend(recs[:1])  # Solo 1 para bajo/medio
        
        # Siempre agregar recomendación general para alto riesgo
        if risk_level in [RiskLevel.ALTO, RiskLevel.CRITICO]:
            recommendations.extend(self.RECOMMENDATIONS['general'][:2])
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        
        return unique_recs
    
    def calculate_priority(
        self, 
        risk_score: float, 
        risk_level: RiskLevel,
        num_risk_factors: int
    ) -> int:
        """
        Calcula la prioridad de intervención (1=máxima, 5=mínima).
        
        Args:
            risk_score: Score de riesgo
            risk_level: Nivel de riesgo
            num_risk_factors: Número de factores de riesgo
            
        Returns:
            Prioridad de 1 a 5
        """
        if risk_level == RiskLevel.CRITICO:
            return 1
        elif risk_level == RiskLevel.ALTO:
            return 2 if num_risk_factors >= 3 else 3
        elif risk_level == RiskLevel.MEDIO:
            return 3 if num_risk_factors >= 2 else 4
        else:
            return 5
    
    def generate_alerts(
        self,
        df_students: pd.DataFrame,
        risk_scores: np.ndarray,
        student_id_column: str = 'estudiante_id'
    ) -> List[StudentAlert]:
        """
        Genera alertas para todos los estudiantes.
        
        Args:
            df_students: DataFrame con datos de estudiantes
            risk_scores: Array con probabilidades de deserción
            student_id_column: Nombre de columna con ID de estudiante
            
        Returns:
            Lista de alertas generadas
        """
        print("=" * 60)
        print("🚨 GENERANDO ALERTAS DE DESERCIÓN")
        print("=" * 60)
        
        self.alerts = []
        
        for idx, (_, student) in enumerate(df_students.iterrows()):
            if idx >= len(risk_scores):
                break
                
            risk_score = risk_scores[idx]
            risk_level = self.classify_risk_level(risk_score)
            risk_factors = self.identify_risk_factors(student)
            recommendations = self.generate_recommendations(risk_factors, risk_level)
            priority = self.calculate_priority(risk_score, risk_level, len(risk_factors))
            
            # Obtener ID del estudiante
            if student_id_column in student.index:
                student_id = int(student[student_id_column])
            else:
                student_id = idx + 1
            
            alert = StudentAlert(
                student_id=student_id,
                risk_score=float(risk_score),
                risk_level=risk_level,
                risk_factors=risk_factors,
                recommendations=recommendations,
                priority=priority
            )
            
            self.alerts.append(alert)
        
        # Calcular estadísticas
        self._calculate_statistics()
        
        print(f"\n📊 RESUMEN DE ALERTAS:")
        print(f"   Total estudiantes analizados: {len(self.alerts)}")
        print(f"   Riesgo Crítico: {self.statistics['critico']} ({self.statistics['critico_pct']:.1f}%)")
        print(f"   Riesgo Alto:    {self.statistics['alto']} ({self.statistics['alto_pct']:.1f}%)")
        print(f"   Riesgo Medio:   {self.statistics['medio']} ({self.statistics['medio_pct']:.1f}%)")
        print(f"   Riesgo Bajo:    {self.statistics['bajo']} ({self.statistics['bajo_pct']:.1f}%)")
        
        return self.alerts
    
    def _calculate_statistics(self) -> None:
        """Calcula estadísticas de las alertas generadas."""
        total = len(self.alerts)
        
        counts = {
            'critico': sum(1 for a in self.alerts if a.risk_level == RiskLevel.CRITICO),
            'alto': sum(1 for a in self.alerts if a.risk_level == RiskLevel.ALTO),
            'medio': sum(1 for a in self.alerts if a.risk_level == RiskLevel.MEDIO),
            'bajo': sum(1 for a in self.alerts if a.risk_level == RiskLevel.BAJO)
        }
        
        self.statistics = {
            **counts,
            'total': total,
            'critico_pct': counts['critico'] / total * 100 if total > 0 else 0,
            'alto_pct': counts['alto'] / total * 100 if total > 0 else 0,
            'medio_pct': counts['medio'] / total * 100 if total > 0 else 0,
            'bajo_pct': counts['bajo'] / total * 100 if total > 0 else 0,
            'need_immediate_action': counts['critico'] + counts['alto']
        }
    
    def get_high_risk_students(
        self, 
        min_risk_level: RiskLevel = RiskLevel.ALTO
    ) -> List[StudentAlert]:
        """
        Obtiene estudiantes con riesgo alto o crítico.
        
        Args:
            min_risk_level: Nivel mínimo de riesgo a incluir
            
        Returns:
            Lista de alertas filtradas
        """
        level_order = [RiskLevel.BAJO, RiskLevel.MEDIO, RiskLevel.ALTO, RiskLevel.CRITICO]
        min_idx = level_order.index(min_risk_level)
        
        return [
            a for a in self.alerts 
            if level_order.index(a.risk_level) >= min_idx
        ]
    
    def get_priority_list(self, top_n: Optional[int] = None) -> List[StudentAlert]:
        """
        Obtiene lista de estudiantes ordenada por prioridad.
        
        Args:
            top_n: Número de estudiantes a retornar (None = todos)
            
        Returns:
            Lista de alertas ordenada por prioridad
        """
        sorted_alerts = sorted(self.alerts, key=lambda x: (x.priority, -x.risk_score))
        
        if top_n:
            return sorted_alerts[:top_n]
        return sorted_alerts
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte las alertas a DataFrame para análisis.
        
        Returns:
            DataFrame con todas las alertas
        """
        data = []
        for alert in self.alerts:
            data.append({
                'estudiante_id': alert.student_id,
                'risk_score': round(alert.risk_score * 100, 1),  # Como porcentaje
                'risk_level': alert.risk_level.value,
                'priority': alert.priority,
                'num_risk_factors': len(alert.risk_factors),
                'risk_factors': ', '.join(alert.risk_factors),
                'recommendations': ' | '.join(alert.recommendations)
            })
        
        return pd.DataFrame(data)
    
    def generate_intervention_report(self) -> str:
        """
        Genera un reporte de intervención para el equipo de bienestar.
        
        Returns:
            String con reporte formateado
        """
        if not self.alerts:
            return "No hay alertas generadas."
        
        high_risk = self.get_high_risk_students(RiskLevel.ALTO)
        critical = [a for a in high_risk if a.risk_level == RiskLevel.CRITICO]
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         REPORTE DE INTERVENCIÓN - SISTEMA DE ALERTAS TEMPRANAS               ║
║                     Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 RESUMEN EJECUTIVO
────────────────────────────────────────────────────────────────────────────────
  Total estudiantes analizados:    {self.statistics['total']:>5}
  Requieren intervención urgente:  {self.statistics['need_immediate_action']:>5} ({(self.statistics['critico_pct'] + self.statistics['alto_pct']):.1f}%)
  
  Distribución por nivel de riesgo:
  ┌─────────────┬──────────┬────────────┐
  │ Nivel       │ Cantidad │ Porcentaje │
  ├─────────────┼──────────┼────────────┤
  │ 🔴 CRÍTICO  │ {self.statistics['critico']:>8} │ {self.statistics['critico_pct']:>9.1f}% │
  │ 🟠 ALTO     │ {self.statistics['alto']:>8} │ {self.statistics['alto_pct']:>9.1f}% │
  │ 🟡 MEDIO    │ {self.statistics['medio']:>8} │ {self.statistics['medio_pct']:>9.1f}% │
  │ 🟢 BAJO     │ {self.statistics['bajo']:>8} │ {self.statistics['bajo_pct']:>9.1f}% │
  └─────────────┴──────────┴────────────┘

🚨 CASOS CRÍTICOS (Requieren acción inmediata)
────────────────────────────────────────────────────────────────────────────────
"""
        
        # Agregar detalles de casos críticos
        for alert in critical[:10]:  # Top 10 críticos
            report += f"""
  Estudiante ID: {alert.student_id}
  ├── Score de Riesgo: {alert.risk_score*100:.1f}%
  ├── Factores de Riesgo:
"""
            for factor in alert.risk_factors[:3]:
                report += f"  │   • {factor}\n"
            report += f"  └── Acción prioritaria: {alert.recommendations[0] if alert.recommendations else 'N/A'}\n"
        
        if len(critical) > 10:
            report += f"\n  ... y {len(critical) - 10} casos críticos adicionales\n"
        
        # Agregar resumen de acciones recomendadas
        all_recommendations = []
        for alert in high_risk:
            all_recommendations.extend(alert.recommendations)
        
        # Contar frecuencia de recomendaciones
        from collections import Counter
        rec_counts = Counter(all_recommendations)
        
        report += f"""
📋 ACCIONES PRIORITARIAS RECOMENDADAS
────────────────────────────────────────────────────────────────────────────────
"""
        for rec, count in rec_counts.most_common(5):
            report += f"  • {rec} (aplicable a {count} estudiantes)\n"
        
        report += f"""
💡 RECOMENDACIONES GENERALES
────────────────────────────────────────────────────────────────────────────────
  1. Convocar reunión urgente del comité de permanencia estudiantil
  2. Asignar tutores para casos críticos (ratio 1:5 máximo)
  3. Programar contacto telefónico con estudiantes críticos en 48 horas
  4. Preparar recursos de apoyo académico y psicosocial
  5. Documentar intervenciones para seguimiento

═══════════════════════════════════════════════════════════════════════════════
  Este reporte fue generado automáticamente por el Sistema de Predicción
  de Deserción Estudiantil basado en Machine Learning.
═══════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def generate_student_report(self, student_id: int) -> str:
        """
        Genera un reporte individual para un estudiante.
        
        Args:
            student_id: ID del estudiante
            
        Returns:
            String con reporte individual
        """
        alert = next((a for a in self.alerts if a.student_id == student_id), None)
        
        if alert is None:
            return f"No se encontró alerta para el estudiante {student_id}"
        
        emoji = {
            RiskLevel.BAJO: "🟢",
            RiskLevel.MEDIO: "🟡",
            RiskLevel.ALTO: "🟠",
            RiskLevel.CRITICO: "🔴"
        }
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           REPORTE INDIVIDUAL DE RIESGO DE DESERCIÓN              ║
╚══════════════════════════════════════════════════════════════════╝

📋 INFORMACIÓN GENERAL
  Estudiante ID:     {alert.student_id}
  Fecha de análisis: {alert.created_at.strftime('%Y-%m-%d')}
  
📊 EVALUACIÓN DE RIESGO
  Score de Riesgo:   {alert.risk_score*100:.1f}%
  Nivel de Riesgo:   {emoji[alert.risk_level]} {alert.risk_level.value}
  Prioridad:         {alert.priority}/5

⚠️ FACTORES DE RIESGO IDENTIFICADOS
"""
        
        for i, factor in enumerate(alert.risk_factors, 1):
            report += f"  {i}. {factor}\n"
        
        if not alert.risk_factors:
            report += "  No se identificaron factores de riesgo significativos\n"
        
        report += f"""
💡 RECOMENDACIONES DE INTERVENCIÓN
"""
        
        for i, rec in enumerate(alert.recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        if not alert.recommendations:
            report += "  Continuar con seguimiento regular\n"
        
        report += """
═══════════════════════════════════════════════════════════════════
"""
        
        return report


# Funciones de conveniencia
def quick_alert_analysis(
    df: pd.DataFrame,
    risk_scores: np.ndarray
) -> Tuple[EarlyAlertSystem, pd.DataFrame]:
    """
    Análisis rápido de alertas.
    
    Args:
        df: DataFrame con datos de estudiantes
        risk_scores: Probabilidades de deserción
        
    Returns:
        Tupla (sistema de alertas, DataFrame de alertas)
    """
    system = EarlyAlertSystem()
    system.generate_alerts(df, risk_scores)
    return system, system.to_dataframe()


if __name__ == "__main__":
    # Demo del sistema de alertas
    print("=" * 60)
    print("DEMO: SISTEMA DE ALERTAS TEMPRANAS")
    print("=" * 60)
    
    # Crear datos sintéticos de ejemplo
    np.random.seed(42)
    n_students = 100
    
    df_demo = pd.DataFrame({
        'estudiante_id': range(1, n_students + 1),
        'promedio_notas': np.random.uniform(2.0, 4.5, n_students),
        'asistencia_porcentaje': np.random.uniform(50, 100, n_students),
        'ratio_creditos_aprobados': np.random.uniform(0.4, 1.0, n_students),
        'materias_perdidas': np.random.poisson(1, n_students),
        'horas_trabajo_semana': np.random.choice([0, 20, 40], n_students),
        'horas_plataforma_semana': np.random.exponential(5, n_students),
        'interacciones_tutorias': np.random.poisson(2, n_students),
        'distancia_campus_km': np.random.exponential(15, n_students)
    })
    
    # Simular scores de riesgo
    risk_scores = np.random.beta(2, 5, n_students)  # Distribución sesgada hacia bajo riesgo
    
    # Generar alertas
    system = EarlyAlertSystem()
    alerts = system.generate_alerts(df_demo, risk_scores)
    
    # Mostrar reporte de intervención
    print(system.generate_intervention_report())
    
    # Mostrar reporte individual de estudiante de alto riesgo
    high_risk = system.get_high_risk_students(RiskLevel.ALTO)
    if high_risk:
        print(system.generate_student_report(high_risk[0].student_id))
