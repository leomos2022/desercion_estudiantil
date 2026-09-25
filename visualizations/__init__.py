"""
Módulo de visualizaciones para el Sistema de Predicción de Deserción.
"""

from .plots import (
    plot_data_distribution,
    plot_correlation_matrix,
    plot_target_distribution,
    plot_feature_importance,
    plot_model_comparison,
    plot_risk_distribution,
    plot_feature_vs_target,
    plot_ml_pipeline_diagram,
    create_summary_dashboard,
    COLORS
)

__all__ = [
    'plot_data_distribution',
    'plot_correlation_matrix',
    'plot_target_distribution',
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_risk_distribution',
    'plot_feature_vs_target',
    'plot_ml_pipeline_diagram',
    'create_summary_dashboard',
    'COLORS'
]
