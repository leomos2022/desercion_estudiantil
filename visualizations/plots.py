"""
Visualizaciones para el Sistema de Predicción de Deserción
==========================================================

Este módulo proporciona funciones de visualización para:
- Análisis exploratorio de datos (EDA)
- Evaluación de modelos
- Interpretación de resultados
- Dashboards de monitoreo

Todas las visualizaciones están diseñadas para ser
claras, informativas y adecuadas para presentaciones
académicas y reportes ejecutivos.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Any, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Configuración global de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Colores personalizados para el proyecto
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#3498DB',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    
    # Colores de riesgo
    'risk_low': '#27AE60',
    'risk_medium': '#F1C40F',
    'risk_high': '#E67E22',
    'risk_critical': '#C0392B'
}


def plot_data_distribution(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la distribución de variables del dataset.
    
    Args:
        df: DataFrame con los datos
        columns: Columnas a visualizar (None = numéricas)
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c not in ['estudiante_id', 'desercion']]
    
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Histograma con KDE
        sns.histplot(data=df, x=col, kde=True, ax=ax, color=COLORS['primary'])
        ax.set_title(f'Distribución: {col}', fontsize=10)
        ax.set_xlabel('')
        
        # Añadir estadísticas
        mean = df[col].mean()
        median = df[col].median()
        ax.axvline(mean, color='red', linestyle='--', alpha=0.7, label=f'Media: {mean:.2f}')
        ax.axvline(median, color='green', linestyle='--', alpha=0.7, label=f'Mediana: {median:.2f}')
        ax.legend(fontsize=8)
    
    # Ocultar ejes vacíos
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribución de Variables del Dataset', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la matriz de correlación entre variables.
    
    Args:
        df: DataFrame con los datos
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calcular correlación
    corr = numeric_df.corr()
    
    # Crear máscara para triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 8}
    )
    
    ax.set_title('Matriz de Correlación - Variables del Dataset', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'desercion',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la distribución de la variable objetivo.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfico de barras
    counts = df[target_col].value_counts()
    labels = ['No Deserta', 'Deserta']
    colors = [COLORS['success'], COLORS['danger']]
    
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Cantidad de Estudiantes')
    axes[0].set_title('Distribución de Clases')
    
    # Añadir valores en las barras
    for bar, count in zip(bars, counts.values):
        height = bar.get_height()
        axes[0].annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Gráfico circular
    explode = (0.05, 0.05)
    axes[1].pie(
        counts.values,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        textprops={'fontsize': 11}
    )
    axes[1].set_title('Proporción de Clases')
    
    # Añadir información de desbalance
    ratio = counts.max() / counts.min()
    fig.text(0.5, -0.02, f'Ratio de desbalance: {ratio:.2f}:1', 
             ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Análisis de la Variable Objetivo (Deserción)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la importancia de características.
    
    Args:
        importance_df: DataFrame con columnas 'feature' e 'importance'
        top_n: Número de features a mostrar
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    # Ordenar y seleccionar top N
    df = importance_df.nlargest(top_n, 'importance')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Barras horizontales
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    
    bars = ax.barh(df['feature'], df['importance'], color=colors, edgecolor='white')
    
    ax.set_xlabel('Importancia')
    ax.set_title(f'Top {top_n} Características más Importantes para Predicción', fontsize=14)
    ax.invert_yaxis()  # Más importante arriba
    
    # Añadir valores
    for bar, val in zip(bars, df['importance']):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'f1_score',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compara el rendimiento de múltiples modelos.
    
    Args:
        results_df: DataFrame con métricas por modelo
        metric: Métrica a visualizar
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ordenar por métrica
    df = results_df.sort_values(metric, ascending=True)
    
    # Colores según rendimiento
    norm = plt.Normalize(df[metric].min(), df[metric].max())
    colors = plt.cm.RdYlGn(norm(df[metric]))
    
    # Barras horizontales
    bars = ax.barh(df.index, df[metric], color=colors, edgecolor='white', linewidth=2)
    
    # Línea de referencia
    ax.axvline(x=df[metric].mean(), color='red', linestyle='--', 
               alpha=0.7, label=f'Promedio: {df[metric].mean():.3f}')
    
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparación de Modelos - {metric.replace("_", " ").title()}', fontsize=14)
    ax.legend()
    
    # Añadir valores
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_risk_distribution(
    alerts_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la distribución de riesgo de deserción.
    
    Args:
        alerts_df: DataFrame con alertas
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Distribución por nivel de riesgo
    risk_colors = {
        'BAJO': COLORS['risk_low'],
        'MEDIO': COLORS['risk_medium'],
        'ALTO': COLORS['risk_high'],
        'CRÍTICO': COLORS['risk_critical']
    }
    
    counts = alerts_df['risk_level'].value_counts()
    order = ['BAJO', 'MEDIO', 'ALTO', 'CRÍTICO']
    counts = counts.reindex(order, fill_value=0)
    
    bars = axes[0].bar(counts.index, counts.values, 
                       color=[risk_colors.get(r, 'gray') for r in counts.index],
                       edgecolor='white', linewidth=2)
    axes[0].set_title('Estudiantes por Nivel de Riesgo')
    axes[0].set_ylabel('Cantidad')
    
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', fontsize=11, fontweight='bold')
    
    # 2. Histograma de scores de riesgo
    axes[1].hist(alerts_df['risk_score'], bins=20, color=COLORS['primary'],
                 edgecolor='white', alpha=0.7)
    axes[1].axvline(alerts_df['risk_score'].mean(), color='red',
                    linestyle='--', label=f'Media: {alerts_df["risk_score"].mean():.1f}%')
    axes[1].axvline(50, color='orange', linestyle=':', label='Umbral 50%')
    axes[1].set_title('Distribución de Scores de Riesgo')
    axes[1].set_xlabel('Score de Riesgo (%)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()
    
    # 3. Distribución por prioridad
    priority_counts = alerts_df['priority'].value_counts().sort_index()
    priority_colors = [COLORS['risk_critical'], COLORS['risk_high'], 
                       COLORS['risk_medium'], COLORS['warning'], COLORS['risk_low']]
    
    axes[2].bar(priority_counts.index, priority_counts.values,
                color=priority_colors[:len(priority_counts)],
                edgecolor='white', linewidth=2)
    axes[2].set_title('Estudiantes por Prioridad de Atención')
    axes[2].set_xlabel('Prioridad (1=Urgente, 5=Baja)')
    axes[2].set_ylabel('Cantidad')
    
    plt.suptitle('Dashboard de Distribución de Riesgo', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_vs_target(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'desercion',
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la relación entre características y la variable objetivo.
    
    Args:
        df: DataFrame con los datos
        features: Lista de features a visualizar
        target: Nombre de la columna objetivo
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = [COLORS['success'], COLORS['danger']]
    labels = ['No Deserta', 'Deserta']
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Boxplot por clase
        df_plot = df[[feature, target]].dropna()
        
        for i, val in enumerate([0, 1]):
            data = df_plot[df_plot[target] == val][feature]
            bp = ax.boxplot([data], positions=[i], widths=0.6,
                           patch_artist=True)
            bp['boxes'][0].set_facecolor(colors[i])
            bp['boxes'][0].set_alpha(0.7)
        
        ax.set_xticklabels(labels)
        ax.set_title(f'{feature}', fontsize=10)
        ax.set_ylabel('Valor')
    
    # Ocultar ejes vacíos
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribución de Variables por Clase de Deserción', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ml_pipeline_diagram(
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera un diagrama del pipeline de Machine Learning.
    
    Esta visualización explica los componentes del sistema
    de forma pedagógica.
    
    Args:
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Definir componentes
    components = [
        {'name': 'ENTRADA\nDE DATOS', 'x': 1, 'y': 5, 'color': COLORS['info'],
         'items': ['Notas', 'Asistencia', 'Socioeconómicos', 'Comportamiento']},
        {'name': 'PREPROCESAMIENTO', 'x': 4, 'y': 5, 'color': COLORS['warning'],
         'items': ['Limpieza', 'Imputación', 'Normalización', 'SMOTE']},
        {'name': 'ENTRENAMIENTO\nMODELO', 'x': 7.5, 'y': 5, 'color': COLORS['secondary'],
         'items': ['Random Forest', 'XGBoost', 'Validación', 'Selección']},
        {'name': 'EVALUACIÓN', 'x': 11, 'y': 5, 'color': COLORS['primary'],
         'items': ['Accuracy', 'Precision', 'Recall', 'AUC-ROC']},
        {'name': 'SALIDAS\nY ALERTAS', 'x': 14.5, 'y': 5, 'color': COLORS['danger'],
         'items': ['Score Riesgo', 'Clasificación', 'Alertas', 'Intervención']}
    ]
    
    # Dibujar componentes
    for comp in components:
        # Caja principal
        rect = plt.Rectangle((comp['x']-1, comp['y']-1.5), 2, 3,
                             facecolor=comp['color'], alpha=0.3,
                             edgecolor=comp['color'], linewidth=2)
        ax.add_patch(rect)
        
        # Título
        ax.text(comp['x'], comp['y']+1, comp['name'],
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Items
        for i, item in enumerate(comp['items']):
            ax.text(comp['x'], comp['y']-0.3-i*0.5, f'• {item}',
                   ha='center', va='center', fontsize=8)
    
    # Flechas de conexión
    for i in range(len(components)-1):
        x1 = components[i]['x'] + 1
        x2 = components[i+1]['x'] - 1
        ax.annotate('', xy=(x2, 5), xytext=(x1, 5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                   lw=2, connectionstyle='arc3,rad=0'))
    
    # Título
    ax.text(8, 9, 'Pipeline de Machine Learning - Predicción de Deserción Estudiantil',
           ha='center', fontsize=14, fontweight='bold')
    
    # Subtítulo
    ax.text(8, 8.3, 'Componentes del Sistema de Aprendizaje Automático',
           ha='center', fontsize=11, style='italic')
    
    # Leyenda explicativa
    ax.text(8, 0.5, 
           'ENTRADA → Datos recolectados | PROCESO → Transformación y entrenamiento | SALIDA → Predicciones y acciones',
           ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_dashboard(
    df_original: pd.DataFrame,
    alerts_df: pd.DataFrame,
    metrics: Dict[str, float],
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Crea un dashboard resumen completo del sistema.
    
    Args:
        df_original: DataFrame original con datos
        alerts_df: DataFrame con alertas generadas
        metrics: Diccionario con métricas del modelo
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig = plt.figure(figsize=figsize)
    
    # Crear grid de subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Distribución de clases (pequeño)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'desercion' in df_original.columns:
        counts = df_original['desercion'].value_counts()
        colors = [COLORS['success'], COLORS['danger']]
        ax1.pie(counts.values, labels=['No Deserta', 'Deserta'],
                colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribución Original', fontsize=10)
    
    # 2. Métricas del modelo
    ax2 = fig.add_subplot(gs[0, 1])
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics.get('accuracy', 0), metrics.get('precision', 0),
                     metrics.get('recall', 0), metrics.get('f1_score', 0)]
    bars = ax2.barh(metric_names, metric_values, color=COLORS['primary'])
    ax2.set_xlim(0, 1)
    ax2.set_title('Métricas del Mejor Modelo', fontsize=10)
    for bar, val in zip(bars, metric_values):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    # 3. Distribución de riesgo
    ax3 = fig.add_subplot(gs[0, 2])
    if 'risk_level' in alerts_df.columns:
        risk_counts = alerts_df['risk_level'].value_counts()
        risk_colors = [COLORS['risk_low'], COLORS['risk_medium'],
                       COLORS['risk_high'], COLORS['risk_critical']]
        order = ['BAJO', 'MEDIO', 'ALTO', 'CRÍTICO']
        risk_counts = risk_counts.reindex(order, fill_value=0)
        ax3.pie(risk_counts.values, labels=order,
                colors=risk_colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Distribución por Nivel de Riesgo', fontsize=10)
    
    # 4. Histograma de scores de riesgo
    ax4 = fig.add_subplot(gs[1, :2])
    if 'risk_score' in alerts_df.columns:
        ax4.hist(alerts_df['risk_score'], bins=25, color=COLORS['primary'],
                 edgecolor='white', alpha=0.7)
        ax4.axvline(50, color='red', linestyle='--', label='Umbral 50%')
        ax4.axvline(alerts_df['risk_score'].mean(), color='green',
                    linestyle=':', label=f'Media: {alerts_df["risk_score"].mean():.1f}%')
        ax4.legend()
    ax4.set_xlabel('Score de Riesgo (%)')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Scores de Riesgo Predichos', fontsize=10)
    
    # 5. Resumen de números
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    total = len(alerts_df)
    high_risk = len(alerts_df[alerts_df['risk_level'].isin(['ALTO', 'CRÍTICO'])])
    
    text = f"""
    RESUMEN EJECUTIVO
    ─────────────────
    
    Total Estudiantes: {total}
    
    Alto Riesgo: {high_risk} ({high_risk/total*100:.1f}%)
    
    AUC-ROC: {metrics.get('auc_roc', 0):.3f}
    
    Requieren Atención
    Inmediata: {high_risk}
    """
    
    ax5.text(0.1, 0.9, text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Top factores de riesgo
    ax6 = fig.add_subplot(gs[2, :])
    if 'risk_factors' in alerts_df.columns:
        # Contar factores
        all_factors = []
        for factors in alerts_df['risk_factors']:
            if isinstance(factors, str):
                all_factors.extend(factors.split(', '))
        
        if all_factors:
            from collections import Counter
            factor_counts = Counter(all_factors).most_common(8)
            factors, counts = zip(*factor_counts)
            
            bars = ax6.barh(list(factors)[::-1], list(counts)[::-1],
                           color=COLORS['warning'])
            ax6.set_xlabel('Frecuencia')
            
            for bar, count in zip(bars, list(counts)[::-1]):
                ax6.text(count + 0.5, bar.get_y() + bar.get_height()/2,
                        str(count), va='center', fontsize=9)
    
    ax6.set_title('Factores de Riesgo más Frecuentes', fontsize=10)
    
    # Título principal
    fig.suptitle('Dashboard del Sistema de Predicción de Deserción Estudiantil',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Demo de visualizaciones
    print("=" * 60)
    print("DEMO: VISUALIZACIONES DEL SISTEMA")
    print("=" * 60)
    
    # Crear datos de ejemplo
    np.random.seed(42)
    n = 100
    
    df_demo = pd.DataFrame({
        'promedio_notas': np.random.normal(3.5, 0.7, n),
        'asistencia_porcentaje': np.random.uniform(60, 100, n),
        'materias_perdidas': np.random.poisson(1, n),
        'desercion': np.random.choice([0, 1], n, p=[0.75, 0.25])
    })
    
    # Generar gráficos
    print("\n📊 Generando gráficos de demostración...")
    
    fig1 = plot_target_distribution(df_demo)
    plt.show()
    
    fig2 = plot_ml_pipeline_diagram()
    plt.show()
    
    print("\n✅ Visualizaciones generadas correctamente")
