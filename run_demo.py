#!/usr/bin/env python3
"""
Script de Ejecución Rápida - Sistema de Predicción de Deserción Estudiantil
============================================================================

Este script ejecuta el pipeline completo de ML para demostrar
todos los componentes del sistema.

Uso:
    python run_demo.py

Autor: Proyecto académico para Componentes de Machine Learning - UNIMINUTO
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
from src.data_generator import StudentDataGenerator
from src.preprocessing import DataPreprocessor
from src.models import DropoutPredictor
from src.evaluation import ModelEvaluator
from src.alert_system import EarlyAlertSystem

def main():
    """Ejecuta el pipeline completo de predicción de deserción."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║    SISTEMA DE PREDICCIÓN DE DESERCIÓN ESTUDIANTIL CON ML        ║
    ║                                                                  ║
    ║    Proyecto: Componentes de Machine Learning                     ║
    ║    Universidad: UNIMINUTO                                        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================
    # PASO 1: GENERACIÓN DE DATOS
    # =========================================
    print("\n" + "="*60)
    print("📊 PASO 1: GENERACIÓN DE DATOS DE ESTUDIANTES")
    print("="*60)
    
    generator = StudentDataGenerator(
        n_students=2000,
        dropout_rate=0.25,
        random_state=42
    )
    df = generator.generate()
    
    print(f"\n✅ Dataset generado:")
    print(f"   - Total estudiantes: {len(df)}")
    print(f"   - Variables: {df.shape[1]}")
    print(f"   - Tasa de deserción: {df['desercion'].mean()*100:.1f}%")
    
    print(f"\n📋 Variables del dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # =========================================
    # PASO 2: PREPROCESAMIENTO
    # =========================================
    print("\n" + "="*60)
    print("🔧 PASO 2: PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_method='mean',
        balance_method='smote'
    )
    
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        df, 
        target_column='desercion',
        test_size=0.2
    )
    
    # =========================================
    # PASO 3: ENTRENAMIENTO DE MODELOS
    # =========================================
    print("\n" + "="*60)
    print("🤖 PASO 3: ENTRENAMIENTO DE MODELOS")
    print("="*60)
    
    predictor = DropoutPredictor(random_state=42)
    
    # Entrenar modelos principales
    models_to_train = ['logistic_regression', 'random_forest', 'gradient_boosting']
    
    try:
        import xgboost
        models_to_train.append('xgboost')
    except ImportError:
        print("⚠️ XGBoost no disponible, continuando sin él...")
    
    trained_models = predictor.train_all_models(
        X_train, y_train,
        tune_hyperparameters=False,
        models_to_train=models_to_train
    )
    
    # Mostrar comparación
    print("\n📊 Comparación de Modelos:")
    print(predictor.get_model_comparison_df().to_string(index=False))
    
    # =========================================
    # PASO 4: EVALUACIÓN
    # =========================================
    print("\n" + "="*60)
    print("📈 PASO 4: EVALUACIÓN DEL MEJOR MODELO")
    print("="*60)
    
    evaluator = ModelEvaluator()
    
    best_model = predictor.best_model
    best_model_name = predictor.best_model_name
    
    metrics = evaluator.evaluate(
        best_model, X_test, y_test, 
        model_name=best_model_name
    )
    
    # Obtener predicciones para alertas
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    # Generar reporte
    print(evaluator.generate_evaluation_report(y_test, y_pred, y_proba, best_model_name))
    
    # =========================================
    # PASO 5: SISTEMA DE ALERTAS
    # =========================================
    print("\n" + "="*60)
    print("🚨 PASO 5: GENERACIÓN DE ALERTAS TEMPRANAS")
    print("="*60)
    
    # Crear DataFrame para alertas con datos originales
    df_test = df.iloc[:len(y_test)].copy()
    
    alert_system = EarlyAlertSystem()
    alerts = alert_system.generate_alerts(df_test, y_proba)
    
    # Generar reporte de intervención
    print(alert_system.generate_intervention_report())
    
    # Mostrar algunos estudiantes de alto riesgo
    high_risk = alert_system.get_high_risk_students()
    if high_risk:
        print("\n📋 EJEMPLO DE REPORTE INDIVIDUAL:")
        print(alert_system.generate_student_report(high_risk[0].student_id))
    
    # =========================================
    # PASO 6: IMPORTANCIA DE CARACTERÍSTICAS
    # =========================================
    print("\n" + "="*60)
    print("🔍 PASO 6: ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
    print("="*60)
    
    importance_df = predictor.get_feature_importance(
        feature_names=preprocessor.feature_names
    )
    
    if not importance_df.empty:
        print("\n📊 Top 10 características más importantes:")
        print(importance_df.head(10).to_string(index=False))
    
    # =========================================
    # RESUMEN FINAL
    # =========================================
    print("\n" + "="*60)
    print("✅ RESUMEN DEL PROYECTO")
    print("="*60)
    
    alerts_df = alert_system.to_dataframe()
    
    print(f"""
    📊 MÉTRICAS DEL MODELO ({best_model_name}):
       - Accuracy:  {metrics['accuracy']:.4f}
       - Precision: {metrics['precision']:.4f}
       - Recall:    {metrics['recall']:.4f}
       - F1-Score:  {metrics['f1_score']:.4f}
       - AUC-ROC:   {metrics.get('auc_roc', 0):.4f}
    
    🚨 ALERTAS GENERADAS:
       - Total estudiantes: {len(alerts_df)}
       - Riesgo Crítico: {len(alerts_df[alerts_df['risk_level']=='CRÍTICO'])}
       - Riesgo Alto: {len(alerts_df[alerts_df['risk_level']=='ALTO'])}
       - Riesgo Medio: {len(alerts_df[alerts_df['risk_level']=='MEDIO'])}
       - Riesgo Bajo: {len(alerts_df[alerts_df['risk_level']=='BAJO'])}
    
    💡 CONCLUSIÓN:
       El sistema puede identificar estudiantes en riesgo de deserción
       con un {metrics['recall']*100:.0f}% de recall, permitiendo
       intervención temprana antes de que abandonen sus estudios.
    """)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    ¡EJECUCIÓN COMPLETADA!                        ║
    ║                                                                  ║
    ║    Para el notebook interactivo, abrir:                          ║
    ║    notebooks/01_main_pipeline.ipynb en Google Colab              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    return {
        'dataframe': df,
        'metrics': metrics,
        'alerts': alerts_df,
        'model': best_model,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Ejecución interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
