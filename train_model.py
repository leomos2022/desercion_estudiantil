"""
Script de Entrenamiento del Modelo de Predicción de Deserción
Alerta Estudiantil Colombia
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    accuracy_score,
    f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def generar_datos_sinteticos(n_samples=5000, seed=42):
    """
    Genera datos sintéticos basados en patrones reales de deserción estudiantil.
    
    Los datos se generan siguiendo las estadísticas del SPADIES:
    - Deserción TyT: ~51%
    - Deserción Universitario: ~41%
    """
    np.random.seed(seed)
    
    # Features
    data = {
        # Académicos
        'promedio_academico': np.clip(np.random.normal(3.3, 0.7, n_samples), 0, 5),
        'asistencia': np.clip(np.random.normal(75, 15, n_samples), 0, 100),
        'ratio_creditos': np.clip(np.random.beta(5, 2, n_samples), 0, 1),
        
        # Socioeconómicos
        'estrato': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, 
                                     p=[0.15, 0.25, 0.30, 0.15, 0.10, 0.05]),
        'tiene_beca': np.random.binomial(1, 0.25, n_samples),
        'trabaja': np.random.binomial(1, 0.45, n_samples),
        
        # Demográficos
        'edad': np.clip(np.random.normal(22, 4, n_samples), 17, 50).astype(int),
        'semestre': np.random.choice(range(1, 11), n_samples, 
                                      p=[0.20, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05]),
        
        # Comportamiento
        'uso_plataforma': np.clip(np.random.exponential(5, n_samples), 0, 30),
        'distancia_campus': np.clip(np.random.exponential(12, n_samples), 0, 100),
        
        # Nivel de formación (0=TyT, 1=Universitario)
        'nivel_formacion': np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    }
    
    df = pd.DataFrame(data)
    
    # Generar variable objetivo (deserción) basada en factores de riesgo
    riesgo_base = np.zeros(n_samples)
    
    # Factores académicos (mayor peso)
    riesgo_base += (5 - df['promedio_academico']) * 0.15
    riesgo_base += (100 - df['asistencia']) * 0.005
    riesgo_base += (1 - df['ratio_creditos']) * 0.2
    
    # Factores socioeconómicos
    riesgo_base += (7 - df['estrato']) * 0.03
    riesgo_base -= df['tiene_beca'] * 0.15
    riesgo_base += df['trabaja'] * 0.1
    
    # Factores de comportamiento
    riesgo_base += np.maximum(0, (5 - df['uso_plataforma'])) * 0.02
    riesgo_base += df['distancia_campus'] * 0.002
    
    # Semestres críticos (1-3 tienen mayor deserción)
    riesgo_base += (df['semestre'] <= 3).astype(int) * 0.15
    
    # TyT tiene mayor deserción que Universitario
    riesgo_base += (df['nivel_formacion'] == 0).astype(int) * 0.1
    
    # Añadir ruido
    riesgo_base += np.random.normal(0, 0.1, n_samples)
    
    # Convertir a probabilidad
    prob_desercion = 1 / (1 + np.exp(-riesgo_base))
    
    # Ajustar para que coincida con tasas reales (~45% promedio)
    prob_desercion = np.clip(prob_desercion, 0.05, 0.95)
    
    # Generar etiquetas
    df['deserto'] = (np.random.random(n_samples) < prob_desercion).astype(int)
    
    print(f"Datos generados: {n_samples} registros")
    print(f"Tasa de deserción: {df['deserto'].mean()*100:.1f}%")
    print(f"  - TyT: {df[df['nivel_formacion']==0]['deserto'].mean()*100:.1f}%")
    print(f"  - Universitario: {df[df['nivel_formacion']==1]['deserto'].mean()*100:.1f}%")
    
    return df


def entrenar_modelo(df):
    """Entrena y evalúa múltiples modelos, selecciona el mejor"""
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO DEL MODELO")
    print("="*60)
    
    # Separar features y target
    feature_columns = [
        'promedio_academico', 'asistencia', 'ratio_creditos',
        'estrato', 'tiene_beca', 'trabaja', 'edad', 'semestre',
        'uso_plataforma', 'distancia_campus', 'nivel_formacion'
    ]
    
    X = df[feature_columns]
    y = df['deserto']
    
    # Split datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDatos de entrenamiento: {len(X_train)}")
    print(f"Datos de prueba: {len(X_test)}")
    print(f"Desbalance (train): {y_train.mean()*100:.1f}% desertores")
    
    # Aplicar SMOTE para balancear clases
    print("\nAplicando SMOTE para balancear clases...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Datos después de SMOTE: {len(X_train_balanced)}")
    print(f"Balance después de SMOTE: {y_train_balanced.mean()*100:.1f}% desertores")
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir modelos a evaluar
    modelos = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=5, random_state=42, 
            use_label_encoder=False, eval_metric='logloss'
        )
    }
    
    resultados = {}
    
    print("\n" + "-"*60)
    print("EVALUACIÓN DE MODELOS")
    print("-"*60)
    
    for nombre, modelo in modelos.items():
        print(f"\n>>> {nombre}")
        
        # Entrenar
        modelo.fit(X_train_scaled, y_train_balanced)
        
        # Predecir
        y_pred = modelo.predict(X_test_scaled)
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Validación cruzada
        cv_scores = cross_val_score(
            modelo, X_train_scaled, y_train_balanced, cv=5, scoring='f1'
        )
        
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC:  {auc:.4f}")
        print(f"  CV F1:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Seleccionar mejor modelo basado en AUC
    mejor_nombre = max(resultados, key=lambda x: resultados[x]['auc'])
    mejor_modelo = resultados[mejor_nombre]['modelo']
    
    print("\n" + "="*60)
    print(f"MEJOR MODELO: {mejor_nombre}")
    print(f"AUC-ROC: {resultados[mejor_nombre]['auc']:.4f}")
    print("="*60)
    
    # Reporte detallado del mejor modelo
    y_pred_final = mejor_modelo.predict(X_test_scaled)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred_final, 
                                target_names=['No Desertó', 'Desertó']))
    
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    # Feature importance (si está disponible)
    if hasattr(mejor_modelo, 'feature_importances_'):
        print("\nImportancia de Features:")
        importances = pd.DataFrame({
            'feature': feature_columns,
            'importance': mejor_modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importances.head(5).iterrows():
            print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    return mejor_modelo, scaler, feature_columns


def guardar_modelo(modelo, scaler, output_dir='models'):
    """Guarda el modelo y el scaler entrenados"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_path = output_path / 'desercion_model.joblib'
    scaler_path = output_path / 'scaler.joblib'
    
    joblib.dump(modelo, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModelo guardado en: {model_path}")
    print(f"Scaler guardado en: {scaler_path}")
    
    return model_path, scaler_path


def main():
    """Función principal para entrenar y guardar el modelo"""
    print("="*60)
    print("ALERTA ESTUDIANTIL COLOMBIA")
    print("Entrenamiento del Modelo de Predicción de Deserción")
    print("="*60)
    
    # 1. Generar datos sintéticos
    print("\n[1/3] Generando datos sintéticos...")
    df = generar_datos_sinteticos(n_samples=5000)
    
    # 2. Entrenar modelo
    print("\n[2/3] Entrenando modelos...")
    modelo, scaler, features = entrenar_modelo(df)
    
    # 3. Guardar modelo
    print("\n[3/3] Guardando modelo...")
    guardar_modelo(modelo, scaler)
    
    print("\n" + "="*60)
    print("¡Entrenamiento completado exitosamente!")
    print("="*60)
    print("\nPara iniciar la API:")
    print("  uvicorn api.main:app --reload")
    print("\nDocumentación disponible en:")
    print("  http://localhost:8000/docs")


if __name__ == "__main__":
    main()
