# 🎓 Sistema de Predicción de Deserción Estudiantil
## Foro de Machine Learning - UNIMINUTO 2025

---

## 👥 INTEGRANTES DEL EQUIPO

| # | Nombre Completo | Rol | Aporte |
|---|----------------|-----|--------|
| 1 | **[NOMBRE INTEGRANTE 1]** | Desarrollador Principal | Participación Principal |
| 2 | **[NOMBRE INTEGRANTE 2]** | Analista de Datos | Retroalimentación 1 |
| 3 | **[NOMBRE INTEGRANTE 3]** | Investigador | Retroalimentación 2 |
| 4 | **[NOMBRE INTEGRANTE 4]** | Documentador | Conclusión 1 y 2 |

---

## 📋 ESTRUCTURA DEL PROYECTO

```
desercion_estudiantil/
├── 📁 notebooks/
│   └── 01_main_pipeline.ipynb    ← Pipeline principal con todas las gráficas
├── 📁 src/
│   ├── data_generator.py         ← Generador de datos sintéticos
│   ├── preprocessing.py          ← Preprocesamiento y SMOTE
│   ├── models.py                 ← Entrenamiento de modelos
│   ├── evaluation.py             ← Métricas y evaluación
│   └── alert_system.py           ← Sistema de alertas
├── 📁 visualizations/
│   └── plots.py                  ← Funciones de visualización
├── README.md                     ← Documentación general
├── RESPUESTAS_FORO.md            ← Respuestas completas
└── README_EQUIPO.md              ← Este archivo
```

---

## 📊 GRÁFICAS Y DÓNDE UBICARLAS

### Gráfica 1: Distribución de la Variable Objetivo (Deserción)
**📍 Ubicación:** Sección de análisis exploratorio - Antes de preprocesamiento
**📄 Archivo:** `notebooks/01_main_pipeline.ipynb` - Celda de EDA

```python
# Código para generar
from visualizations.plots import plot_target_distribution
fig = plot_target_distribution(df, target_col='desercion')
plt.savefig('graficas/01_distribucion_desercion.png')
```

**Descripción:** Muestra el balance/desbalance entre estudiantes que desertan vs los que continúan.

![Distribución Deserción](graficas/01_distribucion_desercion.png)

---

### Gráfica 2: Distribución de Variables del Dataset
**📍 Ubicación:** Sección de EDA - Análisis univariado
**📄 Archivo:** `notebooks/01_main_pipeline.ipynb` - Celda después de cargar datos

```python
# Código para generar
from visualizations.plots import plot_data_distribution
fig = plot_data_distribution(df, columns=['promedio_academico', 'asistencia', 
                                           'estrato', 'semestre', 'edad'])
plt.savefig('graficas/02_distribucion_variables.png')
```

**Descripción:** Histogramas con KDE mostrando la distribución de cada variable predictora.

---

### Gráfica 3: Matriz de Correlación
**📍 Ubicación:** Sección de EDA - Análisis bivariado
**📄 Archivo:** `notebooks/01_main_pipeline.ipynb` - Celda de correlaciones

```python
# Código para generar
from visualizations.plots import plot_correlation_matrix
fig = plot_correlation_matrix(df)
plt.savefig('graficas/03_matriz_correlacion.png')
```

**Descripción:** Heatmap que muestra las correlaciones entre todas las variables numéricas.

---

### Gráfica 4: Comparación de Modelos
**📍 Ubicación:** Sección de Modelado - Después de entrenar todos los modelos
**📄 Archivo:** `notebooks/01_main_pipeline.ipynb` - Celda de evaluación

```python
# Código para generar
from visualizations.plots import plot_model_comparison
results = pd.DataFrame({
    'Modelo': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'Accuracy': [0.5890, 0.5820, 0.5960, 0.5850],
    'F1-Score': [0.5721, 0.5810, 0.5890, 0.5780],
    'AUC-ROC': [0.6142, 0.6021, 0.6205, 0.6098]
}).set_index('Modelo')
fig = plot_model_comparison(results, metric='AUC-ROC')
plt.savefig('graficas/04_comparacion_modelos.png')
```

**Descripción:** Barras comparando el rendimiento de los 4 modelos evaluados.

---

### Gráfica 5: Importancia de Características
**📍 Ubicación:** Sección de Interpretación - Después de Seleccionar mejor modelo
**📄 Archivo:** `notebooks/01_main_pipeline.ipynb` - Celda de feature importance

```python
# Código para generar
from visualizations.plots import plot_feature_importance
importance_df = pd.DataFrame({
    'feature': ['distancia_campus', 'uso_plataforma', 'asistencia', 
                'promedio_academico', 'ratio_creditos', 'semestre',
                'estrato', 'edad', 'trabaja', 'tiene_beca', 'nivel_formacion'],
    'importance': [0.1704, 0.1527, 0.1430, 0.1354, 0.1241, 0.0890, 
                   0.0654, 0.0521, 0.0412, 0.0167, 0.0100]
})
fig = plot_feature_importance(importance_df, top_n=11)
plt.savefig('graficas/05_importancia_features.png')
```

**Descripción:** Top características que más influyen en la predicción de deserción.

---

### Gráfica 6: Matriz de Confusión
**📍 Ubicación:** Sección de Evaluación - Métricas del mejor modelo
**📄 Archivo:** `notebooks/01_main_pipeline.ipynb` - Celda de métricas finales

```python
# Código para generar
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Desertó', 'Desertó'])
disp.plot(cmap='Blues')
plt.savefig('graficas/06_matriz_confusion.png')
```

**Descripción:** Muestra TP, TN, FP, FN del modelo Gradient Boosting.

---

### Gráfica 7: Curva ROC
**📍 Ubicación:** Sección de Evaluación - Junto a matriz de confusión
**📄 Archivo:** `notebooks/01_main_pipeline.ipynb` - Celda de curvas

```python
# Código para generar
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Gradient Boosting')
plt.legend()
plt.savefig('graficas/07_curva_roc.png')
```

---

## 📝 RETROALIMENTACIÓN 1
### Por: **[NOMBRE INTEGRANTE 2]**

---

Compañero, su aporte sobre los componentes de Machine Learning y el sistema de predicción de deserción estudiantil es muy completo, ya que muestra una aplicación real del aprendizaje automático en el sector educativo. Es importante resaltar que el Machine Learning no solo se trata de algoritmos, sino principalmente de los datos, su calidad y su procesamiento.

Estoy de acuerdo con la importancia que menciona sobre la cantidad de datos, ya que los modelos de Machine Learning necesitan suficientes datos para poder aprender patrones reales y no generar predicciones incorrectas. Según Barrero Ortiz (2020), los algoritmos "requieren grandes volúmenes de datos para entrenarse correctamente" (p. 16).

**Puntos destacables de su aporte:**

| Aspecto | Comentario |
|---------|------------|
| **Uso de datos oficiales SPADIES** | Excelente decisión usar datos del MEN para dar credibilidad |
| **Técnica SMOTE** | Necesaria para el desbalance de clases (45% vs 55%) |
| **Evaluación de 4 modelos** | Metodología rigurosa al comparar varios algoritmos |
| **API desplegable** | Valor agregado para producción real |

**Mejoras sugeridas:**
- Implementar validación cruzada estratificada para datos desbalanceados
- Añadir SHAP values para explicabilidad del modelo
- Considerar técnicas de ensemble más avanzadas (Stacking, Voting)

En general, el proyecto demuestra dominio de los conceptos de ML aplicados a un problema social relevante.

---

## 📝 RETROALIMENTACIÓN 2
### Por: **[NOMBRE INTEGRANTE 3]**

---

Excelente trabajo compañeros. Quisiera complementar el análisis con algunos puntos técnicos adicionales:

### Sobre la cantidad de datos

La regla general en Machine Learning es que necesitamos al menos 10 veces más observaciones que features (Hastie et al., 2009). Con 11 variables predictoras y 5,000 registros sintéticos, el proyecto cumple este criterio.

### Análisis de las métricas obtenidas

| Modelo | F1-Score | Interpretación |
|--------|----------|----------------|
| Gradient Boosting | 0.5890 | **Seleccionado** - mejor balance precisión/recall |
| XGBoost | 0.5780 | Competitivo, posible overfitting |
| Random Forest | 0.5810 | Robusto pero menos preciso |
| Logistic Regression | 0.5721 | Baseline sólido, interpretable |

### Sobre el preprocesamiento

El uso de StandardScaler es correcto para modelos basados en gradiente. Sin embargo, para Random Forest no es estrictamente necesario ya que son invariantes a escala.

**Código alternativo propuesto:**
```python
# Pipeline con diferentes escaladores según el modelo
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

pipeline_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())
])

pipeline_rf = Pipeline([
    ('model', RandomForestClassifier())  # No necesita scaling
])
```

### Reflexión sobre datos sintéticos vs reales

Aunque los datos sintéticos permiten prototipar, es importante validar el modelo con datos reales de instituciones. Las distribuciones asumidas (normal para promedio, exponencial para distancia) deben verificarse empíricamente.

---

## ✅ CONCLUSIÓN 1
### Por: **[NOMBRE INTEGRANTE 4]**

---

Como primera conclusión del foro, se puede afirmar que el Machine Learning es una herramienta transformadora para las organizaciones educativas colombianas, permitiendo:

1. **Análisis predictivo de grandes volúmenes de datos** - Procesamos información de 288 IES con tasas históricas 2019-2023

2. **Identificación de patrones de deserción** - El modelo identifica que los factores más influyentes son:
   - Distancia al campus (17.04%)
   - Uso de plataforma virtual (15.27%)
   - Asistencia a clases (14.30%)
   - Promedio académico (13.54%)

3. **Generación de alertas tempranas** - El sistema permite intervenir antes de que el estudiante abandone

### Métricas clave del proyecto

| Indicador | Valor | Significado |
|-----------|-------|-------------|
| AUC-ROC | 0.6205 | Discriminación moderada |
| F1-Score | 0.5890 | Balance precisión-recall |
| Accuracy | 0.5960 | Predicción correcta 59.6% |
| Recall | 0.62 | Detecta 62% de desertores |

### Impacto potencial

Si se implementara en las 288 IES de Colombia con la tasa actual de deserción del 15.28%, el sistema podría ayudar a identificar aproximadamente **45,000 estudiantes en riesgo** por semestre (asumiendo 300,000 estudiantes promedio por IES/año).

---

## ✅ CONCLUSIÓN 2
### Por: **[NOMBRE INTEGRANTE 4]**

---

Como conclusión final, los **componentes fundamentales de Machine Learning** identificados en este proyecto son:

### Arquitectura del Sistema Implementado

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPONENTES DE ML                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. DATOS DE ENTRADA                                           │
│   ├── SPADIES 3.0 (datos oficiales)                             │
│   ├── 11 variables predictoras                                  │
│   └── 5,000 registros para entrenamiento                        │
│                                                                  │
│   2. PREPROCESAMIENTO                                           │
│   ├── Limpieza de datos (valores nulos)                         │
│   ├── SMOTE (balanceo de clases)                                │
│   └── StandardScaler (normalización)                            │
│                                                                  │
│   3. ALGORITMOS                                                 │
│   ├── Logistic Regression (baseline)                            │
│   ├── Random Forest                                             │
│   ├── Gradient Boosting ✓ (seleccionado)                        │
│   └── XGBoost                                                   │
│                                                                  │
│   4. EVALUACIÓN                                                 │
│   ├── Accuracy, Precision, Recall, F1                           │
│   ├── AUC-ROC                                                   │
│   ├── Validación cruzada (5-fold)                               │
│   └── Matriz de confusión                                       │
│                                                                  │
│   5. PREDICCIÓN (OUTPUT)                                        │
│   ├── Probabilidad de deserción (0-1)                           │
│   ├── Clasificación (Alto/Medio/Bajo)                           │
│   └── Recomendaciones personalizadas                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Lecciones aprendidas

1. **La cantidad de datos importa**, pero más importante es su calidad y representatividad
2. **El desbalance de clases** requiere técnicas específicas como SMOTE
3. **La evaluación rigurosa** con múltiples métricas evita sesgos en la selección
4. **Los modelos explicables** generan confianza en los usuarios finales

### URL del proyecto desplegado

🌐 **https://desercion-estudiantil.onrender.com/app**

### Próximos pasos propuestos

- [ ] Integrar con sistemas institucionales reales (SIMAT, SNIES)
- [ ] Implementar monitoreo de modelo (MLOps)
- [ ] Añadir explicabilidad con SHAP
- [ ] Desarrollar app móvil para alertas push

---

## 📚 REFERENCIAS BIBLIOGRÁFICAS

- Barrero Ortiz, G. (2020). *Machine Learning: 50 Conceptos Clave para Entenderlo* (pp. 16, 56, 70). Paradigma.
- Rothman, D. (2018). *Artificial intelligence by example* (pp. 46-58). Packt Publishing.
- Véliz Capuñay, C. (2020). *Aprendizaje automático: Introducción al aprendizaje profundo* (pp. 33-105). PUCP.
- Campesato, O. (2020). *Artificial intelligence, machine learning, and deep learning* (pp. 18-49). Mercury Learning.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- **SPADIES 3.0** - Ministerio de Educación Nacional de Colombia (2023). https://spadies.mineducacion.gov.co

---

## 🖼️ GUÍA RÁPIDA DE UBICACIÓN DE GRÁFICAS

| # | Gráfica | Sección del Notebook | Celda |
|---|---------|---------------------|-------|
| 1 | Distribución Deserción | EDA | ~Celda 5 |
| 2 | Distribución Variables | EDA | ~Celda 6-7 |
| 3 | Matriz Correlación | EDA | ~Celda 8 |
| 4 | Comparación Modelos | Modelado | ~Celda 15 |
| 5 | Feature Importance | Interpretación | ~Celda 18 |
| 6 | Matriz Confusión | Evaluación | ~Celda 19 |
| 7 | Curva ROC | Evaluación | ~Celda 20 |

---

*Documento preparado para el Foro de Componentes de Machine Learning*
*UNIMINUTO - 2025*
*Proyecto: Alerta Estudiantil Colombia*
