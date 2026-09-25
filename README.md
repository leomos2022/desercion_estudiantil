# 🎓 Sistema de Predicción de Deserción Estudiantil con Machine Learning

## 📚 Foro: Componentes de Machine Learning - UNIMINUTO

---

## 🎯 Pregunta Orientadora

### ¿Por qué es importante considerar la cantidad de datos en la aplicación de Machine Learning?

La cantidad de datos es fundamental en Machine Learning porque:

1. **Generalización del modelo**: Con más datos, el modelo aprende patrones más robustos y generaliza mejor a casos nuevos (Barrero Ortiz, 2020, p. 16).

2. **Reducción del sobreajuste (overfitting)**: Conjuntos de datos pequeños tienden a hacer que el modelo memorice en lugar de aprender patrones reales.

3. **Representatividad estadística**: Se necesitan suficientes ejemplos de cada clase para que el modelo aprenda a distinguirlas correctamente.

4. **Validación cruzada**: Permite dividir los datos en conjuntos de entrenamiento, validación y prueba sin perder representatividad.

5. **Ley de los grandes números**: Según Véliz Capuñay (2020, p. 45), los algoritmos de ML mejoran su precisión exponencialmente con más datos de calidad.

---

## 📋 Caso de Estudio: Sistema de Predicción de Deserción Estudiantil

### Contexto del Problema

| Componente | Descripción |
|------------|-------------|
| **Sector** | Educación superior |
| **Problema** | Alta tasa de deserción estudiantil (25-40% en primer año) |
| **Objetivo ML** | Predecir probabilidad de deserción por estudiante para intervención temprana |
| **Datos necesarios** | Notas, asistencia, datos socioeconómicos, uso de plataforma académica |

### Experiencia de Implementación

**Desafíos encontrados:**

1. **Datos dispersos**: Información en múltiples sistemas (académico, financiero, bienestar)
   - *Solución*: Integración mediante pipeline ETL en Python

2. **Desbalance de clases**: Solo 15-25% de estudiantes desertan
   - *Solución*: Técnicas SMOTE para oversampling

3. **Datos faltantes**: Estudiantes sin registro completo
   - *Solución*: Imputación con KNN y análisis de sensibilidad

4. **Privacidad**: Cumplimiento de Ley de Habeas Data
   - *Solución*: Anonimización y encriptación de datos personales

---

## 🔧 Componentes de Machine Learning Identificados

### 1. **Datos de Entrada (Input)**
```
├── Datos Académicos
│   ├── Notas parciales y finales
│   ├── Porcentaje de asistencia
│   └── Créditos aprobados vs matriculados
├── Datos Socioeconómicos
│   ├── Estrato socioeconómico
│   ├── Tipo de financiamiento
│   └── Situación laboral
├── Datos de Comportamiento
│   ├── Uso de plataforma virtual
│   ├── Interacción con tutorías
│   └── Participación en actividades
└── Datos Demográficos
    ├── Edad, género
    └── Distancia al campus
```

### 2. **Preprocesamiento de Datos**
- Limpieza de valores nulos y atípicos
- Normalización y estandarización
- Codificación de variables categóricas
- Selección de características (Feature Selection)
- Balanceo de clases con SMOTE

### 3. **Algoritmos de Entrenamiento**
- **Random Forest**: Robusto ante outliers, interpretable
- **XGBoost**: Alto rendimiento en datos tabulares
- **Regresión Logística**: Baseline interpretable
- **Redes Neuronales**: Para patrones complejos

### 4. **Validación del Modelo**
- Validación cruzada (K-Fold)
- Matriz de confusión
- Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### 5. **Salidas (Output)**
- Score de riesgo (0-100%)
- Clasificación: Alto/Medio/Bajo riesgo
- Alertas tempranas automáticas
- Recomendaciones de intervención

---

## 💡 Mejoras Propuestas para Aprovechar ML

1. **Implementación de MLOps**: Monitoreo continuo del modelo en producción

2. **Modelos explicables (XAI)**: SHAP values para interpretar predicciones

3. **Sistema de retroalimentación**: Loop de mejora continua con datos reales

4. **Integración con sistemas de alerta**: Notificaciones automáticas a tutores

5. **Dashboard interactivo**: Visualización para toma de decisiones

---

## 🚀 Cómo Ejecutar el Proyecto

### En Google Colab

1. Subir los archivos a Google Drive
2. Abrir `notebooks/01_main_pipeline.ipynb`
3. Ejecutar todas las celdas secuencialmente

### Estructura del Proyecto

```
desercion_estudiantil/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_main_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── alert_system.py
└── visualizations/
    └── plots.py
```

---

## 📖 Referencias Bibliográficas

- Barrero Ortiz, G. (2020). *Machine Learning: 50 Conceptos Clave para Entenderlo* (pp. 16, 56, 70). Paradigma.
- Rothman, D. (2018). *Artificial intelligence by example: Develop machine intelligence from scratch using real artificial intelligence use cases* (pp. 46-58). Packt Publishing.
- Véliz Capuñay, C. (2020). *Aprendizaje automático: Introducción al aprendizaje profundo* (pp. 33-105). Pontificia Universidad Católica del Perú.
- Campesato, O. (2020). *Artificial intelligence, machine learning, and deep learning* (pp. 18-19, 23-49). Mercury Learning And Information.

---

## 👥 Participación en el Foro

### Integrante 1 - Participación Principal
Desarrollar el caso de estudio completo con implementación del código.

### Integrante 2 y 3 - Retroalimentación
Analizar los resultados, proponer mejoras y validar las métricas del modelo.

### Integrante 4 - Conclusión
Sintetizar los aprendizajes y responder la pregunta orientadora con base en los resultados.

---

*Proyecto desarrollado para el Foro de Componentes de Machine Learning - 2025*
