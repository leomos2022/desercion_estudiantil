"""
Modelos de Machine Learning para Predicción de Deserción
========================================================

Este módulo implementa múltiples algoritmos de ML para predecir
la deserción estudiantil, incluyendo modelos interpretables y
de alto rendimiento.

Componentes de ML demostrados:
- Algoritmos de clasificación (Random Forest, XGBoost, Logistic Regression)
- Hiperparámetro tuning con GridSearchCV
- Validación cruzada
- Ensambles de modelos
- Interpretabilidad con SHAP

Referencias:
- Rothman, D. (2018). Artificial intelligence by example (pp. 46-58)
- Barrero Ortiz, G. (2020). Machine Learning: 50 Conceptos Clave (pp. 56, 70)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Modelos de sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Validación y métricas
from sklearn.model_selection import (
    cross_val_score, 
    GridSearchCV, 
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.calibration import CalibratedClassifierCV

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no instalado. Instalar con: pip install xgboost")

# SHAP para interpretabilidad
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class DropoutPredictor:
    """
    Predictor de deserción estudiantil con múltiples modelos.
    
    Esta clase proporciona una interfaz unificada para entrenar,
    comparar y utilizar diferentes algoritmos de ML para predecir
    la probabilidad de deserción de un estudiante.
    
    Modelos disponibles:
    - Logistic Regression: Modelo baseline, interpretable
    - Random Forest: Robusto, maneja no-linealidades
    - XGBoost: Alto rendimiento, gradient boosting
    - SVM: Márgenes máximos, kernel tricks
    - MLP: Red neuronal multicapa
    - Ensemble: Combinación de modelos
    
    Attributes:
        models (dict): Diccionario de modelos entrenados
        best_model: Mejor modelo según validación cruzada
        best_model_name: Nombre del mejor modelo
        feature_names: Nombres de las características
        
    Example:
        >>> predictor = DropoutPredictor()
        >>> predictor.train_all_models(X_train, y_train)
        >>> results = predictor.compare_models(X_test, y_test)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa el predictor.
        
        Args:
            random_state: Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.shap_values = None
        
        # Definir modelos base
        self._define_models()
    
    def _define_models(self) -> None:
        """Define los modelos disponibles con configuración base."""
        
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga']
                },
                'description': 'Modelo lineal interpretable - baseline'
            },
            
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'description': 'Ensemble de árboles - robusto y versátil'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state,
                    n_iter_no_change=10
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                'description': 'Gradient Boosting - sklearn nativo'
            },
            
            'svm': {
                'model': SVC(
                    random_state=self.random_state,
                    class_weight='balanced',
                    probability=True
                ),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                'description': 'Support Vector Machine - márgenes máximos'
            },
            
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'description': 'K-Nearest Neighbors - clasificación por proximidad'
            },
            
            'mlp': {
                'model': MLPClassifier(
                    random_state=self.random_state,
                    max_iter=500,
                    early_stopping=True
                ),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'description': 'Red Neuronal Multicapa - patrones complejos'
            },
            
            'adaboost': {
                'model': AdaBoostClassifier(
                    random_state=self.random_state,
                    algorithm='SAMME'
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'description': 'AdaBoost - enfoque en errores difíciles'
            }
        }
        
        # Agregar XGBoost si está disponible
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'description': 'XGBoost - alto rendimiento en datos tabulares'
            }
    
    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = False,
        cv: int = 5
    ) -> Any:
        """
        Entrena un modelo específico.
        
        Args:
            model_name: Nombre del modelo a entrenar
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            tune_hyperparameters: Si hacer búsqueda de hiperparámetros
            cv: Número de folds para validación cruzada
            
        Returns:
            Modelo entrenado
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Modelo '{model_name}' no disponible. "
                           f"Opciones: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_name]
        print(f"\n🔨 Entrenando {model_name}...")
        print(f"   📝 {config['description']}")
        
        if tune_hyperparameters:
            print(f"   🔍 Optimizando hiperparámetros (CV={cv})...")
            
            # Usar RandomizedSearchCV para eficiencia
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=20,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1,
                random_state=self.random_state
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            
            print(f"   ✅ Mejores parámetros: {search.best_params_}")
            print(f"   📊 Mejor F1 (CV): {search.best_score_:.4f}")
        else:
            model = config['model']
            model.fit(X_train, y_train)
        
        # Validación cruzada para evaluar
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        print(f"   📊 F1 Score (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Guardar modelo
        self.models[model_name] = model
        self.model_results[model_name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = False,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Entrena todos los modelos disponibles.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            tune_hyperparameters: Si optimizar hiperparámetros
            models_to_train: Lista de modelos a entrenar (None = todos)
            
        Returns:
            Diccionario con todos los modelos entrenados
        """
        print("=" * 60)
        print("🎯 ENTRENAMIENTO DE MODELOS")
        print("=" * 60)
        
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        for model_name in models_to_train:
            try:
                self.train_single_model(
                    model_name, X_train, y_train, 
                    tune_hyperparameters=tune_hyperparameters
                )
            except Exception as e:
                print(f"   ❌ Error entrenando {model_name}: {e}")
        
        # Identificar mejor modelo
        if self.model_results:
            best = max(self.model_results.items(), key=lambda x: x[1]['cv_mean'])
            self.best_model_name = best[0]
            self.best_model = self.models[self.best_model_name]
            
            print(f"\n🏆 MEJOR MODELO: {self.best_model_name}")
            print(f"   F1 Score: {best[1]['cv_mean']:.4f}")
        
        return self.models
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models_for_ensemble: Optional[List[str]] = None
    ) -> VotingClassifier:
        """
        Crea un ensemble de los mejores modelos.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            models_for_ensemble: Modelos a incluir (top 3 por defecto)
            
        Returns:
            VotingClassifier entrenado
        """
        print("\n🔗 Creando modelo ensemble...")
        
        if models_for_ensemble is None:
            # Usar top 3 modelos
            sorted_models = sorted(
                self.model_results.items(),
                key=lambda x: x[1]['cv_mean'],
                reverse=True
            )[:3]
            models_for_ensemble = [m[0] for m in sorted_models]
        
        # Crear lista de estimadores
        estimators = [
            (name, self.models[name]) 
            for name in models_for_ensemble 
            if name in self.models
        ]
        
        print(f"   Modelos incluidos: {[e[0] for e in estimators]}")
        
        # Crear ensemble con soft voting
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        
        # Evaluar ensemble
        cv_scores = cross_val_score(
            ensemble, X_train, y_train,
            cv=5, scoring='f1'
        )
        
        print(f"   📊 F1 Score Ensemble: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.models['ensemble'] = ensemble
        self.model_results['ensemble'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        return ensemble
    
    def predict(
        self,
        X: np.ndarray,
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X: Features para predicción
            model_name: Modelo a usar (None = mejor modelo)
            
        Returns:
            Array de predicciones (0 o 1)
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.models.get(model_name)
            
        if model is None:
            raise ValueError("No hay modelo entrenado disponible")
        
        return model.predict(X)
    
    def predict_proba(
        self,
        X: np.ndarray,
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Predice probabilidades de deserción.
        
        Args:
            X: Features para predicción
            model_name: Modelo a usar (None = mejor modelo)
            
        Returns:
            Array de probabilidades [prob_no_desercion, prob_desercion]
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError("No hay modelo entrenado disponible")
        
        return model.predict_proba(X)
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Obtiene la importancia de características.
        
        Args:
            feature_names: Nombres de las características
            model_name: Modelo del cual obtener importancias
            
        Returns:
            DataFrame con importancias ordenadas
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        # Extraer importancias según el tipo de modelo
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"⚠️ El modelo {model_name} no tiene importancias de características")
            return pd.DataFrame()
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def explain_with_shap(
        self,
        X_train: np.ndarray,
        X_explain: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Genera explicaciones SHAP para interpretabilidad.
        
        SHAP (SHapley Additive exPlanations) proporciona explicaciones
        locales y globales del modelo basadas en teoría de juegos.
        
        Args:
            X_train: Datos de entrenamiento (para background)
            X_explain: Datos a explicar
            feature_names: Nombres de características
            model_name: Modelo a explicar
            
        Returns:
            Objeto shap_values si SHAP está disponible
        """
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP no instalado. Instalar con: pip install shap")
            return None
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        
        print(f"\n🔍 Generando explicaciones SHAP para {model_name}...")
        
        # Crear explainer apropiado
        if model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
            explainer = shap.TreeExplainer(model)
        else:
            # Usar KernelExplainer para otros modelos
            background = shap.kmeans(X_train, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
        
        self.shap_values = explainer.shap_values(X_explain)
        
        return self.shap_values
    
    def get_model_comparison_df(self) -> pd.DataFrame:
        """
        Retorna un DataFrame comparativo de todos los modelos.
        
        Returns:
            DataFrame con métricas de cada modelo
        """
        if not self.model_results:
            return pd.DataFrame()
        
        data = []
        for name, results in self.model_results.items():
            data.append({
                'Modelo': name,
                'F1 Score (CV)': f"{results['cv_mean']:.4f}",
                'Desv. Est.': f"±{results['cv_std']:.4f}",
                'Min': f"{results['cv_scores'].min():.4f}",
                'Max': f"{results['cv_scores'].max():.4f}"
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('F1 Score (CV)', ascending=False)
        
        return df.reset_index(drop=True)


def quick_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'random_forest'
) -> Any:
    """
    Función de conveniencia para entrenamiento rápido.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Labels
        model_type: Tipo de modelo
        
    Returns:
        Modelo entrenado
    """
    predictor = DropoutPredictor()
    predictor.train_single_model(model_type, X_train, y_train)
    return predictor.models[model_type]


if __name__ == "__main__":
    # Demo de entrenamiento
    from sklearn.datasets import make_classification
    
    print("=" * 60)
    print("DEMO: ENTRENAMIENTO DE MODELOS")
    print("=" * 60)
    
    # Crear datos sintéticos
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        weights=[0.75, 0.25],
        random_state=42
    )
    
    # Entrenar modelos
    predictor = DropoutPredictor()
    predictor.train_all_models(X, y, tune_hyperparameters=False)
    
    # Mostrar comparación
    print("\n📊 COMPARACIÓN DE MODELOS:")
    print(predictor.get_model_comparison_df().to_string(index=False))
