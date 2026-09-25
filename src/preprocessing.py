"""
Preprocesamiento de Datos para ML
=================================

Este módulo maneja todas las transformaciones necesarias para preparar
los datos de estudiantes para el entrenamiento del modelo.

Componentes de ML demostrados:
- Limpieza de datos (handling missing values)
- Codificación de variables categóricas
- Normalización/Estandarización
- Selección de características
- Balanceo de clases (SMOTE)
- División train/test
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importar SMOTE para balanceo de clases
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("⚠️ Advertencia: imbalanced-learn no instalado. Instalar con: pip install imbalanced-learn")


class DataPreprocessor:
    """
    Clase para preprocesamiento completo de datos de deserción estudiantil.
    
    Esta clase implementa un pipeline de preprocesamiento que incluye:
    1. Limpieza de valores nulos
    2. Codificación de variables categóricas
    3. Normalización de variables numéricas
    4. Balanceo de clases desbalanceadas
    5. Selección de características más relevantes
    
    Attributes:
        scaler: Escalador para normalización
        label_encoders: Diccionario de encoders para variables categóricas
        imputer: Imputador para valores faltantes
        feature_selector: Selector de características
        is_fitted: Indica si el preprocesador ha sido ajustado
    
    Example:
        >>> preprocessor = DataPreprocessor()
        >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, 'desercion')
    """
    
    def __init__(
        self,
        scaling_method: str = 'standard',
        imputation_method: str = 'knn',
        balance_method: str = 'smote',
        n_features_to_select: Optional[int] = None
    ):
        """
        Inicializa el preprocesador.
        
        Args:
            scaling_method: 'standard' (Z-score) o 'minmax' (0-1)
            imputation_method: 'mean', 'median', 'knn' o 'most_frequent'
            balance_method: 'smote', 'adasyn', 'undersample', 'smote_tomek' o None
            n_features_to_select: Número de features a seleccionar (None = todas)
        """
        self.scaling_method = scaling_method
        self.imputation_method = imputation_method
        self.balance_method = balance_method
        self.n_features_to_select = n_features_to_select
        
        # Componentes del pipeline
        self.scaler = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.imputer = None
        self.feature_selector = None
        
        # Metadatos
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_names = []
        self.is_fitted = False
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str = 'desercion',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ajusta el preprocesador y transforma los datos.
        
        Args:
            df: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            test_size: Proporción para conjunto de prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        print("=" * 60)
        print("🔄 PREPROCESAMIENTO DE DATOS")
        print("=" * 60)
        
        # Separar features y target
        df_clean = df.copy()
        
        # Eliminar columna ID si existe
        if 'estudiante_id' in df_clean.columns:
            df_clean = df_clean.drop('estudiante_id', axis=1)
        
        y = df_clean[target_column].values
        X = df_clean.drop(target_column, axis=1)
        
        # Identificar tipos de columnas
        self._identify_column_types(X)
        
        print(f"\n📊 Forma original: {X.shape}")
        print(f"📊 Columnas numéricas: {len(self.numeric_columns)}")
        print(f"📊 Columnas categóricas: {len(self.categorical_columns)}")
        print(f"📊 Distribución de clases: {np.bincount(y)}")
        
        # 1. Imputación de valores faltantes
        X = self._handle_missing_values(X)
        
        # 2. Codificación de variables categóricas
        X = self._encode_categorical(X)
        
        # Guardar nombres de features después de encoding
        self.feature_names = list(X.columns)
        
        # 3. Convertir a numpy array
        X = X.values
        
        # 4. División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✂️ División train/test: {1-test_size:.0%}/{test_size:.0%}")
        print(f"   Train: {X_train.shape[0]} muestras")
        print(f"   Test: {X_test.shape[0]} muestras")
        
        # 5. Normalización (solo en train, aplicar a test)
        X_train, X_test = self._normalize_data(X_train, X_test)
        
        # 6. Balanceo de clases (solo en train)
        X_train, y_train = self._balance_classes(X_train, y_train)
        
        # 7. Selección de características (opcional)
        if self.n_features_to_select:
            X_train, X_test = self._select_features(X_train, X_test, y_train)
        
        self.is_fitted = True
        
        print(f"\n✅ Preprocesamiento completado")
        print(f"   X_train final: {X_train.shape}")
        print(f"   X_test final: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforma nuevos datos usando los parámetros ajustados.
        
        Args:
            X: DataFrame con nuevos datos
            
        Returns:
            Array numpy con datos transformados
        """
        if not self.is_fitted:
            raise ValueError("El preprocesador no ha sido ajustado. Usar fit_transform primero.")
        
        X_copy = X.copy()
        
        # Eliminar ID si existe
        if 'estudiante_id' in X_copy.columns:
            X_copy = X_copy.drop('estudiante_id', axis=1)
        
        # Eliminar target si existe
        if 'desercion' in X_copy.columns:
            X_copy = X_copy.drop('desercion', axis=1)
        
        # Aplicar transformaciones
        X_copy = self._handle_missing_values(X_copy, fit=False)
        X_copy = self._encode_categorical(X_copy, fit=False)
        X_transformed = X_copy.values
        
        # Normalizar
        X_transformed = self.scaler.transform(X_transformed)
        
        return X_transformed
    
    def _identify_column_types(self, X: pd.DataFrame) -> None:
        """Identifica columnas numéricas y categóricas."""
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _handle_missing_values(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Maneja valores faltantes usando el método especificado.
        
        Implementa tres estrategias según Véliz Capuñay (2020):
        - Media/Mediana para numéricas
        - Moda para categóricas
        - KNN para imputación más sofisticada
        """
        X_copy = X.copy()
        missing_count = X_copy.isnull().sum().sum()
        
        if missing_count > 0:
            print(f"\n🔧 Imputando {missing_count} valores faltantes...")
            
            if fit:
                if self.imputation_method == 'knn':
                    # KNN solo para numéricas, moda para categóricas
                    self.imputer = KNNImputer(n_neighbors=5)
                    X_copy[self.numeric_columns] = self.imputer.fit_transform(
                        X_copy[self.numeric_columns]
                    )
                elif self.imputation_method == 'mean':
                    self.imputer = SimpleImputer(strategy='mean')
                    X_copy[self.numeric_columns] = self.imputer.fit_transform(
                        X_copy[self.numeric_columns]
                    )
                elif self.imputation_method == 'median':
                    self.imputer = SimpleImputer(strategy='median')
                    X_copy[self.numeric_columns] = self.imputer.fit_transform(
                        X_copy[self.numeric_columns]
                    )
                
                # Imputar categóricas con moda
                for col in self.categorical_columns:
                    if X_copy[col].isnull().any():
                        mode_value = X_copy[col].mode()[0]
                        X_copy[col].fillna(mode_value, inplace=True)
            else:
                if self.imputer is not None:
                    X_copy[self.numeric_columns] = self.imputer.transform(
                        X_copy[self.numeric_columns]
                    )
        else:
            print("\n✓ No hay valores faltantes")
        
        return X_copy
    
    def _encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Codifica variables categóricas.
        
        Usa One-Hot Encoding para variables nominales y Label Encoding
        para variables ordinales si es aplicable.
        """
        X_copy = X.copy()
        
        if len(self.categorical_columns) == 0:
            return X_copy
        
        print(f"\n🔄 Codificando {len(self.categorical_columns)} variables categóricas...")
        
        # Usar One-Hot Encoding para todas las categóricas
        for col in self.categorical_columns:
            if fit:
                # Crear dummies
                dummies = pd.get_dummies(X_copy[col], prefix=col, drop_first=True)
                self.label_encoders[col] = dummies.columns.tolist()
            else:
                # Usar los mismos dummies que en training
                dummies = pd.get_dummies(X_copy[col], prefix=col, drop_first=True)
                # Asegurar mismas columnas
                for expected_col in self.label_encoders[col]:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0
                dummies = dummies[self.label_encoders[col]]
            
            # Agregar dummies y eliminar original
            X_copy = pd.concat([X_copy, dummies], axis=1)
            X_copy = X_copy.drop(col, axis=1)
        
        return X_copy
    
    def _normalize_data(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normaliza los datos usando el método especificado.
        
        StandardScaler: Z-score normalization (media=0, std=1)
        MinMaxScaler: Escala a rango [0, 1]
        """
        print(f"\n📏 Normalizando datos ({self.scaling_method})...")
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test
    
    def _balance_classes(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balancea clases desbalanceadas usando técnicas de resampling.
        
        SMOTE (Synthetic Minority Over-sampling Technique):
        Crea ejemplos sintéticos de la clase minoritaria interpolando
        entre ejemplos existentes (Barrero Ortiz, 2020, p. 70).
        """
        if self.balance_method is None:
            return X_train, y_train
        
        if not IMBALANCED_AVAILABLE:
            print("⚠️ imbalanced-learn no disponible. Saltando balanceo.")
            return X_train, y_train
        
        print(f"\n⚖️ Balanceando clases ({self.balance_method})...")
        print(f"   Antes: {np.bincount(y_train)}")
        
        if self.balance_method == 'smote':
            resampler = SMOTE(random_state=42)
        elif self.balance_method == 'adasyn':
            resampler = ADASYN(random_state=42)
        elif self.balance_method == 'undersample':
            resampler = RandomUnderSampler(random_state=42)
        elif self.balance_method == 'smote_tomek':
            resampler = SMOTETomek(random_state=42)
        else:
            return X_train, y_train
        
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        print(f"   Después: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def _select_features(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selecciona las características más relevantes.
        
        Usa SelectKBest con información mutua o ANOVA F-value.
        """
        print(f"\n🎯 Seleccionando {self.n_features_to_select} mejores características...")
        
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif,
            k=min(self.n_features_to_select, X_train.shape[1])
        )
        
        X_train = self.feature_selector.fit_transform(X_train, y_train)
        X_test = self.feature_selector.transform(X_test)
        
        # Actualizar nombres de features seleccionados
        if len(self.feature_names) > 0:
            mask = self.feature_selector.get_support()
            self.feature_names = [f for f, m in zip(self.feature_names, mask) if m]
        
        return X_train, X_test
    
    def get_feature_importance_from_preprocessing(self) -> pd.DataFrame:
        """
        Retorna la importancia de características basada en selección.
        
        Returns:
            DataFrame con scores de importancia por característica
        """
        if self.feature_selector is None:
            return pd.DataFrame()
        
        scores = self.feature_selector.scores_
        
        return pd.DataFrame({
            'feature': self.feature_names[:len(scores)] if self.feature_names else range(len(scores)),
            'score': scores
        }).sort_values('score', ascending=False)


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Crea un pipeline de sklearn para preprocesamiento.
    
    Esta función crea un ColumnTransformer que puede ser usado
    directamente en un Pipeline de sklearn.
    
    Args:
        numeric_features: Lista de nombres de columnas numéricas
        categorical_features: Lista de nombres de columnas categóricas
        
    Returns:
        ColumnTransformer configurado
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


if __name__ == "__main__":
    # Ejemplo de uso
    from data_generator import StudentDataGenerator
    
    print("=" * 60)
    print("DEMO: PREPROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    # Generar datos de ejemplo
    generator = StudentDataGenerator(n_students=1000, dropout_rate=0.25)
    df = generator.generate()
    
    # Introducir algunos valores faltantes para demostración
    mask = np.random.random(df.shape[0]) < 0.05
    df.loc[mask, 'promedio_notas'] = np.nan
    
    # Preprocesar
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_method='knn',
        balance_method='smote'
    )
    
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, 'desercion')
    
    print(f"\n📊 Resultados finales:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train distribution: {np.bincount(y_train)}")
    print(f"   y_test distribution: {np.bincount(y_test)}")
