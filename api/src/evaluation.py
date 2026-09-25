"""
Evaluación y Métricas de Modelos ML
===================================

Este módulo proporciona herramientas completas para evaluar
modelos de predicción de deserción estudiantil.

Componentes de ML demostrados:
- Métricas de clasificación (Accuracy, Precision, Recall, F1, AUC-ROC)
- Matriz de confusión
- Curvas ROC y Precision-Recall
- Análisis de umbrales
- Validación cruzada estratificada

Referencias:
- Véliz Capuñay, C. (2020). Aprendizaje automático (pp. 85-105)
- Campesato, O. (2020). AI, ML, and Deep Learning (pp. 40-49)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Métricas de sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Evaluador completo de modelos de Machine Learning.
    
    Proporciona métricas estándar y visualizaciones para evaluar
    el rendimiento de modelos de clasificación binaria.
    
    Attributes:
        metrics_history (dict): Historial de métricas por modelo
        
    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate(model, X_test, y_test, 'RandomForest')
        >>> evaluator.plot_confusion_matrix(y_test, predictions)
    """
    
    def __init__(self):
        """Inicializa el evaluador."""
        self.metrics_history = {}
        self.all_predictions = {}
        self.all_probabilities = {}
        
    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = 'Model',
        threshold: float = 0.5,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evalúa un modelo con múltiples métricas.
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Labels de prueba
            model_name: Nombre del modelo para registro
            threshold: Umbral de clasificación
            verbose: Si mostrar resultados
            
        Returns:
            Diccionario con todas las métricas
        """
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Probabilidades (si están disponibles)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            has_proba = True
        except AttributeError:
            y_proba = y_pred.astype(float)
            has_proba = False
        
        # Aplicar umbral personalizado si es diferente a 0.5
        if threshold != 0.5 and has_proba:
            y_pred = (y_proba >= threshold).astype(int)
        
        # Calcular métricas
        metrics = {
            # Métricas básicas
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_test, y_pred),
            
            # Métricas avanzadas
            'cohen_kappa': cohen_kappa_score(y_test, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        }
        
        if has_proba:
            metrics.update({
                'auc_roc': roc_auc_score(y_test, y_proba),
                'auc_pr': average_precision_score(y_test, y_proba),
                'log_loss': log_loss(y_test, y_proba),
                'brier_score': brier_score_loss(y_test, y_proba)
            })
        
        # Guardar historial
        self.metrics_history[model_name] = metrics
        self.all_predictions[model_name] = y_pred
        self.all_probabilities[model_name] = y_proba
        
        if verbose:
            self._print_metrics(model_name, metrics)
        
        return metrics
    
    def _calculate_specificity(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Calcula la especificidad (True Negative Rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _print_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Imprime las métricas de forma formateada."""
        print(f"\n{'=' * 60}")
        print(f"📊 EVALUACIÓN: {model_name}")
        print(f"{'=' * 60}")
        
        print(f"\n🎯 MÉTRICAS DE CLASIFICACIÓN")
        print(f"   Accuracy:    {metrics['accuracy']:.4f}  (% predicciones correctas)")
        print(f"   Precision:   {metrics['precision']:.4f}  (% de predichos positivos correctos)")
        print(f"   Recall:      {metrics['recall']:.4f}  (% de positivos reales detectados)")
        print(f"   F1-Score:    {metrics['f1_score']:.4f}  (media armónica precision-recall)")
        print(f"   Specificity: {metrics['specificity']:.4f}  (% de negativos correctamente identificados)")
        
        if 'auc_roc' in metrics:
            print(f"\n📈 MÉTRICAS DE PROBABILIDAD")
            print(f"   AUC-ROC:     {metrics['auc_roc']:.4f}  (área bajo curva ROC)")
            print(f"   AUC-PR:      {metrics['auc_pr']:.4f}  (área bajo curva Precision-Recall)")
            print(f"   Log Loss:    {metrics['log_loss']:.4f}  (pérdida logarítmica)")
            print(f"   Brier Score: {metrics['brier_score']:.4f}  (calibración de probabilidades)")
        
        print(f"\n📏 MÉTRICAS AVANZADAS")
        print(f"   Cohen Kappa: {metrics['cohen_kappa']:.4f}  (concordancia ajustada por azar)")
        print(f"   MCC:         {metrics['matthews_corrcoef']:.4f}  (coeficiente de Matthews)")
        
        # Interpretación
        print(f"\n💡 INTERPRETACIÓN:")
        if metrics['f1_score'] >= 0.8:
            print("   ✅ Excelente rendimiento del modelo")
        elif metrics['f1_score'] >= 0.6:
            print("   ✓ Buen rendimiento, pero hay margen de mejora")
        else:
            print("   ⚠️ Rendimiento moderado, considerar mejoras")
        
        if metrics.get('auc_roc', 0) >= 0.8:
            print("   ✅ Excelente discriminación entre clases")
        
        if metrics['recall'] < 0.7:
            print("   ⚠️ Recall bajo: muchos desertores no están siendo detectados")
    
    def evaluate_multiple_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evalúa múltiples modelos y crea tabla comparativa.
        
        Args:
            models: Diccionario {nombre: modelo}
            X_test: Features de prueba
            y_test: Labels de prueba
            
        Returns:
            DataFrame con comparación de métricas
        """
        for name, model in models.items():
            self.evaluate(model, X_test, y_test, name, verbose=False)
        
        return self.get_comparison_table()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Genera tabla comparativa de todos los modelos evaluados.
        
        Returns:
            DataFrame con métricas de cada modelo
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics_history).T
        df = df.round(4)
        df = df.sort_values('f1_score', ascending=False)
        
        return df
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Genera reporte de clasificación detallado.
        
        Args:
            y_true: Labels reales
            y_pred: Predicciones
            target_names: Nombres de las clases
            
        Returns:
            String con reporte formateado
        """
        if target_names is None:
            target_names = ['No Deserta', 'Deserta']
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1',
        verbose: bool = True
    ) -> Tuple[float, float]:
        """
        Encuentra el umbral óptimo para maximizar una métrica.
        
        En el contexto de deserción estudiantil, puede ser más importante
        maximizar el recall (detectar todos los estudiantes en riesgo) aunque
        sacrifiquemos algo de precision.
        
        Args:
            y_true: Labels reales
            y_proba: Probabilidades predichas
            metric: Métrica a optimizar ('f1', 'recall', 'precision', 'balanced')
            verbose: Si mostrar resultados
            
        Returns:
            Tupla (umbral_optimo, valor_metrica)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        scores_by_threshold = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced':
                # Balance entre precision y recall con preferencia por recall
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                score = 0.3 * prec + 0.7 * rec  # Peso mayor a recall
            else:
                raise ValueError(f"Métrica '{metric}' no soportada")
            
            scores_by_threshold.append({'threshold': thresh, 'score': score})
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        if verbose:
            print(f"\n🎯 OPTIMIZACIÓN DE UMBRAL (métrica: {metric})")
            print(f"   Umbral óptimo: {best_threshold:.2f}")
            print(f"   Mejor {metric}: {best_score:.4f}")
            
            # Comparar con umbral default
            y_pred_default = (y_proba >= 0.5).astype(int)
            if metric == 'f1':
                default_score = f1_score(y_true, y_pred_default, zero_division=0)
            elif metric == 'recall':
                default_score = recall_score(y_true, y_pred_default, zero_division=0)
            else:
                default_score = f1_score(y_true, y_pred_default, zero_division=0)
            
            improvement = (best_score - default_score) / default_score * 100 if default_score > 0 else 0
            print(f"   Mejora vs umbral 0.5: {improvement:+.1f}%")
        
        return best_threshold, best_score
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Matriz de Confusión',
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza la matriz de confusión.
        
        La matriz de confusión muestra:
        - TN (True Negative): Estudiantes que no desertaron y fueron predichos correctamente
        - FP (False Positive): Estudiantes que no desertaron pero fueron predichos como desertores
        - FN (False Negative): Estudiantes que desertaron pero no fueron detectados (IMPORTANTE)
        - TP (True Positive): Estudiantes que desertaron y fueron detectados correctamente
        
        Args:
            y_true: Labels reales
            y_pred: Predicciones
            title: Título del gráfico
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la imagen
            
        Returns:
            Figura de matplotlib
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No Deserta', 'Deserta'],
            yticklabels=['No Deserta', 'Deserta'],
            ax=ax
        )
        
        ax.set_ylabel('Valor Real')
        ax.set_xlabel('Predicción')
        ax.set_title(title)
        
        # Añadir interpretación
        tn, fp, fn, tp = cm.ravel()
        text = (
            f'\n TN={tn} (correcto) | FP={fp} (falsa alarma)\n'
            f' FN={fn} (no detectado) | TP={tp} (detectado)'
        )
        plt.figtext(0.5, 0.01, text, ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'Curva ROC',
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza la curva ROC (Receiver Operating Characteristic).
        
        La curva ROC muestra la relación entre la tasa de verdaderos positivos
        (sensibilidad/recall) y la tasa de falsos positivos (1-especificidad).
        
        Args:
            y_true: Labels reales
            y_proba: Probabilidades predichas
            title: Título del gráfico
            figsize: Tamaño de la figura
            save_path: Ruta para guardar
            
        Returns:
            Figura de matplotlib
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', label='Aleatorio (AUC = 0.5)')
        
        ax.fill_between(fpr, tpr, alpha=0.3)
        
        ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (Recall)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Añadir punto óptimo (J de Youden)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                   c='red', s=100, marker='*', 
                   label=f'Punto óptimo (umbral={thresholds[optimal_idx]:.2f})')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'Curva Precision-Recall',
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza la curva Precision-Recall.
        
        Esta curva es especialmente útil para datasets desbalanceados
        como el de deserción estudiantil.
        
        Args:
            y_true: Labels reales
            y_proba: Probabilidades predichas
            title: Título del gráfico
            figsize: Tamaño de la figura
            save_path: Ruta para guardar
            
        Returns:
            Figura de matplotlib
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap:.3f})')
        
        # Línea base (proporción de positivos)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='r', linestyle='--', 
                   label=f'Baseline ({baseline:.3f})')
        
        ax.fill_between(recall, precision, alpha=0.3)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_all_evaluation_curves(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'Modelo',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Genera un dashboard con todas las curvas de evaluación.
        
        Args:
            y_true: Labels reales
            y_pred: Predicciones
            y_proba: Probabilidades
            model_name: Nombre del modelo
            save_path: Ruta para guardar
            
        Returns:
            Figura con subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Dashboard de Evaluación - {model_name}', fontsize=14, y=1.02)
        
        # 1. Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Deserta', 'Deserta'],
                   yticklabels=['No Deserta', 'Deserta'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Matriz de Confusión')
        axes[0, 0].set_ylabel('Valor Real')
        axes[0, 0].set_xlabel('Predicción')
        
        # 2. Curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 1].fill_between(fpr, tpr, alpha=0.3)
        axes[0, 1].set_xlabel('FPR')
        axes[0, 1].set_ylabel('TPR')
        axes[0, 1].set_title('Curva ROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Curva PR
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        axes[1, 0].plot(recall, precision, 'b-', linewidth=2, label=f'AP = {ap:.3f}')
        axes[1, 0].fill_between(recall, precision, alpha=0.3)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Curva Precision-Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribución de probabilidades
        axes[1, 1].hist(y_proba[y_true == 0], bins=30, alpha=0.6, 
                        label='No Deserta', color='blue')
        axes[1, 1].hist(y_proba[y_true == 1], bins=30, alpha=0.6, 
                        label='Deserta', color='red')
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', 
                           label='Umbral (0.5)')
        axes[1, 1].set_xlabel('Probabilidad de Deserción')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Probabilidades')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'Modelo'
    ) -> str:
        """
        Genera un reporte textual completo de evaluación.
        
        Args:
            y_true: Labels reales
            y_pred: Predicciones
            y_proba: Probabilidades
            model_name: Nombre del modelo
            
        Returns:
            String con reporte formateado
        """
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'auc_pr': average_precision_score(y_true, y_proba)
        }
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           REPORTE DE EVALUACIÓN - {model_name.upper():<27}║
╚══════════════════════════════════════════════════════════════════╝

📊 MÉTRICAS DE RENDIMIENTO
───────────────────────────────────────────────────────────────────
  Accuracy:       {metrics['accuracy']:.4f}   │ Porcentaje de predicciones correctas
  Precision:      {metrics['precision']:.4f}   │ De los predichos como desertores, % correctos
  Recall:         {metrics['recall']:.4f}   │ De los desertores reales, % detectados
  F1-Score:       {metrics['f1']:.4f}   │ Balance entre precision y recall
  AUC-ROC:        {metrics['auc_roc']:.4f}   │ Capacidad de discriminación global
  AUC-PR:         {metrics['auc_pr']:.4f}   │ Rendimiento en clase minoritaria

📋 MATRIZ DE CONFUSIÓN
───────────────────────────────────────────────────────────────────
                    │     PREDICCIÓN
                    │  No Deserta  │  Deserta
  ──────────────────┼──────────────┼──────────
  REAL  No Deserta  │     {tn:4d}     │   {fp:4d}
        Deserta     │     {fn:4d}     │   {tp:4d}

📈 ANÁLISIS DE RESULTADOS
───────────────────────────────────────────────────────────────────
  ✓ Verdaderos Negativos (TN): {tn:4d} estudiantes correctamente identificados como NO en riesgo
  ⚠ Falsos Positivos (FP):     {fp:4d} falsos alarmas (pérdida de recursos)
  ✗ Falsos Negativos (FN):     {fn:4d} desertores NO detectados (CRÍTICO)
  ✓ Verdaderos Positivos (TP): {tp:4d} desertores correctamente identificados

💡 RECOMENDACIONES
───────────────────────────────────────────────────────────────────
"""
        
        # Agregar recomendaciones basadas en métricas
        if metrics['recall'] < 0.7:
            report += "  • Recall bajo: Considerar bajar el umbral de clasificación para detectar más desertores\n"
        if metrics['precision'] < 0.5:
            report += "  • Precision baja: Muchas falsas alarmas, considerar subir el umbral o mejorar features\n"
        if metrics['auc_roc'] < 0.7:
            report += "  • AUC bajo: El modelo tiene dificultades para discriminar, considerar más features o modelos más complejos\n"
        if fn > tp:
            report += "  • ⚠ Más desertores no detectados que detectados: URGENTE mejorar recall\n"
        if metrics['f1'] >= 0.8:
            report += "  • ✅ Excelente rendimiento general del modelo\n"
        
        return report


if __name__ == "__main__":
    # Demo de evaluación
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("DEMO: EVALUACIÓN DE MODELOS")
    print("=" * 60)
    
    # Crear datos sintéticos
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_classes=2,
        weights=[0.75, 0.25],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test, 'RandomForest')
    
    # Generar reporte
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(evaluator.generate_evaluation_report(y_test, y_pred, y_proba, 'RandomForest'))
