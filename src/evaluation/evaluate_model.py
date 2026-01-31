"""Model evaluation module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
)


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.metrics = {}
        self.predictions = None
        self.probabilities = None
    
    def evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for ROC-AUC)
            average: Averaging method for multi-class ('weighted', 'macro', 'micro')
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        
        # Add ROC-AUC if probabilities provided
        if y_proba is not None:
            try:
                if y_proba.ndim > 1:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
                else:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = None
        
        self.metrics = metrics
        return metrics
    
    def evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary of evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        
        self.metrics = metrics
        return metrics
    
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix from last evaluation."""
        if 'confusion_matrix' in self.metrics:
            return np.array(self.metrics['confusion_matrix'])
        return None
    
    def get_classification_report(self) -> Optional[Dict]:
        """Get detailed classification report."""
        if 'classification_report' in self.metrics:
            return self.metrics['classification_report']
        return None
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✓ Metrics saved to {filepath}")
    
    def print_metrics(self) -> None:
        """Print metrics to console."""
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        
        for key, value in self.metrics.items():
            if key in ['confusion_matrix', 'classification_report']:
                continue
            
            if isinstance(value, float):
                print(f"{key.upper()}: {value:.4f}")
            else:
                print(f"{key.upper()}: {value}")
        
        print("="*50 + "\n")


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    problem_type: str = 'classification',
    y_proba: Optional[np.ndarray] = None,
    metrics_save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        problem_type: 'classification' or 'regression'
        y_proba: Prediction probabilities (classification only)
        metrics_save_path: Path to save metrics JSON
    
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator()
    
    if problem_type == 'classification':
        metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
    elif problem_type == 'regression':
        metrics = evaluator.evaluate_regression(y_true, y_pred)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    evaluator.print_metrics()
    
    if metrics_save_path:
        evaluator.save_metrics(metrics_save_path)
    
    return metrics


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Create confusion matrix visualization data.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with matrix and labels for visualization
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    
    return {
        'matrix': cm.tolist(),
        'labels': labels.tolist(),
        'title': 'Confusion Matrix'
    }


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Create ROC curve visualization data.
    
    Args:
        y_true: True labels (binary)
        y_proba: Prediction probabilities
    
    Returns:
        Dictionary with FPR, TPR, and AUC for visualization
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': auc,
        'title': f'ROC Curve (AUC = {auc:.4f})'
    }


def plot_feature_importance(model_feature_importance: Dict[str, float], top_n: int = 10) -> Dict[str, Any]:
    """
    Create feature importance visualization data.
    
    Args:
        model_feature_importance: Dictionary of feature names and importance
        top_n: Number of top features to show
    
    Returns:
        Dictionary with features and importance values for visualization
    """
    sorted_features = sorted(
        model_feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    features, importance = zip(*sorted_features)
    
    return {
        'features': list(features),
        'importance': list(importance),
        'title': f'Top {top_n} Feature Importance'
    }
