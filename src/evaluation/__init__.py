"""Evaluation module."""

from .evaluate_model import (
    ModelEvaluator,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

__all__ = [
    'ModelEvaluator',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_importance'
]
