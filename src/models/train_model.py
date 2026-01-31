"""Model training module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pickle
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score


class ModelTrainer:
    """Train and manage ML models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.model_name = None
        self.training_history = {}
    
    def get_model(self, algorithm: str, problem_type: str, params: Optional[Dict[str, Any]] = None):
        """
        Get model instance based on algorithm and problem type.
        
        Args:
            algorithm: Algorithm name ('random_forest', 'xgboost', 'logistic_regression', etc.)
            problem_type: 'classification' or 'regression'
            params: Model hyperparameters
        
        Returns:
            Model instance
        """
        params = params or {}
        params = params.copy()  # Don't modify original params
        
        # Filter parameters based on algorithm
        valid_rf_params = {'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state'}
        valid_gb_params = {'n_estimators', 'max_depth', 'learning_rate', 'min_samples_split', 'min_samples_leaf', 'random_state'}
        valid_lr_params = {'C', 'fit_intercept', 'max_iter'}
        valid_svm_params = {'C', 'kernel', 'gamma', 'probability'}
        
        if algorithm == 'random_forest':
            filtered_params = {k: v for k, v in params.items() if k in valid_rf_params}
            filtered_params['random_state'] = self.random_state
            if problem_type == 'classification':
                return RandomForestClassifier(**filtered_params)
            else:
                return RandomForestRegressor(**filtered_params)
        
        elif algorithm == 'gradient_boosting':
            filtered_params = {k: v for k, v in params.items() if k in valid_gb_params}
            filtered_params['random_state'] = self.random_state
            if problem_type == 'classification':
                return GradientBoostingClassifier(**filtered_params)
            else:
                return GradientBoostingRegressor(**filtered_params)
        
        elif algorithm == 'logistic_regression':
            if problem_type == 'classification':
                filtered_params = {k: v for k, v in params.items() if k in valid_lr_params}
                return LogisticRegression(random_state=self.random_state, **filtered_params)
            else:
                raise ValueError("Logistic regression is for classification only")
        
        elif algorithm == 'linear_regression':
            if problem_type == 'regression':
                return LinearRegression()
            else:
                raise ValueError("Linear regression is for regression only")
        
        elif algorithm == 'svm':
            filtered_params = {k: v for k, v in params.items() if k in valid_svm_params}
            if problem_type == 'classification':
                return SVC(**filtered_params)
            else:
                return SVR(**filtered_params)
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: str,
        problem_type: str,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Train a model.
        
        Args:
            X_train: Training features
            y_train: Training target
            algorithm: Algorithm name
            problem_type: 'classification' or 'regression'
            params: Model hyperparameters
        """
        self.model = self.get_model(algorithm, problem_type, params)
        self.model_name = algorithm
        
        print(f"Training {algorithm} model...")
        self.model.fit(X_train, y_train)
        
        self.training_history = {
            'algorithm': algorithm,
            'problem_type': problem_type,
            'train_samples': X_train.shape[0],
            'train_features': X_train.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Features to predict on
        
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("This model does not support predict_proba")
        
        return self.model.predict_proba(X)
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
        
        Returns:
            Cross-validation results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (if available).
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        # Assuming features are in order
        return {
            f'feature_{i}': imp
            for i, imp in enumerate(self.model.feature_importances_)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ Model loaded from {filepath}")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm: str = 'random_forest',
    problem_type: str = 'classification',
    params: Optional[Dict[str, Any]] = None,
    model_save_path: Optional[str] = None
):
    """
    Train a model with standard pipeline.
    
    Args:
        X_train: Training features
        y_train: Training target
        algorithm: Algorithm name
        problem_type: 'classification' or 'regression'
        params: Model hyperparameters
        model_save_path: Path to save trained model
    
    Returns:
        Trained model
    """
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, algorithm, problem_type, params)
    
    if model_save_path:
        trainer.save_model(model_save_path)
    
    return trainer.model
