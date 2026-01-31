"""Feature engineering module."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineer:
    """Handle feature engineering and transformation."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = None
        self.one_hot_encoder = None
        self.poly_transformer = None
    
    def scale_features(
        self,
        X: pd.DataFrame,
        method: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            X: Input features
            method: 'standard' (zero-mean, unit-variance), 'minmax' (0-1), 'robust' (outlier-resistant)
            fit: If True, fit scaler. If False, use existing.
        
        Returns:
            Scaled features
        """
        X_scaled = X.copy()
        
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.scaler = scaler_class()
            X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X_scaled
    
    def create_polynomial_features(
        self,
        X: pd.DataFrame,
        degree: int = 2,
        include_bias: bool = False,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            X: Input features
            degree: Degree of polynomial features
            include_bias: Include bias term
            fit: If True, fit transformer. If False, use existing.
        
        Returns:
            Dataframe with polynomial features
        """
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.poly_transformer = PolynomialFeatures(
                degree=degree,
                include_bias=include_bias
            )
            poly_features = self.poly_transformer.fit_transform(X[numerical_cols])
        else:
            if self.poly_transformer is None:
                raise ValueError("Transformer not fitted. Set fit=True first.")
            poly_features = self.poly_transformer.transform(X[numerical_cols])
        
        # Create feature names
        feature_names = self.poly_transformer.get_feature_names_out(numerical_cols)
        
        # Create dataframe with other columns
        X_poly = X.copy()
        X_poly = X_poly.drop(columns=numerical_cols)
        X_poly = pd.concat([
            X_poly,
            pd.DataFrame(poly_features, columns=feature_names, index=X.index)
        ], axis=1)
        
        return X_poly
    
    def create_interaction_features(
        self,
        X: pd.DataFrame,
        column_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features from column pairs.
        
        Args:
            X: Input features
            column_pairs: List of (col1, col2) tuples to create interactions
        
        Returns:
            Dataframe with interaction features added
        """
        X_inter = X.copy()
        
        for col1, col2 in column_pairs:
            if col1 in X.columns and col2 in X.columns:
                interaction_name = f"{col1}_x_{col2}"
                X_inter[interaction_name] = X[col1] * X[col2]
        
        return X_inter
    
    def create_binned_features(
        self,
        X: pd.DataFrame,
        columns: List[str],
        n_bins: int = 5,
        method: str = 'equal_width'
    ) -> pd.DataFrame:
        """
        Create binned (discretized) features.
        
        Args:
            X: Input features
            columns: Columns to bin
            n_bins: Number of bins
            method: 'equal_width' or 'equal_frequency'
        
        Returns:
            Dataframe with binned features added
        """
        X_binned = X.copy()
        
        for col in columns:
            if col in X.columns:
                if method == 'equal_width':
                    X_binned[f"{col}_binned"] = pd.cut(X[col], bins=n_bins, labels=False)
                elif method == 'equal_frequency':
                    X_binned[f"{col}_binned"] = pd.qcut(X[col], q=n_bins, labels=False, duplicates='drop')
        
        return X_binned
    
    def create_statistical_features(
        self,
        X: pd.DataFrame,
        group_cols: List[str],
        stat_cols: List[str],
        statistics: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Create statistical aggregation features.
        
        Args:
            X: Input features
            group_cols: Columns to group by
            stat_cols: Columns to compute statistics on
            statistics: List of statistics ('mean', 'std', 'min', 'max', 'median')
        
        Returns:
            Dataframe with statistical features
        """
        X_stats = X.copy()
        
        for col in stat_cols:
            for stat in statistics:
                if stat == 'mean':
                    feature_name = f"{col}_mean_by_{'_'.join(group_cols)}"
                    X_stats[feature_name] = X.groupby(group_cols)[col].transform('mean')
                elif stat == 'std':
                    feature_name = f"{col}_std_by_{'_'.join(group_cols)}"
                    X_stats[feature_name] = X.groupby(group_cols)[col].transform('std')
                elif stat == 'min':
                    feature_name = f"{col}_min_by_{'_'.join(group_cols)}"
                    X_stats[feature_name] = X.groupby(group_cols)[col].transform('min')
                elif stat == 'max':
                    feature_name = f"{col}_max_by_{'_'.join(group_cols)}"
                    X_stats[feature_name] = X.groupby(group_cols)[col].transform('max')
                elif stat == 'median':
                    feature_name = f"{col}_median_by_{'_'.join(group_cols)}"
                    X_stats[feature_name] = X.groupby(group_cols)[col].transform('median')
        
        return X_stats
    
    def select_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        method: str = 'mutual_info'
    ) -> List[str]:
        """
        Select top features based on importance.
        
        Args:
            X: Input features
            y: Target variable
            n_features: Number of features to select
            method: 'mutual_info' (classification) or 'f_regression' (regression)
        
        Returns:
            List of top feature names
        """
        from sklearn.feature_selection import mutual_info_classif, f_regression, SelectKBest
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
        elif method == 'f_regression':
            selector = SelectKBest(f_regression, k=min(n_features, X.shape[1]))
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        selector.fit(X, y)
        selected_mask = selector.get_support()
        return X.columns[selected_mask].tolist()


def build_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaling_method: str = 'standard',
    create_poly: bool = False,
    poly_degree: int = 2,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete feature engineering pipeline.
    
    Args:
        X_train: Training features
        X_test: Test features
        scaling_method: Method for scaling ('standard', 'minmax', 'robust')
        create_poly: Whether to create polynomial features
        poly_degree: Degree of polynomial features
        interaction_pairs: List of column pairs for interactions
    
    Returns:
        Tuple of (X_train_engineered, X_test_engineered)
    """
    engineer = FeatureEngineer()
    
    # Scale features
    X_train_scaled = engineer.scale_features(X_train, method=scaling_method, fit=True)
    X_test_scaled = engineer.scale_features(X_test, method=scaling_method, fit=False)
    
    X_train_eng = X_train_scaled.copy()
    X_test_eng = X_test_scaled.copy()
    
    # Create polynomial features if requested
    if create_poly:
        X_train_eng = engineer.create_polynomial_features(
            X_train_eng, degree=poly_degree, fit=True
        )
        X_test_eng = engineer.create_polynomial_features(
            X_test_eng, degree=poly_degree, fit=False
        )
    
    # Create interaction features if provided
    if interaction_pairs:
        X_train_eng = engineer.create_interaction_features(X_train_eng, interaction_pairs)
        X_test_eng = engineer.create_interaction_features(X_test_eng, interaction_pairs)
    
    return X_train_eng, X_test_eng
