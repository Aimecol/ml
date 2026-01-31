"""Data loading, cleaning, and preprocessing module."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


class DataProcessor:
    """Handle data loading, cleaning, and preprocessing."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize data processor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            filepath: Path to data file (CSV, Excel, etc.)
        
        Returns:
            Loaded dataframe
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return df
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'drop',
        fill_value: Any = None
    ) -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: Input dataframe
            strategy: 'drop' (remove rows), 'mean', 'median', 'forward_fill', 'value'
            fill_value: Value to use for strategy='value'
        
        Returns:
            Dataframe with missing values handled
        """
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif strategy == 'value':
            df = df.fillna(fill_value)
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input dataframe
            subset: Columns to consider for duplicates. If None, all columns.
        
        Returns:
            Dataframe with duplicates removed
        """
        return df.drop_duplicates(subset=subset)
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers using IQR or Z-score method.
        
        Args:
            df: Input dataframe
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
        
        Returns:
            Dataframe with outliers removed
        """
        df = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input dataframe
            columns: Columns to encode
            fit: If True, fit new encoders. If False, use existing.
        
        Returns:
            Dataframe with encoded categorical variables
        """
        df = df.copy()
        
        for col in columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        validation_size: Optional[float] = None
    ) -> Tuple:
        """
        Split data into train/test or train/validation/test.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            test_size: Fraction for test set
            validation_size: Fraction for validation set. If None, only train/test split.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if validation_size is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y if len(y.unique()) < 20 else None
            )
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=test_size + validation_size,
                random_state=self.random_state,
                stratify=y if len(y.unique()) < 20 else None
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=test_size / (test_size + validation_size),
                random_state=self.random_state,
                stratify=y_temp if len(y_temp.unique()) < 20 else None
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of data.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dictionary with data summary
        """
        return {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'duplicates': df.duplicated().sum()
        }


def make_dataset(
    raw_data_path: str,
    processed_data_path: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data preparation pipeline.
    
    Args:
        raw_data_path: Path to raw data file
        processed_data_path: Path to save processed data
        target_col: Name of target column
        test_size: Test set fraction
        random_state: Random seed
    
    Returns:
        Tuple of (train_data, test_data)
    """
    processor = DataProcessor(random_state=random_state)
    
    # Load data
    df = processor.load_data(raw_data_path)
    
    # Clean data
    df = processor.handle_missing_values(df, strategy='drop')
    df = processor.remove_duplicates(df)
    
    # Get data summary
    summary = processor.get_data_summary(df)
    print("Data Summary:")
    print(f"  Shape: {summary['shape']}")
    print(f"  Duplicates: {summary['duplicates']}")
    print(f"  Missing values: {summary['missing_values']}")
    
    # Encode categorical variables
    categorical_cols = summary['categorical_columns']
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    if categorical_cols:
        df = processor.encode_categorical(df, categorical_cols, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(
        df, target_col, test_size=test_size
    )
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    train_data.to_csv(processed_data_path.replace('.csv', '_train.csv'), index=False)
    test_data.to_csv(processed_data_path.replace('.csv', '_test.csv'), index=False)
    
    return train_data, test_data
