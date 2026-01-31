"""End-to-end ML pipeline example script."""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import get_logger, load_config
from src.data import DataProcessor
from src.features import FeatureEngineer, build_features
from src.models import ModelTrainer
from src.evaluation import evaluate_model


def create_sample_data(problem_type='classification', n_samples=1000, n_features=10):
    """Create sample dataset for demonstration."""
    print(f"\n📊 Creating sample {problem_type} dataset...")
    
    if problem_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_classes=2,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            random_state=42
        )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    return df


def main():
    """Run the complete ML pipeline."""
    
    # Setup logging
    logger = get_logger('ml_pipeline', log_dir='logs')
    logger.info("="*60)
    logger.info("ML PROJECT FRAMEWORK - END-TO-END PIPELINE")
    logger.info("="*60)
    
    # Load configuration
    logger.info("[*] Loading configuration...")
    config = load_config()
    
    problem_type = config.get('problem.type', 'classification')
    algorithm = config.get('model.algorithm', 'random_forest')
    model_params = config.get('model.params', {})
    
    logger.info(f"Problem Type: {problem_type}")
    logger.info(f"Algorithm: {algorithm}")
    
    # Step 1: Create/Load Data
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("="*60)
    
    # Create sample data (replace with your actual data loading)
    print(f"\n[*] Creating sample {problem_type} dataset...")
    df = create_sample_data(problem_type, n_samples=1000, n_features=10)
    logger.info(f"[+] Dataset created: {df.shape}")
    
    # Data preprocessing
    processor = DataProcessor(random_state=42)
    summary = processor.get_data_summary(df)
    logger.info(f"  - Features: {summary['numerical_columns']}")
    logger.info(f"  - Target: target")
    logger.info(f"  - Shape: {summary['shape']}")
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(
        df,
        target_col='target',
        test_size=0.2
    )
    logger.info(f"[+] Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    
    # Step 2: Feature Engineering
    logger.info("\n" + "="*60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*60)
    
    X_train_eng, X_test_eng = build_features(
        X_train,
        X_test,
        scaling_method='standard',
        create_poly=False,
        interaction_pairs=None
    )
    logger.info(f"[+] Features engineered")
    logger.info(f"  - Train shape: {X_train_eng.shape}")
    logger.info(f"  - Test shape: {X_test_eng.shape}")
    
    # Step 3: Model Training
    logger.info("\n" + "="*60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*60)
    
    trainer = ModelTrainer(random_state=42)
    trainer.train(
        X_train_eng,
        y_train,
        algorithm=algorithm,
        problem_type=problem_type,
        params=model_params
    )
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/final/{algorithm}_model_{timestamp}.pkl'
    trainer.save_model(model_path)
    logger.info(f"[+] Model saved: {model_path}")
    
    # Step 4: Model Evaluation
    logger.info("\n" + "="*60)
    logger.info("STEP 4: MODEL EVALUATION")
    logger.info("="*60)
    
    # Training performance
    y_train_pred = trainer.predict(X_train_eng)
    if problem_type == 'classification':
        train_metrics = evaluate_model(
            y_train, y_train_pred,
            problem_type=problem_type,
            y_proba=trainer.predict_proba(X_train_eng) if hasattr(trainer.model, 'predict_proba') else None
        )
    else:
        train_metrics = evaluate_model(y_train, y_train_pred, problem_type=problem_type)
    
    logger.info("Training Set Performance:")
    for key, value in train_metrics.items():
        if key not in ['confusion_matrix', 'classification_report']:
            logger.info(f"  {key}: {value}")
    
    # Test performance
    y_test_pred = trainer.predict(X_test_eng)
    if problem_type == 'classification':
        test_metrics = evaluate_model(
            y_test, y_test_pred,
            problem_type=problem_type,
            y_proba=trainer.predict_proba(X_test_eng) if hasattr(trainer.model, 'predict_proba') else None,
            metrics_save_path=f'experiments/metrics_{timestamp}.json'
        )
    else:
        test_metrics = evaluate_model(
            y_test, y_test_pred,
            problem_type=problem_type,
            metrics_save_path=f'experiments/metrics_{timestamp}.json'
        )
    
    logger.info("Test Set Performance:")
    for key, value in test_metrics.items():
        if key not in ['confusion_matrix', 'classification_report']:
            logger.info(f"  {key}: {value}")
    
    # Step 5: Feature Importance (if available)
    if hasattr(trainer.model, 'feature_importances_'):
        logger.info("\n" + "="*60)
        logger.info("STEP 5: FEATURE IMPORTANCE")
        logger.info("="*60)
        
        feature_names = X_train_eng.columns
        importances = trainer.model.feature_importances_
        
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 Important Features:")
        for feat, imp in top_features:
            logger.info(f"  {feat}: {imp:.4f}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY [+]")
    logger.info("="*60)
    logger.info(f"Outputs saved to:")
    logger.info(f"  - Model: {model_path}")
    logger.info(f"  - Metrics: experiments/metrics_{timestamp}.json")
    logger.info(f"  - Logs: logs/")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
