"""Web interface routes and API endpoints."""

from flask import Blueprint, render_template, request, jsonify, send_file, current_app
import os
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import get_logger, load_config
from src.data import DataProcessor
from src.features import FeatureEngineer
from src.models import ModelTrainer
from src.evaluation import evaluate_model

# Create blueprints
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)

# Get logger
logger = get_logger(__name__)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


# ==================== Web Routes ====================

@main_bp.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('index.html')


@main_bp.route('/pipeline')
def pipeline():
    """Pipeline configuration page."""
    config = load_config()
    return render_template('pipeline.html', config=config)


@main_bp.route('/results')
def results():
    """Results and experiments page."""
    experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    experiments = []
    
    if os.path.exists(experiments_dir):
        for file in sorted(os.listdir(experiments_dir), reverse=True):
            if file.endswith('.json') and file.startswith('metrics_'):
                experiments.append(file)
    
    return render_template('results.html', experiments=experiments)


@main_bp.route('/documentation')
def documentation():
    """Documentation page."""
    return render_template('documentation.html')


# ==================== API Endpoints ====================

@api_bp.route('/status', methods=['GET'])
def status():
    """Get system status."""
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload dataset file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(current_app.config["ALLOWED_EXTENSIONS"])}'}), 400
        
        # Save file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/preview-data', methods=['POST'])
def preview_data():
    """Preview uploaded data."""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        processor = DataProcessor(logger=logger)
        df = processor.load_data(filepath)
        
        preview_rows = df.head(10).to_dict('records')
        
        return jsonify({
            'success': True,
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'preview': preview_rows,
            'missing_values': df.isnull().sum().to_dict()
        })
    
    except Exception as e:
        logger.error(f"Preview error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/train-model', methods=['POST'])
def train_model():
    """Train model with specified parameters."""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        config_overrides = data.get('config', {})
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        logger.info(f"Starting model training with file: {filepath}")
        
        # Load and process data
        processor = DataProcessor(logger=logger)
        df = processor.load_data(filepath)
        
        # Handle missing values
        missing_strategy = config_overrides.get('missing_value_strategy', 'drop')
        df = processor.handle_missing_values(df, strategy=missing_strategy)
        
        # Remove duplicates
        df = processor.remove_duplicates(df)
        
        # Separate features and target
        target_col = config_overrides.get('target_column', 'target')
        if target_col not in df.columns:
            return jsonify({'error': f'Target column "{target_col}" not found in data'}), 400
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Feature engineering
        engineer = FeatureEngineer(logger=logger)
        
        # Remove outliers if specified
        if config_overrides.get('remove_outliers', False):
            outlier_method = config_overrides.get('outlier_method', 'iqr')
            X, y = engineer.remove_outliers(X, y, method=outlier_method)
        
        # Scale features
        scaling_method = config_overrides.get('scaling_method', 'standard')
        X_scaled = engineer.scale_features(X, method=scaling_method)
        
        # Split data
        test_size = config_overrides.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = processor.split_data(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train model
        algorithm = config_overrides.get('algorithm', 'random_forest')
        model_params = config_overrides.get('model_params', {})
        
        trainer = ModelTrainer(logger=logger)
        model = trainer.train_model(
            X_train, y_train,
            algorithm=algorithm,
            **model_params
        )
        
        # Evaluate model
        problem_type = config_overrides.get('problem_type', 'classification')
        metrics = evaluate_model(
            model, X_test, y_test,
            problem_type=problem_type,
            logger=logger
        )
        
        # Save results
        experiment_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        os.makedirs(experiment_dir, exist_ok=True)
        
        experiment_name = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        experiment_path = os.path.join(experiment_dir, experiment_name)
        
        with open(experiment_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Experiment saved: {experiment_name}")
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'experiment_name': experiment_name,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        })
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/experiments', methods=['GET'])
def list_experiments():
    """List all experiments."""
    try:
        experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        experiments = []
        
        if os.path.exists(experiments_dir):
            for file in sorted(os.listdir(experiments_dir), reverse=True):
                if file.endswith('.json') and file.startswith('metrics_'):
                    filepath = os.path.join(experiments_dir, file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    experiments.append({
                        'filename': file,
                        'timestamp': file.replace('metrics_', '').replace('.json', ''),
                        'metrics': data
                    })
        
        return jsonify({'success': True, 'experiments': experiments})
    
    except Exception as e:
        logger.error(f"Experiments listing error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/experiment/<filename>', methods=['GET'])
def get_experiment(filename):
    """Get specific experiment details."""
    try:
        experiment_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        filepath = os.path.join(experiment_dir, filename)
        
        if not os.path.exists(filepath) or not filename.endswith('.json'):
            return jsonify({'error': 'Experiment not found'}), 404
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return jsonify({'success': True, 'data': data})
    
    except Exception as e:
        logger.error(f"Experiment retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    try:
        config = load_config()
        return jsonify({'success': True, 'config': config})
    
    except Exception as e:
        logger.error(f"Config retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/algorithms', methods=['GET'])
def get_algorithms():
    """Get available algorithms."""
    algorithms = {
        'classification': [
            {'name': 'random_forest', 'label': 'Random Forest'},
            {'name': 'gradient_boosting', 'label': 'Gradient Boosting'},
            {'name': 'logistic_regression', 'label': 'Logistic Regression'},
            {'name': 'svm', 'label': 'Support Vector Machine'}
        ],
        'regression': [
            {'name': 'random_forest', 'label': 'Random Forest'},
            {'name': 'gradient_boosting', 'label': 'Gradient Boosting'},
            {'name': 'linear_regression', 'label': 'Linear Regression'},
            {'name': 'svm', 'label': 'Support Vector Machine'}
        ]
    }
    return jsonify({'success': True, 'algorithms': algorithms})


@api_bp.route('/generate-sample-data', methods=['POST'])
def generate_sample_data():
    """Generate sample dataset for learning."""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification, make_regression
        
        data = request.get_json()
        problem_type = data.get('problem_type', 'classification')
        n_samples = data.get('n_samples', 500)
        n_features = data.get('n_features', 10)
        
        logger.info(f"Generating {problem_type} sample data: {n_samples} samples, {n_features} features")
        
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
        
        # Save sample data
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = f"sample_{problem_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(upload_folder, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Sample data generated: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'preview': df.head(5).to_dict('records')
        })
    
    except Exception as e:
        logger.error(f"Sample data generation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/help', methods=['GET'])
def get_help():
    """Get interactive help content."""
    try:
        help_content = {
            'algorithms': {
                'random_forest': {
                    'name': 'Random Forest',
                    'description': 'Ensemble method combining multiple decision trees. Great for most problems.',
                    'best_for': ['Classification', 'Regression', 'Feature importance'],
                    'pros': ['Fast', 'Handles non-linear relationships', 'Robust to outliers'],
                    'cons': ['Can overfit on small datasets', 'Uses more memory'],
                    'tips': 'Start with default parameters. Increase n_estimators for better performance.',
                    'example_params': {'n_estimators': 100, 'max_depth': 10}
                },
                'gradient_boosting': {
                    'name': 'Gradient Boosting',
                    'description': 'Sequentially builds trees to correct errors. Very powerful but slower.',
                    'best_for': ['Complex patterns', 'High accuracy requirements'],
                    'pros': ['Often best accuracy', 'Handles complex relationships'],
                    'cons': ['Slower training', 'Risk of overfitting'],
                    'tips': 'Use lower learning rate (0.01-0.1) for better generalization.',
                    'example_params': {'n_estimators': 100, 'learning_rate': 0.1}
                },
                'logistic_regression': {
                    'name': 'Logistic Regression',
                    'description': 'Linear model for classification. Fast and interpretable.',
                    'best_for': ['Fast predictions', 'Interpretability', 'Linear relationships'],
                    'pros': ['Very fast', 'Easy to interpret', 'Good baseline'],
                    'cons': ['Only for linear problems', 'Classification only'],
                    'tips': 'Great as a baseline. Compare with other models.',
                    'example_params': {'max_iter': 1000}
                },
                'svm': {
                    'name': 'Support Vector Machine',
                    'description': 'Powerful model for complex decision boundaries.',
                    'best_for': ['High-dimensional data', 'Complex boundaries'],
                    'pros': ['Works well in high dimensions', 'Flexible kernels'],
                    'cons': ['Slower on large datasets', 'Needs scaling'],
                    'tips': 'Always scale features before using SVM.',
                    'example_params': {'kernel': 'rbf', 'C': 1.0}
                },
                'linear_regression': {
                    'name': 'Linear Regression',
                    'description': 'Fits a linear relationship between features and target.',
                    'best_for': ['Linear relationships', 'Regression problems'],
                    'pros': ['Interpretable', 'Fast', 'Good baseline'],
                    'cons': ['Only linear patterns', 'Sensitive to outliers'],
                    'tips': 'Use Robust Scaler if you have outliers.',
                    'example_params': {}
                }
            },
            'scaling_methods': {
                'standard': {
                    'name': 'Standard Scaler',
                    'description': 'Mean=0, Standard Deviation=1. Good for normally distributed data.',
                    'when_to_use': 'Most algorithms, normally distributed features',
                    'formula': '(x - mean) / std_dev',
                    'pro_tip': 'Default choice for most cases'
                },
                'minmax': {
                    'name': 'MinMax Scaler',
                    'description': 'Scales to [0, 1] range. Preserves zero-centered data.',
                    'when_to_use': 'Neural networks, image data, bounded features',
                    'formula': '(x - min) / (max - min)',
                    'pro_tip': 'Good when you know the feature bounds'
                },
                'robust': {
                    'name': 'Robust Scaler',
                    'description': 'Uses median and IQR. Resistant to outliers.',
                    'when_to_use': 'Data with outliers',
                    'formula': '(x - median) / IQR',
                    'pro_tip': 'Use when your data has extreme values'
                }
            },
            'missing_value_strategies': {
                'drop': {
                    'name': 'Drop Rows',
                    'description': 'Remove any row with missing values.',
                    'when_to_use': 'When missing values are rare (<5%)',
                    'pro_tip': 'Fastest but loses data',
                    'impact': 'May lose valuable information'
                },
                'mean': {
                    'name': 'Fill with Mean',
                    'description': 'Replace missing with column average.',
                    'when_to_use': 'When missing values are random',
                    'pro_tip': 'Good for numeric features',
                    'impact': 'Reduces variance slightly'
                },
                'median': {
                    'name': 'Fill with Median',
                    'description': 'Replace missing with column median.',
                    'when_to_use': 'When you have outliers',
                    'pro_tip': 'More robust than mean',
                    'impact': 'Better with skewed distributions'
                }
            },
            'general_tips': {
                'data_preparation': [
                    'Ensure target column exists and is named correctly',
                    'Features should be numeric (encode text first)',
                    'Check for missing values using data preview',
                    'Look for outliers in your data'
                ],
                'model_selection': [
                    'Start with Random Forest as a baseline',
                    'Compare multiple algorithms',
                    'Check if your problem is linear or non-linear',
                    'Consider training time vs accuracy trade-off'
                ],
                'interpreting_results': [
                    'Accuracy: Overall correctness (watch for class imbalance)',
                    'F1 Score: Balance between precision and recall',
                    'ROC-AUC: Model\'s ability to distinguish classes',
                    'Confusion Matrix: See types of errors made'
                ]
            }
        }
        return jsonify({'success': True, 'help': help_content})
    
    except Exception as e:
        logger.error(f"Help retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500
