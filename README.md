# ML Project Framework - Complete Documentation

A professional, production-ready Python framework for machine learning projects that runs locally on your machine.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [API Reference](#api-reference)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)

## 🎯 Overview

The ML Project Framework provides a complete, modular solution for end-to-end machine learning workflows. It follows industry best practices with:

- **Structured project layout** - Organized, scalable directory structure
- **Configuration management** - YAML-based, environment-agnostic configuration
- **Complete data pipeline** - Loading, cleaning, preprocessing, and validation
- **Feature engineering** - Scaling, encoding, and feature creation
- **Model training** - Support for multiple algorithms with hyperparameter tuning
- **Model evaluation** - Comprehensive metrics, visualizations, and reporting
- **Experiment tracking** - Automatic logging and results management
- **Documentation** - Templates and guides for project documentation

## ✨ Features

### Data Processing

- CSV, Excel, and Parquet file loading
- Missing value handling (drop, mean, median, forward fill, custom)
- Duplicate removal
- Outlier detection and removal (IQR and Z-score methods)
- Automatic categorical encoding
- Train/test/validation split with stratification

### Feature Engineering

- Standardization (StandardScaler, MinMaxScaler, RobustScaler)
- Polynomial feature generation
- Interaction feature creation
- Feature binning (equal width and equal frequency)
- Statistical aggregation features
- Feature selection (mutual information, F-test)

### Model Training

- Random Forest (Classification & Regression)
- Gradient Boosting (XGBoost, LightGBM compatible)
- Logistic Regression
- Linear Regression
- Support Vector Machines
- Easy extensibility for custom algorithms

### Model Evaluation

- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression: R², MSE, RMSE, MAE, MAPE
- Confusion matrices
- Classification reports
- Cross-validation support
- Feature importance analysis
- ROC and Precision-Recall curves

### Experiment Tracking

- Automatic experiment logging
- Model serialization and versioning
- Metrics and artifacts storage
- Experiment comparison and analysis
- Results visualization

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- pip package manager

### 2. Installation

```bash
# Clone or download the framework
cd ml-project-framework

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the framework in editable mode
pip install -e .
```

### 3. Run Example Pipeline

```bash
python run_pipeline.py
```

This will create sample data and run the complete ML pipeline, saving results to `experiments/` and `models/`.

### 4. Use Interactive Notebook

```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

## 📁 Project Structure

```
ml-project-framework/
├── README.md                    # This file
├── QUICKSTART.md               # Quick reference guide
├── requirements.txt            # Python package dependencies
├── setup.py                    # Package installation script
├── run_pipeline.py             # End-to-end example pipeline
│
├── config/
│   └── config.yaml            # Main configuration file
│
├── docs/
│   └── project_requirements.md # Project documentation template
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py    # Data loading and processing
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py  # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   └── train_model.py     # Model training
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_model.py  # Model evaluation
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       └── logger.py          # Logging utilities
│
├── notebooks/
│   └── 01_getting_started.ipynb  # Tutorial Jupyter notebook
│
├── data/
│   ├── raw/                   # Raw input data (not tracked in git)
│   └── processed/             # Processed data
│
├── models/
│   ├── final/                 # Final trained models
│   └── checkpoints/           # Intermediate model checkpoints
│
├── experiments/               # Experiment results
│   ├── metrics_*.json        # Performance metrics
│   ├── experiment_summary_*.json  # Experiment summaries
│   └── visualizations/        # Generated plots and charts
│
├── logs/                      # Execution logs
│   └── pipeline_*.log        # Pipeline execution logs
│
└── tests/                     # Unit tests
    └── test_*.py             # Test files
```

## ⚙️ Configuration

### config.yaml Structure

```yaml
project:
  name: "project_name"
  description: "Project description"
  version: "1.0.0"

problem:
  type: "classification" # classification, regression, clustering, etc.
  task: "binary_classification"

data:
  sources:
    - name: "dataset_name"
      path: "data/raw/data.csv"
      format: "csv"
  preprocessing:
    handle_missing: "drop" # drop, mean, median, forward_fill, value
    remove_duplicates: true
    remove_outliers:
      enabled: false
      method: "iqr" # iqr, zscore
  target_variable: "target"
  test_split: 0.2
  validation_split: 0.1

features:
  scaling:
    enabled: true
    method: "standard" # standard, minmax, robust
  engineering:
    polynomial_features:
      enabled: false
      degree: 2
    interaction_features:
      enabled: false
      pairs: []
    binning:
      enabled: false
      columns: []
      n_bins: 5

model:
  algorithm: "random_forest" # Algorithm selection
  params:
    n_estimators: 100
    max_depth: 10

training:
  cross_validation:
    enabled: false
    n_folds: 5
  early_stopping:
    enabled: false
    patience: 10

evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"

output:
  models_dir: "models/final"
  experiments_dir: "experiments"
  logs_dir: "logs"
```

## 📖 Usage

### Basic Pipeline

```python
from src.utils import load_config, get_logger
from src.data import DataProcessor
from src.features import FeatureEngineer, build_features
from src.models import ModelTrainer
from src.evaluation import evaluate_model

# Load configuration
config = load_config()
logger = get_logger()

# Load and prepare data
processor = DataProcessor()
df = processor.load_data('data/raw/data.csv')
X_train, X_test, y_train, y_test = processor.split_data(df, target_col='target')

# Engineer features
X_train_eng, X_test_eng = build_features(X_train, X_test)

# Train model
trainer = ModelTrainer()
trainer.train(X_train_eng, y_train, algorithm='random_forest')

# Evaluate
y_pred = trainer.predict(X_test_eng)
metrics = evaluate_model(y_test, y_pred, problem_type='classification')

# Save
trainer.save_model('models/final/model.pkl')
```

### Advanced Usage

#### Custom Feature Engineering

```python
engineer = FeatureEngineer()

# Create polynomial features
X_poly = engineer.create_polynomial_features(X, degree=2)

# Create interactions
X_inter = engineer.create_interaction_features(X, [('feat1', 'feat2')])

# Create binned features
X_binned = engineer.create_binned_features(X, ['age', 'income'], n_bins=5)

# Select top features
top_features = engineer.select_top_features(X, y, n_features=10)
```

#### Cross-Validation

```python
from src.models import ModelTrainer

trainer = ModelTrainer()
trainer.train(X_train, y_train, algorithm='random_forest')

cv_results = trainer.cross_validate(X, y, cv=5)
print(f"Mean CV Score: {cv_results['mean']:.4f}")
print(f"Std: {cv_results['std']:.4f}")
```

#### Model Comparison

```python
algorithms = ['random_forest', 'gradient_boosting', 'logistic_regression']
results = {}

for algo in algorithms:
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, algorithm=algo)
    y_pred = trainer.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    results[algo] = metrics['f1']

best_algo = max(results, key=results.get)
print(f"Best algorithm: {best_algo}")
```

## 🔧 API Reference

### DataProcessor

```python
processor = DataProcessor(random_state=42)

# Load data
df = processor.load_data('path/to/data.csv')

# Handle missing values
df_clean = processor.handle_missing_values(df, strategy='drop')

# Remove duplicates
df_clean = processor.remove_duplicates(df_clean)

# Remove outliers
df_clean = processor.remove_outliers(df_clean, columns=['col1', 'col2'], method='iqr')

# Encode categorical variables
df_encoded = processor.encode_categorical(df, columns=['cat1', 'cat2'], fit=True)

# Split data
X_train, X_test, y_train, y_test = processor.split_data(df, target_col='target')

# Get summary
summary = processor.get_data_summary(df)
```

### FeatureEngineer

```python
engineer = FeatureEngineer()

# Scale features
X_scaled = engineer.scale_features(X, method='standard', fit=True)

# Create polynomial features
X_poly = engineer.create_polynomial_features(X, degree=2, fit=True)

# Create interactions
X_inter = engineer.create_interaction_features(X, [('col1', 'col2')])

# Create binned features
X_binned = engineer.create_binned_features(X, columns=['col1'], n_bins=5)

# Create statistical features
X_stats = engineer.create_statistical_features(X, group_cols=['group'], stat_cols=['value'])

# Select features
selected = engineer.select_top_features(X, y, n_features=10, method='mutual_info')
```

### ModelTrainer

```python
trainer = ModelTrainer(random_state=42)

# Train model
trainer.train(X_train, y_train, algorithm='random_forest', problem_type='classification')

# Make predictions
y_pred = trainer.predict(X_test)

# Get probabilities
y_proba = trainer.predict_proba(X_test)

# Cross-validation
cv_results = trainer.cross_validate(X, y, cv=5, scoring='f1')

# Feature importance
importance = trainer.get_feature_importance()

# Save/load
trainer.save_model('path/to/model.pkl')
trainer.load_model('path/to/model.pkl')
```

### Model Evaluation

```python
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance

# Evaluate classification
metrics = evaluate_model(y_true, y_pred, problem_type='classification', y_proba=y_proba)

# Evaluate regression
metrics = evaluate_model(y_true, y_pred, problem_type='regression')

# Confusion matrix
cm_data = plot_confusion_matrix(y_true, y_pred)

# Feature importance
fi_data = plot_feature_importance(feature_importance_dict, top_n=10)
```

## 🏆 Best Practices

1. **Data Separation**
   - Keep raw data in `data/raw/` untouched
   - Process data programmatically
   - Store processed data in `data/processed/`

2. **Configuration-Driven**
   - Use YAML configuration for all parameters
   - Version control configuration files
   - Document configuration changes

3. **Reproducibility**
   - Set random seeds consistently
   - Document data preprocessing steps
   - Save model artifacts with metadata

4. **Version Control**
   - Commit code and configuration
   - Exclude `data/`, `models/`, `experiments/` (use .gitignore)
   - Use meaningful commit messages

5. **Code Organization**
   - Keep data logic in `src/data/`
   - Keep feature logic in `src/features/`
   - Keep model logic in `src/models/`
   - Keep evaluation logic in `src/evaluation/`

6. **Documentation**
   - Fill in `docs/project_requirements.md`
   - Add inline comments for complex logic
   - Update README.md with project-specific information

7. **Testing**
   - Create unit tests in `tests/`
   - Test data processing steps
   - Test feature engineering
   - Test model predictions

8. **Logging**
   - Use provided logger for all operations
   - Check logs in `logs/` directory
   - Review logs for debugging

## 🐛 Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**:

```bash
pip install -e .
```

### Configuration Not Loading

**Problem**: `FileNotFoundError: Config file not found`
**Solution**:

- Check config path in config manager
- Ensure `config/config.yaml` exists
- Use absolute paths

### Out of Memory

**Problem**: `MemoryError` during training
**Solution**:

- Reduce dataset size
- Reduce number of features
- Use smaller batch size
- Enable data sampling

### Model Not Converging

**Problem**: Poor training results
**Solution**:

- Increase number of iterations
- Adjust learning rate
- Normalize/scale features
- Try different algorithm

### Slow Training

**Problem**: Training takes too long
**Solution**:

- Use subset of data for testing
- Reduce number of features
- Use faster algorithm (Logistic Regression vs Random Forest)
- Enable parallel processing

## 🤝 Contributing

To improve the framework:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙋 Support

For issues, questions, or suggestions:

1. Check the QUICKSTART.md guide
2. Review example notebooks
3. Check inline code documentation
4. Review troubleshooting section

## 🎓 Learning Resources

- **Quick Start**: See `QUICKSTART.md`
- **Interactive Tutorial**: Open `notebooks/01_getting_started.ipynb`
- **Project Planning**: Use `docs/project_requirements.md`
- **Configuration Guide**: Review inline comments in `config/config.yaml`
- **API Documentation**: See docstrings in source files

---

**Built with ❤️ for the ML community**

Happy coding! 🚀
