# ML Project Framework - Quick Start Guide

Get up and running with the ML Project Framework in minutes!

## 📋 5-Minute Setup

### Step 1: Install (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install
pip install -r requirements.txt
pip install -e .
```

### Step 2: Run Example (2 minutes)

```bash
python run_pipeline.py
```

### Step 3: Explore (1 minute)

```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

✅ Done! Check `experiments/` and `models/` for results.

---

## 🔧 Common Tasks

### Use Your Own Data

1. **Add your data**

   ```bash
   cp /path/to/data.csv data/raw/
   ```

2. **Update config**

   ```yaml
   # config/config.yaml
   data:
     sources:
       - name: "my_data"
         path: "data/raw/data.csv"
   data:
     target_variable: "my_target_column"
   ```

3. **Run pipeline**
   ```bash
   python run_pipeline.py
   ```

### Change Algorithm

Edit `config/config.yaml`:

```yaml
model:
  algorithm: "xgboost" # or "logistic_regression", "linear_regression"
  params:
    n_estimators: 200
    max_depth: 8
```

### Try Different Problem Types

```yaml
problem:
  type: "regression" # "classification", "regression", "clustering"
  task: "linear_regression"
```

### Enable Feature Scaling

```yaml
features:
  scaling:
    enabled: true
    method: "standard" # "minmax", "robust"
```

### Add Polynomial Features

```yaml
features:
  engineering:
    polynomial_features:
      enabled: true
      degree: 2
```

### Use Cross-Validation

```yaml
training:
  cross_validation:
    enabled: true
    n_folds: 5
```

---

## 📊 Common Code Snippets

### Load and Explore Data

```python
from src.data import DataProcessor

processor = DataProcessor()
df = processor.load_data('data/raw/data.csv')

summary = processor.get_data_summary(df)
print(f"Shape: {summary['shape']}")
print(f"Missing: {summary['missing_values']}")
```

### Train Multiple Models

```python
from src.models import ModelTrainer

algorithms = ['random_forest', 'gradient_boosting']
results = {}

for algo in algorithms:
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, algorithm=algo)
    y_pred = trainer.predict(X_test)
    results[algo] = evaluate_model(y_test, y_pred)
```

### Feature Selection

```python
from src.features import FeatureEngineer

engineer = FeatureEngineer()
top_features = engineer.select_top_features(X_train, y_train, n_features=10)
X_train_selected = X_train[top_features]
```

### Save and Load Models

```python
from src.models import ModelTrainer

trainer = ModelTrainer()
trainer.train(X_train, y_train, algorithm='random_forest')
trainer.save_model('models/my_model.pkl')

# Later...
trainer.load_model('models/my_model.pkl')
y_pred = trainer.predict(X_test)
```

### Customize Features

```python
from src.features import FeatureEngineer

engineer = FeatureEngineer()

# Polynomial features
X_poly = engineer.create_polynomial_features(X, degree=2)

# Interaction features
X_inter = engineer.create_interaction_features(X, [('age', 'income')])

# Binned features
X_binned = engineer.create_binned_features(X, ['age'], n_bins=5)
```

### Get Detailed Metrics

```python
from src.evaluation import evaluate_model

metrics = evaluate_model(y_test, y_pred, problem_type='classification', y_proba=y_proba)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")

# Confusion matrix
cm = metrics['confusion_matrix']
print(cm)

# Classification report
report = metrics['classification_report']
```

---

## 📂 Key Directories

| Directory         | Purpose                     |
| ----------------- | --------------------------- |
| `config/`         | Configuration files         |
| `data/raw/`       | Your raw datasets           |
| `data/processed/` | Cleaned, processed data     |
| `models/final/`   | Trained models              |
| `experiments/`    | Results, metrics, summaries |
| `logs/`           | Execution logs              |
| `src/data/`       | Data processing code        |
| `src/features/`   | Feature engineering code    |
| `src/models/`     | Model training code         |
| `src/evaluation/` | Evaluation code             |
| `notebooks/`      | Jupyter notebooks           |

---

## 🎯 Workflow Examples

### Example 1: Binary Classification

```python
from src.utils import load_config
from src.data import DataProcessor
from src.features import build_features
from src.models import ModelTrainer
from src.evaluation import evaluate_model

config = load_config()

# Load data
processor = DataProcessor()
df = processor.load_data('data/raw/data.csv')
X_train, X_test, y_train, y_test = processor.split_data(df, target_col='target')

# Engineer features
X_train_eng, X_test_eng = build_features(X_train, X_test)

# Train
trainer = ModelTrainer()
trainer.train(X_train_eng, y_train, algorithm='random_forest', problem_type='classification')

# Evaluate
y_pred = trainer.predict(X_test_eng)
metrics = evaluate_model(y_test, y_pred, problem_type='classification')

print(f"F1 Score: {metrics['f1']:.4f}")
```

### Example 2: Regression

```python
from src.utils import load_config
from src.data import DataProcessor
from src.features import build_features
from src.models import ModelTrainer
from src.evaluation import evaluate_model

# Load data
processor = DataProcessor()
df = processor.load_data('data/raw/data.csv')
X_train, X_test, y_train, y_test = processor.split_data(df, target_col='price')

# Engineer features
X_train_eng, X_test_eng = build_features(X_train, X_test)

# Train
trainer = ModelTrainer()
trainer.train(X_train_eng, y_train, algorithm='random_forest', problem_type='regression')

# Evaluate
y_pred = trainer.predict(X_test_eng)
metrics = evaluate_model(y_test, y_pred, problem_type='regression')

print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Example 3: Hyperparameter Tuning

```python
from src.models import ModelTrainer

results = {}

param_sets = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 15},
]

for params in param_sets:
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, algorithm='random_forest', params=params)
    y_pred = trainer.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    key = f"est={params['n_estimators']}, depth={params['max_depth']}"
    results[key] = metrics['f1']

best_params = max(results, key=results.get)
print(f"Best params: {best_params}")
```

---

## ⚡ Performance Tips

1. **Reduce data size** - Start with a sample
2. **Select features** - Use `select_top_features()`
3. **Scale data** - Use StandardScaler
4. **Try simpler models** - LogisticRegression before RandomForest
5. **Use subset** - Test on 10% of data first
6. **Parallelize** - Use `n_jobs=-1` in models

---

## 📚 More Resources

- **Full README**: See `README.md` for complete documentation
- **Tutorial Notebook**: Open `notebooks/01_getting_started.ipynb`
- **Project Planning**: Use `docs/project_requirements.md`
- **Configuration Details**: Review `config/config.yaml`

---

## ❓ FAQ

**Q: How do I add my own data?**
A: Place CSV file in `data/raw/`, update config, run pipeline.

**Q: How do I use a different algorithm?**
A: Edit `config/config.yaml`, change `model.algorithm`.

**Q: Can I add custom features?**
A: Yes! Edit `src/features/build_features.py` and `FeatureEngineer` class.

**Q: How do I track experiments?**
A: Results auto-save to `experiments/` with timestamps.

**Q: Can I use multiple GPUs?**
A: Most algorithms don't support it natively. Use libraries like XGBoost with CUDA.

**Q: How do I deploy my model?**
A: Save with `trainer.save_model()`, load with `trainer.load_model()`.

---

## 🚀 Next Steps

1. ✅ Run `python run_pipeline.py`
2. ✅ Check `experiments/` for results
3. ⬜ Add your own data
4. ⬜ Customize configuration
5. ⬜ Run your first experiment
6. ⬜ Iterate and improve

---

**Happy ML! 🎉**
