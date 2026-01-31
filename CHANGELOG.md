# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-31

### Added
- Initial release of ML Project Framework
- Complete data pipeline with loading, cleaning, and preprocessing
- Feature engineering module with scaling, encoding, and transformations
- Model training support for Random Forest, Gradient Boosting, Logistic Regression, Linear Regression, and SVM
- Comprehensive evaluation metrics for classification and regression
- Experiment tracking and automatic result logging
- Configuration management using YAML
- Interactive Jupyter notebook tutorial
- Example end-to-end pipeline script
- Full documentation and quickstart guide
- Unit test structure
- GitHub Actions CI/CD ready (setup)

### Features
#### Data Processing
- CSV, Excel, and Parquet file loading
- Missing value handling (drop, mean, median, forward fill, custom value)
- Duplicate row detection and removal
- Outlier detection using IQR and Z-score methods
- Categorical variable encoding
- Stratified train/test/validation splitting

#### Feature Engineering
- Multiple scaling methods (Standard, MinMax, Robust)
- Polynomial feature generation
- Interaction feature creation
- Feature binning (equal width and frequency)
- Statistical aggregation features
- Feature selection (mutual information, F-test)

#### Model Training
- Multiple algorithm support with hyperparameter configuration
- Cross-validation support
- Feature importance extraction
- Model serialization and loading

#### Model Evaluation
- Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression metrics: R², MSE, RMSE, MAE, MAPE
- Confusion matrices and classification reports
- Metric visualization functions
- Automatic metrics saving to JSON

#### Utilities
- YAML-based configuration management
- Comprehensive logging with UTF-8 and console support
- Windows compatibility (proper unicode handling)

### Documentation
- Comprehensive README with usage examples
- QUICKSTART.md for rapid onboarding
- CONTRIBUTING.md for development guidelines
- Project requirements template
- Inline code documentation with docstrings
- Interactive tutorial notebook

### Infrastructure
- Python 3.8+ support
- Virtual environment setup instructions
- requirements.txt with all dependencies
- setup.py for package installation
- .gitignore for proper version control
- MIT License
- GitHub repository integration

## Planned for Future Releases

### Version 1.1.0
- [ ] XGBoost integration
- [ ] LightGBM integration
- [ ] MLflow experiment tracking
- [ ] Advanced visualization tools
- [ ] Model deployment utilities

### Version 1.2.0
- [ ] GPU acceleration support
- [ ] Distributed training capabilities
- [ ] AutoML features
- [ ] Hyperparameter optimization

### Version 2.0.0
- [ ] Time series support
- [ ] NLP utilities
- [ ] Computer vision utilities
- [ ] Ensemble methods
- [ ] Anomaly detection

---

## Getting Started

If you're new to the framework, start with:
1. [QUICKSTART.md](QUICKSTART.md) - 5-minute setup guide
2. [notebooks/01_getting_started.ipynb](notebooks/01_getting_started.ipynb) - Interactive tutorial
3. [run_pipeline.py](run_pipeline.py) - See a complete example

## Installation

```bash
git clone https://github.com/Aimecol/ml.git
cd ml
pip install -e .
```

## Support

For issues, questions, or feature requests:
- Open an issue on [GitHub Issues](https://github.com/Aimecol/ml/issues)
- Check [QUICKSTART.md](QUICKSTART.md) for troubleshooting
- Review existing documentation in [docs/](docs/)

---

**Version 1.0.0** marks the initial public release of ML Project Framework!
