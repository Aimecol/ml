# Web Interface Guide

## Overview

The ML Project Framework includes a modern, user-friendly web interface for managing machine learning projects without writing code. The web interface provides:

- **Interactive Dashboard**: View experiment history and system status
- **Data Upload & Preview**: Upload datasets and preview data before training
- **Pipeline Configuration**: Configure training parameters with a visual form
- **Model Training**: Train models with custom algorithms and hyperparameters
- **Results Visualization**: Compare experiments and view detailed metrics
- **REST API**: Access all features programmatically

## Installation

### Prerequisites
- Python 3.8 or higher
- The ML Project Framework installed

### Install Web Dependencies

The web interface requires additional packages:

```bash
pip install flask flask-cors
```

Or install with the requirements file:

```bash
pip install -r requirements.txt
```

## Running the Web Interface

### Basic Usage

Start the web server:

```bash
python run_web.py
```

The interface will be available at `http://localhost:5000`

### Advanced Options

**Change port:**
```bash
python run_web.py --port 8000
```

**Make accessible from other machines:**
```bash
python run_web.py --host 0.0.0.0 --port 8000
```

**Production mode (no debug, no auto-reload):**
```bash
python run_web.py --no-debug --no-reload
```

**View all options:**
```bash
python run_web.py --help
```

## Interface Guide

### 1. Dashboard

The dashboard provides a high-level overview of your experiments:

- **Total Experiments**: Number of completed training runs
- **Last Accuracy**: Best metric from the most recent experiment
- **Framework Status**: System health and version
- **Quick Start**: Direct links to get started
- **Recent Experiments**: List of your last 5 experiments

### 2. Pipeline Tab

The pipeline tab is where you train models. It consists of 4 steps:

#### Step 1: Upload Data
- Click the upload area or drag and drop a file
- Supported formats: CSV, Excel (.xlsx, .xls), Parquet
- Maximum file size: 16 MB

#### Step 2: Data Preview
- Automatically displays after upload
- Shows data shape (rows × columns)
- Lists all column names
- Displays data types
- Shows count of missing values
- First 10 rows of data

#### Step 3: Configure Training
- **Problem Type**: Choose between Classification or Regression
- **Target Column**: Name of the column to predict (default: "target")
- **Algorithm**: Select from:
  - Random Forest (recommended for most cases)
  - Gradient Boosting (powerful but slower)
  - Logistic Regression (fast, linear)
  - Support Vector Machine (good for complex boundaries)
- **Feature Scaling**:
  - Standard Scaler: Mean 0, standard deviation 1 (recommended)
  - MinMax Scaler: Scale to [0, 1] range
  - Robust Scaler: Resistant to outliers
- **Missing Values**:
  - Drop: Remove rows with missing values
  - Mean: Fill with column average
  - Median: Fill with column median
- **Test Set Size**: Percentage of data for testing (0.1-0.5)
- **Remove Outliers**: Optional outlier detection using IQR method
- **Model Parameters**: Advanced JSON configuration (optional)

#### Step 4: Results
- Automatically displayed after training completes
- Shows key metrics (accuracy, F1 score, etc.)
- Training and test set sizes
- Experiment identifier for reference

### 3. Results Tab

Browse and compare all experiments:

- **Experiment Cards**: Each experiment shows:
  - Filename with timestamp
  - Main metric (Accuracy or R² Score)
  - Additional metrics
  - "View Full Report" button

- **Full Report**: Click to see:
  - All numeric metrics
  - Confusion matrix (classification)
  - Classification report
  - Detailed performance breakdown

### 4. Documentation Tab

Complete in-app documentation including:

- Feature overview
- Step-by-step guide
- Configuration explanations
- Tips and best practices
- Troubleshooting guide
- API endpoint reference
- FAQ

## API Reference

### File Operations

#### Upload File
```
POST /api/upload
Content-Type: multipart/form-data

Parameters:
  file: <file> - The data file (CSV, Excel, Parquet)

Response:
{
  "success": true,
  "filename": "upload_20260131_120000_data.csv",
  "filepath": "/path/to/file"
}
```

#### Preview Data
```
POST /api/preview-data
Content-Type: application/json

Body:
{
  "filepath": "/path/to/file"
}

Response:
{
  "success": true,
  "shape": {"rows": 1000, "columns": 15},
  "columns": ["feature_1", "feature_2", ..., "target"],
  "dtypes": {"feature_1": "int64", ...},
  "preview": [... 10 rows ...],
  "missing_values": {"feature_1": 5, ...}
}
```

### Training

#### Train Model
```
POST /api/train-model
Content-Type: application/json

Body:
{
  "filepath": "/path/to/data.csv",
  "config": {
    "problem_type": "classification",
    "target_column": "target",
    "algorithm": "random_forest",
    "scaling_method": "standard",
    "missing_value_strategy": "drop",
    "test_size": 0.2,
    "remove_outliers": false,
    "model_params": {
      "n_estimators": 100,
      "max_depth": 10
    }
  }
}

Response:
{
  "success": true,
  "metrics": {...},
  "experiment_name": "metrics_20260131_120000.json",
  "training_samples": 800,
  "test_samples": 200
}
```

### Experiments

#### List All Experiments
```
GET /api/experiments

Response:
{
  "success": true,
  "experiments": [
    {
      "filename": "metrics_20260131_120000.json",
      "timestamp": "20260131_120000",
      "metrics": {...}
    },
    ...
  ]
}
```

#### Get Experiment Details
```
GET /api/experiment/{filename}

Response:
{
  "success": true,
  "data": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.97,
    "f1_score": 0.95,
    ...
    "confusion_matrix": [[...], [...]],
    "classification_report": {...}
  }
}
```

### Configuration & Status

#### Get System Status
```
GET /api/status

Response:
{
  "status": "running",
  "version": "1.0.0",
  "timestamp": "2026-01-31T12:00:00"
}
```

#### Get Configuration
```
GET /api/config

Response:
{
  "success": true,
  "config": {
    "data": {...},
    "features": {...},
    "model": {...},
    ...
  }
}
```

#### Get Available Algorithms
```
GET /api/algorithms

Response:
{
  "success": true,
  "algorithms": {
    "classification": [
      {"name": "random_forest", "label": "Random Forest"},
      ...
    ],
    "regression": [
      {"name": "random_forest", "label": "Random Forest"},
      ...
    ]
  }
}
```

## Tips & Best Practices

### Data Preparation
- **Target Column**: Always have a column named "target" or specify the correct column name
- **Data Types**: Numeric features are recommended. Use a preprocessing tool for text/categorical data
- **Missing Values**: Handle appropriately - remove rows or fill with mean/median
- **Data Size**: For large datasets (>1GB), consider sampling or preprocessing separately

### Model Training
- **Start Simple**: Begin with Random Forest and default parameters
- **Monitor Progress**: Watch the browser console for any errors
- **Experiment Systematically**: Change one parameter at a time to understand its effect
- **Check Results**: Review confusion matrix and classification report, not just accuracy

### Performance
- **Large Datasets**: May take time to upload and train. Be patient!
- **Complex Models**: Gradient Boosting and SVM may be slow on large datasets
- **Memory**: Training uses RAM. Close other applications if you run out of memory

## Troubleshooting

### Upload Fails
- **File too large**: Maximum is 16 MB. Split your data.
- **Unsupported format**: Use CSV, Excel (.xlsx), or Parquet only
- **Corrupted file**: Re-export or recreate the file

### Training Fails
- **Target column not found**: Check spelling and capitalization
- **Invalid JSON parameters**: Make sure JSON syntax is correct
- **Out of memory**: Reduce dataset size or use a simpler model
- **Check logs**: Look in `logs/` folder for detailed error messages

### Results Not Showing
- **Page not refreshing**: Hard refresh (Ctrl+F5 or Cmd+Shift+R)
- **Slow processing**: Large datasets take time. Wait a moment and check the Results tab again

### Server Issues
- **Address already in use**: Another app is using port 5000. Use `--port 8000`
- **Connection refused**: Make sure the server is running
- **SSL errors**: Development server doesn't use SSL. Use HTTPS proxy for production

## Deployment

### Development
```bash
python run_web.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "web:create_app()"
```

### Docker
A Dockerfile can be added in the future for containerized deployment.

### Using Behind a Proxy
For HTTPS/production, use a reverse proxy like Nginx:

```nginx
upstream ml_framework {
    server 127.0.0.1:5000;
}

server {
    listen 443 ssl http2;
    server_name example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://ml_framework;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Security Considerations

### Current Limitations
- **Development Mode**: The default server is not suitable for production use
- **No Authentication**: No login system implemented
- **File Upload**: No file validation beyond extension checking
- **HTTPS**: Not enabled by default

### Recommendations for Production
1. Use HTTPS with valid certificates
2. Add authentication/authorization
3. Validate and sanitize all uploads
4. Use a production WSGI server (Gunicorn, uWSGI)
5. Run behind a reverse proxy (Nginx, Apache)
6. Implement rate limiting
7. Monitor and log all activities
8. Keep dependencies updated

## FAQ

**Q: Can I use this with large datasets?**
A: Yes, but upload and training may be slow. Maximum file size is 16 MB.

**Q: What if I have categorical features?**
A: Encode them before uploading, or use another preprocessing tool first.

**Q: Can I download my models?**
A: Models are saved to `models/final/` directory as pickle files.

**Q: Can I integrate this with my own application?**
A: Yes! Use the REST API endpoints to integrate the framework into your application.

**Q: Is the data stored permanently?**
A: Uploaded data goes to `data/raw/`. Experiments are saved to `experiments/` as JSON files.

**Q: Can I customize the interface?**
A: Yes! The HTML, CSS, and JavaScript files are in the `web/templates/` and `web/static/` directories.

## Support

- **Documentation**: See [README.md](../README.md) and in-app documentation
- **Issues**: Report on [GitHub](https://github.com/Aimecol/ml/issues)
- **Questions**: Check the FAQ section above
- **Contributions**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

Happy experimenting! 🚀
