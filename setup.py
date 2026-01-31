"""Setup configuration for ML Project Framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-project-framework",
    version="1.0.0",
    author="ML Framework Team",
    description="A professional, production-ready Python framework for machine learning projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-project-framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "ipython>=7.0.0",
    ],
    extras_require={
        "xgboost": ["xgboost>=1.5.0"],
        "lightgbm": ["lightgbm>=3.3.0"],
        "mlflow": ["mlflow>=1.20.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9.0"],
    },
)
