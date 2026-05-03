"""Setup pour installer tout le repo DataScientist.

Usage sur Kaggle:
    !pip install git+https://github.com/dadflip/DataScientist.git

Ou en cellule Python:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "git+https://github.com/dadflip/DataScientist.git",
                          "-q"])

Après installation:
    from ml_pipeline import load_config, PipelineState, styles
"""
from setuptools import setup, find_packages
import os

# Lire le README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="datascientist",
    version="2.0.0",
    author="dadflip",
    description="DataScientist - ML Pipeline avec dashboard Voilà",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dadflip/DataScientist",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ml_pipeline": ["*.toml", "*.json", "*.yaml", "*.yml", "default.toml"],
    },
    data_files=[
        ("config", ["config/default.toml"]),
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "ipywidgets",
        "voila",
        "jupyter",
        "toml",
        "jinja2",
    ],
    extras_require={
        "prod": ["voila", "jupyter"],
        "boosting": ["xgboost", "lightgbm", "catboost"],
        "vision": ["opencv-python-headless", "Pillow"],
        "nlp": ["nltk", "spacy", "transformers"],
        "graph": ["networkx", "rdflib", "owlready2"],
        "all": [
            "xgboost", "lightgbm", "catboost",
            "category_encoders",
            "opencv-python-headless", "Pillow",
            "networkx", "rdflib", "owlready2",
            "pyarrow", "tqdm", "joblib",
            "voila", "jupyter",
        ],
    },
    entry_points={
        "console_scripts": [
            "datascientist=ml_pipeline.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
