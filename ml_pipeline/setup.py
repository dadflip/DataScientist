"""Setup pour installer ml_pipeline comme package depuis GitHub.

Usage sur Kaggle:
    !pip install git+https://github.com/dadflip/DataScientist.git#subdirectory=ml_pipeline

Ou en cellule Python:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "git+https://github.com/dadflip/DataScientist.git#subdirectory=ml_pipeline",
                          "-q"])
"""
from setuptools import setup, find_packages

setup(
    name="ml-pipeline",
    version="2.0.0",
    author="dadflip",
    description="ML Pipeline - Framework d'apprentissage automatique",
    long_description=open("../README.md", encoding="utf-8").read() if __import__('os').path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/dadflip/DataScientist",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["*.toml", "*.json", "*.yaml", "*.yml"],
    },
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "ipywidgets",
        "toml",
        "jinja2",
    ],
    extras_require={
        "boosting": ["xgboost", "lightgbm", "catboost"],
        "vision": ["opencv-python-headless", "Pillow"],
        "nlp": ["nltk", "spacy", "transformers"],
        "graph": ["networkx"],
        "all": [
            "xgboost", "lightgbm", "catboost",
            "category_encoders",
            "opencv-python-headless", "Pillow",
            "networkx",
            "rdflib", "owlready2",
            "pyarrow",
            "tqdm", "joblib",
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
