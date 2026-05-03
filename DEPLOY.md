# Déployer ml_pipeline

## 1. Pousser sur GitHub (déjà fait ?)

```bash
cd c:/Users/david/Documents/GitHub/DataScientist
git add .
git commit -m "Add pip install support"
git push origin main
```

## 2. Tester sur Kaggle immédiatement

Colle cette cellule dans un notebook Kaggle :

```python
# Installation depuis GitHub
!pip install "git+https://github.com/dadflip/DataScientist.git#subdirectory=ml_pipeline[all]" -q

# Test
from ml_pipeline import load_config, PipelineState, styles
print("ml_pipeline installé avec succès!")
```

## 3. Publier sur PyPI (optionnel mais recommandé)

### Prérequis
```bash
pip install build twine
```

### Build et upload
```bash
cd c:/Users/david/Documents/GitHub/DataScientist/ml_pipeline
python -m build
python -m twine upload dist/*
```

### Utilisation après PyPI
```python
!pip install ml-pipeline
```

## Fichiers créés

- `ml_pipeline/setup.py` — Package pour pip install
- `setup.py` — Package racine (optionnel)
- `DEPLOY.md` — Ce fichier

## URL

GitHub : https://github.com/dadflip/DataScientist
