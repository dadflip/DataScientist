# Déployer ml_pipeline

## Installation sur Kaggle

```python
# Installation depuis GitHub (méthode racine - recommandée)
!pip install "git+https://github.com/dadflip/DataScientist.git[all]" -q

# Test
from ml_pipeline import load_config, PipelineState, styles
print("✅ ml_pipeline installé avec succès!")
```

## Installation locale

```bash
cd c:/Users/david/Documents/GitHub/DataScientist
pip install -e .
```

## Fichiers de déploiement

- `setup.py` — Package racine (méthode utilisée)
- `ml_pipeline/setup.py` — Package sous-dossier (non utilisé)
- `ml_pipeline/pyproject.toml` — Alternative (non utilisé)

## URL

GitHub : https://github.com/dadflip/DataScientist
