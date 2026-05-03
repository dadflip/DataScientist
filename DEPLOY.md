# Déployer ml_pipeline

## Installation sur Kaggle

```python
# Installation depuis GitHub
!pip install "git+https://github.com/dadflip/DataScientist.git[all]" -q

# Test
from ml_pipeline import load_config, PipelineState, styles
state = PipelineState()
state.config = load_config('ml_pipeline/default.toml')
print("ml_pipeline installé avec succès!")
```

## Installation locale

```bash
cd c:/Users/david/Documents/GitHub/DataScientist
pip install -e .
```

## Fichiers de déploiement

- `pyproject.toml` — Source de vérité (racine)
- `MANIFEST.in` — Inclusion des fichiers de données

## URL

GitHub : https://github.com/dadflip/DataScientist
