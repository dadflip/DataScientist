# Scalabilité - Calculs Lourds en Production

## Problème
Les étapes S08 (Modeling), S10 (Optimisation), S03 (EDA gros datasets) peuvent être très gourmandes en CPU/RAM.

## Solutions par niveau de charge

### Niveau 1 : Container scaling (Docker/Heroku)

**Docker Compose avec resources limites :**

```yaml
version: '3.8'
services:
  voila:
    build: .
    ports:
      - "8866:8866"
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    environment:
      - OMP_NUM_THREADS=4
      - MKL_NUM_THREADS=4
```

**Heroku (dynos plus puissants) :**
```bash
# Standard (2x) = 1GB RAM
# Performance (M) = 2.5GB RAM
# Performance (L) = 14GB RAM
heroku ps:type performance-l
```

---

### Niveau 2 : Async Processing (Celery + Redis)

**Architecture :**
```
[User] → [Voilà Dashboard] → [Redis Queue] → [Celery Worker]
                                           ↓
                                    [Heavy ML Task]
                                           ↓
                                    [Résultat en BDD]
```

**Avantage** : Le dashboard reste réactif, calculs en background.

**Implémentation :**
1. Extraire les steps lourdes (`s08_modeling.py`, `s10_optimization.py`)
2. Créer une API FastAPI + Celery
3. Le dashboard poll le statut des jobs

---

### Niveau 3 : Cloud GPU/CPU (AWS/GCP/Azure)

**AWS SageMaker :**
- Entraînement sur instances `ml.m5.4xlarge` (16 vCPU, 64GB)
- Ou GPU `ml.g4dn.xlarge` pour deep learning

**Architecture hybride :**
```
Voilà Dashboard (light) → API Gateway → SageMaker Training
                                   ↓
                            S3 (modèles/résultats)
```

**Azure ML / GCP Vertex AI** : Équivalent

---

### Niveau 4 : Ray / Dask (Distributed Computing)

**Pour datasets > 10GB :**

```python
# Dans le notebook, remplacer pandas par Dask
import dask.dataframe as dd
df = dd.read_csv('huge_file.csv')

# Ou utiliser Ray pour paralléliser sklearn
from ray import train
from sklearn.ensemble import RandomForestClassifier

@train.remote
def train_model(data):
    return RandomForestClassifier().fit(data)
```

**Cluster Ray auto-scaling :**
```yaml
# cluster.yaml
min_workers: 2
max_workers: 10
head_node_type:
  resources: {"CPU": 8, "memory": 32000000000}
worker_node_types:
  - resources: {"CPU": 4, "memory": 16000000000}
```

---

## Recommandation par use case

| Use Case | Solution | Coût estimé |
|----------|----------|-------------|
| Prototype / démo | Voilà simple + Docker local | Gratuit |
| Petit équipe (< 10 users) | Heroku Performance-L | ~250€/mois |
| Production avec file d'attente | Voilà + Celery + Redis | ~100-500€/mois |
| Big Data (> 100GB) | Ray Cluster / Dask | Variable |
| ML Enterprise | AWS SageMaker + API | ~500-2000€/mois |

---

## Implémentation rapide : Celery Jobs

**1. Créer `tasks.py` :**
```python
from celery import Celery
import joblib

app = Celery('ml_pipeline', broker='redis://localhost:6379')

@app.task
def run_modeling(config, data_path):
    # Exécution lourde ici
    model = train_heavy_model(config, data_path)
    joblib.dump(model, 'output/model.pkl')
    return {'status': 'done', 'path': 'output/model.pkl'}
```

**2. Modifier le notebook pour lancer des jobs :**
```python
from tasks import run_modeling

# Au lieu de exécuter directement :
# s08.run_modeling()

# Lancer en background :
job = run_modeling.delay(state.config, 'data/train.csv')
state.current_job_id = job.id
```

**3. Polling dans le dashboard :**
```python
def check_status():
    result = run_modeling.AsyncResult(state.current_job_id)
    return result.status  # PENDING / SUCCESS / FAILURE
```

---

## Monitoring

**Prometheus + Grafana** pour surveiller :
- CPU/RAM usage
- Temps d'exécution des jobs
- Queue depth (Celery)

```python
# Dans le Dockerfile
pip install prometheus-client
```

## Prochaines étapes recommandées

1. **Commencer simple** : Docker local avec 4-8GB RAM
2. **Si besoin** : Ajouter Celery pour les jobs longs
3. **Si big data** : Migrer vers Ray/Dask
4. **Si production critique** : AWS SageMaker ou équivalent
