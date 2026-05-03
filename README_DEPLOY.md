# Déploiement du ML Pipeline en Production

## Option 1 : Heroku (Recommandé - gratuit)

```bash
# 1. Installer Heroku CLI et se connecter
heroku login

# 2. Créer l'app
heroku create ml-pipeline-dashboard

# 3. Configurer le buildpack
heroku buildpacks:set heroku/python

# 4. Déployer
git add .
git commit -m "Setup Voilà deployment"
git push heroku main
```

L'app sera disponible sur `https://ml-pipeline-dashboard.herokuapp.com`

## Option 2 : Docker Local

```bash
# Construire l'image
docker build -t ml-pipeline .

# Lancer
docker run -p 8866:8866 ml-pipeline
```

Accès : http://localhost:8866

## Option 3 : Jupyter Server local

```bash
# Installer Voilà
pip install voila

# Lancer le dashboard
voila notebook.ipynb --port=8866 --enable_nbextensions=True
```

Accès : http://localhost:8866

## Option 4 : Hugging Face Spaces (Gratuit)

1. Créer un repo sur [Hugging Face Spaces](https://huggingface.co/spaces)
2. Choisir le SDK "Docker"
3. Pousser ce repo
4. Le dashboard est automatiquement déployé

## Configuration

- `Procfile` : Commande de démarrage Heroku
- `requirements.txt` : Dépendances Python
- `runtime.txt` : Version Python
- `voila.json` : Configuration Voilà

## Sécurité

Le notebook est exécuté côté serveur. Les utilisateurs ne voient que les widgets interactifs, pas le code source.

Pour restreindre l'accès, ajoutez une authentification :
```bash
voila notebook.ipynb --password=your_password
```
