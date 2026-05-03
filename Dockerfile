FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
# gcc/g++ nécessaires pour compiler xgboost/lightgbm/catboost
# lors de l'installation dynamique via la cellule S00 du notebook
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposition du port Voilà
EXPOSE 8866

# Commande de démarrage
CMD ["voila", "notebook.ipynb", "--port=8866", "--ip=0.0.0.0", "--no-browser", "--enable_nbextensions=True", "--Voila.tornado_settings={\"allow_origin\":\"*\"}", "--VoilaConfiguration.file_whitelist=['.*']"]
