import sys
import os
from pathlib import Path

# Ajout du répertoire racine au chemin de recherche des modules
_repo_root = str(Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Installation de ipywidgets (uniquement dans un environnement Jupyter)
def install_ipywidgets():
    try:
        # Vérifie si le script est exécuté dans un notebook Jupyter
        if "IPython" in sys.modules:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                print("Installing ipywidgets...", end=" ", flush=True)
                ipython.run_line_magic("pip", "install ipywidgets>=8.0 --quiet")
                print("Done. You can now run 00_install.py.")
            else:
                print("Not running in an IPython environment. Skipping ipywidgets installation.")
        else:
            print("Not running in an IPython environment. Skipping ipywidgets installation.")
    except Exception as e:
        print(f"Error while installing ipywidgets: {e}")

# Exécuter l'installation
install_ipywidgets()