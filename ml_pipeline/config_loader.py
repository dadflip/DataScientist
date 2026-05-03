"""Chargement de la configuration TOML → dict Python."""

from __future__ import annotations

import pathlib


try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli
    except ImportError:
        tomllib = None  # type: ignore


def load_config(path: str | pathlib.Path = "default.toml") -> dict:
    """
    Charge le fichier TOML et retourne un dict.
    Fallback : si tomllib/tomli absent, installe tomli via pip.
    """
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if tomllib is None:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli", "-q"])
        import tomli as _tomllib  # type: ignore
    else:
        _tomllib = tomllib  # type: ignore

    with open(path, "rb") as f:
        return _tomllib.load(f)
