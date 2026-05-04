"""Étapes du pipeline ML — imports lazy pour éviter les erreurs au démarrage."""
from __future__ import annotations


def install_runner(config: dict):
    from .s00_install import runner
    return runner(config)


def loading_runner(state):
    from .s01_loading import runner
    return runner(state)


def domains_runner(state):
    from .s02_domains import runner
    return runner(state)


def eda_runner(state):
    from .s03_eda import runner
    return runner(state)


def feature_eng_runner(state):
    from .s04_feature_eng import runner
    return runner(state)


def ontology_feature_eng_runner(state):
    from .s04_ontology import runner
    return runner(state)


def cleaning_runner(state):
    from .s05_cleaning import runner
    return runner(state)


def encoding_runner(state):
    from .s06_encoding import runner
    return runner(state)


def split_runner(state):
    from .s07_split import runner
    return runner(state)


def modeling_runner(state):
    from .s08_modeling import runner
    return runner(state)


def metrics_runner(state):
    from .s09_metrics import runner
    return runner(state)


def optimization_runner(state):
    from .s10_optimization import runner
    return runner(state)


def predictions_runner(state):
    from .s11_predictions import runner
    return runner(state)


def export_runner(state):
    from .s12_export import runner
    return runner(state)


__all__ = [
    "install_runner", "loading_runner", "domains_runner", "eda_runner",
    "feature_eng_runner", "ontology_feature_eng_runner", "cleaning_runner", "encoding_runner",
    "split_runner", "modeling_runner", "metrics_runner", "optimization_runner",
    "predictions_runner", "export_runner",
]
