"""PipelineState — objet partagé entre toutes les étapes."""
from __future__ import annotations
import os
import shutil


class PipelineState:
    """Source unique de vérité pour tout le pipeline."""

    def __init__(self):
        self.data_raw: dict         = {}
        self.data_cleaned: dict     = {}
        self.data_encoded: dict     = {}
        self.data_splits: dict      = {}
        self.data_types: dict       = {}
        self.meta: dict             = {}
        self.business_context: dict = {}
        self.models: dict           = {}
        self.trained_models: dict   = {}
        self.predictions: dict      = {}
        self.history: list          = []
        self.config: dict           = {}
        self.eda_dashboard: list    = []
        self.feature_columns: list  = []
        self.imbalance_config: dict = {}
        self.best_model             = None
        self.best_model_name: str   = ""
        self.final_predictions: dict = {}

    # ── logging ───────────────────────────────────────────────────────────────
    def log_step(self, step: str, action: str, details: dict) -> None:
        self.history.append({"step": step, "action": action, "details": details})

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Remet à zéro tout l'état (garde la même référence objet)."""
        self.__init__()

    # ── résumé ────────────────────────────────────────────────────────────────
    def summary(self) -> None:
        print("=== PipelineState Summary ===")
        print(f"  Raw datasets    : {list(self.data_raw.keys())}")
        print(f"  Cleaned         : {list(self.data_cleaned.keys())}")
        print(f"  Encoded         : {list(self.data_encoded.keys())}")
        print(f"  Splits          : {list(self.data_splits.keys())}")
        print(f"  Models          : {list(self.models.keys())}")
        print(f"  History steps   : {len(self.history)}")
        print(f"  Dashboard plots : {len(self.eda_dashboard)}")

    # ── nettoyage fichiers temp ───────────────────────────────────────────────
    def clean_temp_files(self, temp_dir: str = "temp") -> None:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        self.log_step("cleanup", "temp_files_removed", {"directory": temp_dir})
