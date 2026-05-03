import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt

try:
    from ml_pipeline.cell_0d_styles import styles
except ImportError:
    if 'styles' in globals():
        styles = globals()['styles']
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "styles", os.path.join(os.path.dirname(__file__), "cell_0d_styles.py"))
        styles = importlib.util.module_from_spec(spec)
        sys.modules["styles"] = styles
        spec.loader.exec_module(styles)


def dynamic_import(import_string):
    if not import_string:
        return None
    import importlib
    parts = import_string.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _is_inference_mode(splits):
    return splits.get("y_test") is None and splits.get("X_test") is not None


# ══════════════════════════════════════════════════════════════════════════════
# ALIGNEMENT DE COLONNES
# Problème : X_train peut avoir des colonnes créées par le Feature Engineering
# qui n'existent pas dans X_test/X_inference (et vice-versa).
# Solution : aligner X_test sur les colonnes de X_train (ajout NaN + suppression).
# ══════════════════════════════════════════════════════════════════════════════

def align_columns(X_ref: pd.DataFrame, X_target: pd.DataFrame,
                  fill_value=0) -> pd.DataFrame:
    """
    Aligne X_target sur les colonnes de X_ref :
      - Colonnes présentes dans X_ref mais absentes de X_target → ajoutées (fill_value)
      - Colonnes présentes dans X_target mais absentes de X_ref → supprimées
      - Ordre des colonnes identique à X_ref
    Retourne une copie alignée de X_target.
    """
    X_out = X_target.copy()

    ref_cols  = list(X_ref.columns)
    tgt_cols  = set(X_out.columns)

    # Colonnes manquantes dans X_target
    missing = [c for c in ref_cols if c not in tgt_cols]
    if missing:
        for c in missing:
            X_out[c] = fill_value

    # Colonnes en trop dans X_target
    extra = [c for c in X_out.columns if c not in ref_cols]
    if extra:
        X_out.drop(columns=extra, inplace=True)

    # Réordonner
    X_out = X_out[ref_cols]
    return X_out


def _align_report(X_ref, X_target, label="X_test") -> str:
    """Génère un rapport HTML des différences de colonnes."""
    ref_cols = set(X_ref.columns)
    tgt_cols = set(X_target.columns)
    missing  = ref_cols - tgt_cols
    extra    = tgt_cols - ref_cols
    lines = []
    if missing:
        lines.append(f"<b style='color:#f59e0b;'>{label} — {len(missing)} col(s) manquante(s) → remplies avec 0 :</b> "
                     + ", ".join(sorted(missing)))
    if extra:
        lines.append(f"<b style='color:#64748b;'>{label} — {len(extra)} col(s) en trop → supprimée(s) :</b> "
                     + ", ".join(sorted(extra)))
    if not lines:
        lines.append(f"<span style='color:#10b981;'>{label} — colonnes identiques à X_train.</span>")
    return "<br>".join(lines)


class ModelingUI:
    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg(
                "Configuration non chargée. Exécutez d'abord la cellule Config.")
            return
        self.config = state.config.get("modeling", {})
        self.splits = state.data_splits
        self._build_ui()

    # ── détection automatique de la tâche ────────────────────────────────────
    def _auto_detect(self):
        y_train = self.splits.get("y_train")
        prob_type = target_type = None

        if hasattr(self.state, "business_context") and self.state.business_context.get("domain"):
            dv = self.state.business_context["domain"]
            for d in self.state.config.get("domains", []):
                if d["value"] == dv:
                    prob_type  = d.get("task")
                    target_type = d.get("subtask")
                    break

        if not prob_type or not target_type:
            if y_train is not None:
                is_clf = y_train.nunique() <= 20
                prob_type   = "classification" if is_clf else "regression"
                target_type = ("binary" if y_train.nunique() == 2 else "multiclass") \
                               if is_clf else "continuous"
            else:
                prob_type, target_type = "classification", "binary"

        return prob_type, target_type

    def _refresh_catalog(self):
        self._model_checkboxes = {}
        task    = self.dd_task.value
        subtask = self.dd_subtask.value
        catalog = self.config.get(task, {}).get(subtask, []) if subtask \
                  else self.config.get(task, [])
        if isinstance(catalog, dict):
            catalog = []

        rows = []
        for m in catalog:
            if isinstance(m, dict) and "name" in m:
                cb = widgets.Checkbox(value=False, description=m["name"])
                self._model_checkboxes[m["name"]] = {"checkbox": cb, "config": m}
                rows.append(cb)

        self.chk_container.children = rows if rows else [
            widgets.HTML("<div style='color:#64748b;'>Aucun modèle pour cette config.</div>")]

    def _on_task_changed(self, change):
        if change.new:
            sub = list(self.config.get(change.new, {}).keys())
            self.dd_subtask.options  = sub
            self.dd_subtask.value    = sub[0] if sub else None
            self.dd_subtask.disabled = not bool(sub)
            self._refresh_catalog()

    def _on_subtask_changed(self, change):
        self._refresh_catalog()

    def _build_ui(self):
        if not self.splits:
            self.ui = styles.error_msg("Aucun split détecté. Exécutez l'étape Splitting d'abord.")
            return

        prob_type, target_type = self._auto_detect()
        inference_mode = _is_inference_mode(self.splits)

        avail_tasks   = list(self.config.keys())
        default_task  = prob_type if prob_type in avail_tasks else (avail_tasks[0] if avail_tasks else "")
        self.dd_task  = widgets.Dropdown(options=avail_tasks, value=default_task,
                                         description="Task:", style={'description_width': 'initial'})

        avail_sub    = list(self.config.get(default_task, {}).keys())
        default_sub  = target_type if target_type in avail_sub else (avail_sub[0] if avail_sub else None)
        self.dd_subtask = widgets.Dropdown(options=avail_sub, value=default_sub,
                                            description="Subtask:", disabled=not bool(avail_sub),
                                            style={'description_width': 'initial'})

        self.dd_task.observe(self._on_task_changed, names='value')
        self.dd_subtask.observe(self._on_subtask_changed, names='value')

        self.slider_val_split = widgets.FloatSlider(
            value=0.2, min=0.05, max=0.5, step=0.05,
            description="Val split:", readout_format='.0%',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='380px'))

        # Option de remplissage pour les colonnes manquantes
        self.fill_missing_dd = widgets.Dropdown(
            options=[("Remplir avec 0 (recommandé)", 0),
                     ("Remplir avec la médiane X_train", "median"),
                     ("Remplir avec la moyenne X_train",  "mean")],
            value=0,
            description="Colonnes manquantes :",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='420px'))

        self.chk_container = widgets.VBox([])
        self._refresh_catalog()

        self.btn_train  = widgets.Button(description="Entraîner les modèles",
                                          button_style=styles.BTN_PRIMARY)
        self.btn_train.on_click(self._train_models)
        self.output = widgets.Output()

        header  = widgets.HTML(styles.card_html("Model Training", "Train Machine Learning Models", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items='center', margin='0 0 12px 0',
            padding='0 0 10px 0', border_bottom='2px solid #ede9fe'))

        mode_banner = styles.help_box(
            "<b>Mode inférence :</b> y_test absent. Le train sera découpé en train/val "
            "pour l'évaluation. Les prédictions finales seront générées sur X_test."
            if inference_mode else
            "<b>Mode évaluation :</b> y_test présent. Évaluation directe sur le test set.",
            "#f59e0b" if inference_mode else "#10b981")

        align_info = styles.help_box(
            "<b>Alignement automatique des colonnes :</b> Si X_train contient des colonnes "
            "créées par le Feature Engineering absentes de X_test/X_inference, elles seront "
            "ajoutées automatiquement (valeur configurable ci-dessous). Les colonnes en trop "
            "dans X_test seront ignorées. Un rapport détaillé s'affiche à l'entraînement.",
            "#6366f1")

        split_row = widgets.HBox([self.slider_val_split],
                                  layout=widgets.Layout(
                                      margin='8px 0',
                                      display='flex' if inference_mode else 'none'))

        self.ui = widgets.VBox([
            top_bar,
            mode_banner,
            align_info,
            widgets.HBox([self.dd_task, self.dd_subtask],
                          layout=widgets.Layout(margin='10px 0', gap='20px')),
            split_row,
            self.fill_missing_dd,
            widgets.HTML("<hr style='border:1px solid #f1f5f9; margin:8px 0;'>"),
            widgets.HTML("<b>Algorithmes disponibles :</b>"),
            self.chk_container,
            widgets.HTML("<hr/>"),
            self.btn_train,
            self.output,
        ], layout=widgets.Layout(
            width='100%', max_width='1000px',
            border='1px solid #e5e7eb',
            padding='18px', border_radius='10px',
            background_color='#ffffff'))

    # ── helper : résolution de la valeur de remplissage ──────────────────────
    def _resolve_fill(self, X_train: pd.DataFrame):
        v = self.fill_missing_dd.value
        if v == "median":
            return X_train.median()   # Series col→valeur
        if v == "mean":
            return X_train.mean()
        return v  # 0 ou autre constante

    # ── alignement avec rapport ───────────────────────────────────────────────
    def _align_and_report(self, X_train, X_target, label, fill):
        report = _align_report(X_train, X_target, label)
        if isinstance(fill, pd.Series):
            # remplissage colonne par colonne
            X_out = X_target.copy()
            ref_cols = list(X_train.columns)
            for c in ref_cols:
                if c not in X_out.columns:
                    X_out[c] = fill.get(c, 0)
            extra = [c for c in X_out.columns if c not in ref_cols]
            if extra:
                X_out.drop(columns=extra, inplace=True)
            X_out = X_out[ref_cols]
        else:
            X_out = align_columns(X_train, X_target, fill_value=fill)
        return X_out, report

    # ── entraînement ─────────────────────────────────────────────────────────
    def _train_models(self, b):
        with self.output:
            clear_output(wait=True)
            selected = {n: info for n, info in self._model_checkboxes.items()
                        if info["checkbox"].value}
            if not selected:
                display(HTML("<div style='color:#ef4444;'>Sélectionnez au moins un modèle.</div>"))
                return

            inference_mode = _is_inference_mode(self.splits)
            target_col      = self.splits.get("target")
            target_pred_col = self.splits.get("target_pred_col", f"{target_col}_pred")

            # ── splits bruts ─────────────────────────────────────────────────
            if inference_mode:
                from sklearn.model_selection import train_test_split
                X_full = self.splits["X_train"]
                y_full = self.splits["y_train"]
                val_sz = self.slider_val_split.value
                task   = self.dd_task.value
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_full, y_full, test_size=val_sz, random_state=42,
                    stratify=y_full if task == "classification" else None)
                X_inference_raw = self.splits["X_test"]
                display(HTML(
                    f"<div style='color:#6366f1; font-size:0.85em;'>"
                    f"Mode inférence — train={len(X_tr):,}, val={len(X_val):,}, "
                    f"inference={len(X_inference_raw):,}</div>"))
            else:
                X_tr  = self.splits["X_train"]
                y_tr  = self.splits["y_train"]
                X_val = self.splits["X_test"]
                y_val = self.splits["y_test"]
                X_inference_raw = None
                display(HTML(
                    f"<div style='color:#10b981; font-size:0.85em;'>"
                    f"Mode évaluation — train={len(X_tr):,}, test={len(X_val):,}</div>"))

            # ── résolution du fill ────────────────────────────────────────────
            fill = self._resolve_fill(X_tr)

            # ── alignement X_val ─────────────────────────────────────────────
            X_val_aligned, report_val = self._align_and_report(X_tr, X_val, "X_val", fill)
            display(HTML(f"<div style='font-size:0.82em; margin:4px 0;'>{report_val}</div>"))

            # ── alignement X_inference ────────────────────────────────────────
            X_inference = None
            if X_inference_raw is not None:
                X_inference, report_inf = self._align_and_report(
                    X_tr, X_inference_raw, "X_inference", fill)
                display(HTML(f"<div style='font-size:0.82em; margin:4px 0;'>{report_inf}</div>"))

            # ── stockage de la référence colonnes pour les étapes suivantes ──
            self.state.feature_columns = list(X_tr.columns)

            # ── boucle d'entraînement ─────────────────────────────────────────
            import joblib
            models_dir = pathlib.Path("models")
            models_dir.mkdir(exist_ok=True)

            if not hasattr(self.state, 'predictions'):
                self.state.predictions = {}
            if not hasattr(self.state, 'trained_models'):
                self.state.trained_models = {}

            for name, info in selected.items():
                cfg        = info["config"]
                ModelClass = dynamic_import(f"{cfg['module']}.{cfg['class_name']}")
                if not ModelClass:
                    display(HTML(f"<div style='color:#ef4444;'>Erreur import : {name}</div>"))
                    continue

                display(HTML(f"<hr style='border:1px solid #f1f5f9;'>"
                              f"<b style='color:#1e293b;'>Entraînement : {name}</b>"))

                params = cfg.get("params", {}).copy()
                if hasattr(self.state, "imbalance_config") and target_col:
                    if self.state.imbalance_config.get(target_col, {}).get("method") == "class_weights":
                        try:
                            if hasattr(ModelClass(), 'class_weight'):
                                params['class_weight'] = 'balanced'
                                display(HTML("<span style='color:#6366f1; font-size:0.82em;'>"
                                             "class_weight='balanced' appliqué</span>"))
                        except Exception:
                            pass

                try:
                    model = ModelClass(**params)
                    model.fit(X_tr, y_tr)
                except Exception as e:
                    display(HTML(f"<div style='color:#ef4444; font-size:0.82em;'>"
                                  f"Erreur fit : {e}</div>"))
                    continue

                self.state.models[name] = model

                safe_name  = name.replace(' ', '_').replace('/', '_')
                model_path = models_dir / f"{safe_name}.pkl"
                joblib.dump(model, model_path)
                display(HTML(f"<span style='color:#64748b; font-size:0.8em;'>"
                              f"Sauvegardé → {model_path}</span>"))

                # ── évaluation sur val ────────────────────────────────────────
                from sklearn.metrics import (accuracy_score, r2_score,
                                             classification_report, confusion_matrix)
                try:
                    y_pred_val = model.predict(X_val_aligned)
                    if self.dd_task.value == "classification":
                        sc = accuracy_score(y_val, y_pred_val)
                        display(HTML(f"<div style='color:#10b981; font-weight:bold;'>"
                                      f"Accuracy (val) : {sc:.4f}</div>"))
                        display(HTML(f"<pre style='font-size:0.78em; background:#f8fafc; "
                                      f"padding:8px; border-radius:4px;'>"
                                      f"{classification_report(y_val, y_pred_val)}</pre>"))
                    else:
                        sc = r2_score(y_val, y_pred_val)
                        display(HTML(f"<div style='color:#10b981; font-weight:bold;'>"
                                      f"R² (val) : {sc:.4f}</div>"))
                except Exception as e:
                    display(HTML(f"<div style='color:#ef4444; font-size:0.82em;'>"
                                  f"Erreur predict val : {e}</div>"))
                    y_pred_val = None

                # ── stockage predictions ──────────────────────────────────────
                pred_entry = {
                    "model":          name,
                    "X_train":        X_tr,
                    "y_train":        y_tr,
                    "X_val":          X_val_aligned,
                    "y_val":          y_val,
                    "y_pred_val":     y_pred_val,
                    "target_col":     target_col,
                    "target_pred_col": target_pred_col,
                    "feature_columns": list(X_tr.columns),
                }

                if X_inference is not None:
                    try:
                        y_pred_inf = model.predict(X_inference)
                        X_inf_out  = X_inference.copy()
                        X_inf_out[target_pred_col] = y_pred_inf
                        pred_entry["X_inference_with_pred"] = X_inf_out
                        pred_entry["y_pred_inference"]       = y_pred_inf
                        display(HTML(
                            f"<div style='color:#6366f1; font-size:0.82em;'>"
                            f"Prédictions inférence : {len(y_pred_inf):,} lignes "
                            f"→ colonne '{target_pred_col}'</div>"))
                    except Exception as e:
                        display(HTML(f"<div style='color:#ef4444; font-size:0.82em;'>"
                                      f"Erreur predict inference : {e}</div>"))

                self.state.predictions[name] = pred_entry
                self.state.trained_models[name] = {
                    'model':          model,
                    'model_path':     str(model_path),
                    'params':         params,
                    'target_col':     target_col,
                    'target_pred_col': target_pred_col,
                    'inference_mode': inference_mode,
                    'feature_columns': list(X_tr.columns),
                }

            self.state.log_step("Modeling", "Models Trained", {
                "models":          list(selected.keys()),
                "target_col":      target_col,
                "inference_mode":  inference_mode,
                "feature_columns": self.state.feature_columns,
            })
            display(HTML("<div style='color:#10b981; font-weight:bold; margin-top:10px;'>"
                          "Entraînement termine.</div>"))


def runner(state):
    m = ModelingUI(state)
    display(m.ui)
    return m

try:
    runner(state)
except NameError:
    pass