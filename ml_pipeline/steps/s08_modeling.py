"""Étape 8 — Modélisation (ModelingUI)."""
from __future__ import annotations
import pathlib
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles


def _dynamic_import(import_string: str):
    if not import_string:
        return None
    import importlib
    parts = import_string.split(".")
    module = importlib.import_module(".".join(parts[:-1]))
    return getattr(module, parts[-1])


def _is_inference_mode(splits: dict) -> bool:
    y_test = splits.get("y_test")
    X_test = splits.get("X_test")
    if y_test is None and X_test is not None:
        return True
    if isinstance(y_test, pd.Series) and y_test.empty:
        return True
    return False


def align_columns(X_ref: pd.DataFrame, X_target: pd.DataFrame, fill_value=0) -> pd.DataFrame:
    """Aligne X_target sur les colonnes de X_ref."""
    X_out = X_target.copy()
    ref_cols = list(X_ref.columns)
    tgt_set  = set(X_out.columns)
    for c in ref_cols:
        if c not in tgt_set:
            X_out[c] = fill_value
    extra = [c for c in X_out.columns if c not in ref_cols]
    if extra:
        X_out.drop(columns=extra, inplace=True)
    return X_out[ref_cols]


def _align_report(X_ref: pd.DataFrame, X_target: pd.DataFrame, label: str = "X_test") -> str:
    ref_cols = set(X_ref.columns); tgt_cols = set(X_target.columns)
    missing = ref_cols - tgt_cols; extra = tgt_cols - ref_cols
    lines = []
    if missing: lines.append(f"<b style='color:#f59e0b;'>{label} — {len(missing)} col(s) manquante(s) → remplies avec 0 :</b> " + ", ".join(sorted(missing)))
    if extra:   lines.append(f"<b style='color:#64748b;'>{label} — {len(extra)} col(s) en trop → supprimées :</b> " + ", ".join(sorted(extra)))
    if not lines: lines.append(f"<span style='color:#10b981;'>{label} — colonnes identiques à X_train.</span>")
    return "<br>".join(lines)


class ModelingUI:
    """Interface d'entraînement des modèles ML."""

    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("Configuration non chargée.")
            return
        self.config = state.config.get("modeling", {})
        self.splits = state.data_splits
        self._build_ui()

    def _auto_detect(self) -> tuple[str, str]:
        y_train = self.splits.get("y_train")
        prob_type = target_type = None
        if hasattr(self.state, "business_context") and self.state.business_context.get("domain"):
            dv = self.state.business_context["domain"]
            for d in self.state.config.get("domains", {}).get("supported", []):
                if d["value"] == dv:
                    prob_type  = d.get("task")
                    target_type = d.get("subtask")
                    break
        if not prob_type and y_train is not None:
            is_clf = y_train.nunique() <= 20
            prob_type   = "classification" if is_clf else "regression"
            target_type = ("binary" if y_train.nunique() == 2 else "multiclass") if is_clf else "continuous"
        return prob_type or "classification", target_type or "binary"

    def _refresh_catalog(self) -> None:
        self._model_checkboxes = {}
        task    = self.dd_task.value
        subtask = self.dd_subtask.value
        catalog = self.config.get(task, {}).get(subtask, []) if subtask else self.config.get(task, [])
        if isinstance(catalog, dict):
            catalog = []
        rows = []
        for m in catalog:
            if isinstance(m, dict) and "name" in m:
                cb = widgets.Checkbox(value=False, description=m["name"])
                self._model_checkboxes[m["name"]] = {"checkbox": cb, "config": m}
                rows.append(cb)
        self.chk_container.children = rows or [widgets.HTML("<div style='color:#64748b;'>Aucun modèle pour cette config.</div>")]

    def _on_task_changed(self, change) -> None:
        if change["new"]:
            sub = list(self.config.get(change["new"], {}).keys())
            self.dd_subtask.options  = sub
            self.dd_subtask.value    = sub[0] if sub else None
            self.dd_subtask.disabled = not bool(sub)
            self._refresh_catalog()

    def _on_subtask_changed(self, change) -> None:
        self._refresh_catalog()

    def _build_ui(self) -> None:
        if not self.splits:
            self.ui = styles.error_msg("Aucun split détecté. Exécutez l'étape Splitting d'abord.")
            return
        prob_type, target_type = self._auto_detect()
        inference_mode = _is_inference_mode(self.splits)
        avail_tasks  = list(self.config.keys())
        default_task = prob_type if prob_type in avail_tasks else (avail_tasks[0] if avail_tasks else "")
        self.dd_task = widgets.Dropdown(options=avail_tasks, value=default_task,
                                         description="Task:", style={"description_width": "initial"})
        avail_sub   = list(self.config.get(default_task, {}).keys())
        default_sub = target_type if target_type in avail_sub else (avail_sub[0] if avail_sub else None)
        self.dd_subtask = widgets.Dropdown(options=avail_sub, value=default_sub,
                                            description="Subtask:", disabled=not bool(avail_sub),
                                            style={"description_width": "initial"})
        self.dd_task.observe(self._on_task_changed, names="value")
        self.dd_subtask.observe(self._on_subtask_changed, names="value")
        self.slider_val_split = widgets.FloatSlider(
            value=0.2, min=0.05, max=0.5, step=0.05,
            description="Val split:", readout_format=".0%",
            style={"description_width": "initial"}, layout=widgets.Layout(width="380px"))
        self.fill_missing_dd = widgets.Dropdown(
            options=[("Remplir avec 0 (recommandé)", 0), ("Médiane X_train", "median"), ("Moyenne X_train", "mean")],
            value=0, description="Colonnes manquantes :", style={"description_width": "initial"},
            layout=widgets.Layout(width="420px"))
        self.chk_container = widgets.VBox([])
        self._refresh_catalog()
        self.btn_train = widgets.Button(description="Entraîner les modèles", button_style=styles.BTN_PRIMARY)
        self.btn_train.on_click(self._train_models)
        self.output = widgets.Output()
        header  = widgets.HTML(styles.card_html("Model Training", "Train Machine Learning Models", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        mode_banner = styles.help_box(
            "<b>Mode inférence :</b> y_test absent. Train découpé en train/val pour l'évaluation."
            if inference_mode else
            "<b>Mode évaluation :</b> y_test présent. Évaluation directe sur le test set.",
            "#f59e0b" if inference_mode else "#10b981")
        self.ui = widgets.VBox([
            top_bar, mode_banner,
            widgets.HBox([self.dd_task, self.dd_subtask], layout=widgets.Layout(margin="10px 0", gap="20px")),
            widgets.HBox([self.slider_val_split], layout=widgets.Layout(display="flex" if inference_mode else "none")),
            self.fill_missing_dd,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HTML("<b>Algorithmes disponibles :</b>"),
            self.chk_container,
            widgets.HTML("<hr/>"),
            self.btn_train, self.output,
        ], layout=widgets.Layout(width="100%", max_width="1000px",
                                  border="1px solid #e5e7eb", padding="18px",
                                  border_radius="10px", background_color="#ffffff"))

    def _resolve_fill(self, X_train: pd.DataFrame):
        v = self.fill_missing_dd.value
        if v == "median": return X_train.median()
        if v == "mean":   return X_train.mean()
        return v

    def _align_and_report(self, X_train, X_target, label, fill):
        report = _align_report(X_train, X_target, label)
        if isinstance(fill, pd.Series):
            X_out = X_target.copy()
            for c in X_train.columns:
                if c not in X_out.columns:
                    X_out[c] = fill.get(c, 0)
            extra = [c for c in X_out.columns if c not in X_train.columns]
            if extra: X_out.drop(columns=extra, inplace=True)
            X_out = X_out[list(X_train.columns)]
        else:
            X_out = align_columns(X_train, X_target, fill_value=fill)
        return X_out, report

    def _train_models(self, b) -> None:
        with self.output:
            clear_output(wait=True)
            selected = {n: info for n, info in self._model_checkboxes.items() if info["checkbox"].value}
            if not selected:
                display(HTML("<div style='color:#ef4444;'>Sélectionnez au moins un modèle.</div>")); return
            inference_mode = _is_inference_mode(self.splits)
            target_col     = self.splits.get("target")
            target_pred_col = self.splits.get("target_pred_col", f"{target_col}_pred")
            if inference_mode:
                from sklearn.model_selection import train_test_split
                X_full = self.splits["X_train"]; y_full = self.splits["y_train"]
                val_sz = self.slider_val_split.value
                X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=val_sz, random_state=42,
                    stratify=y_full if self.dd_task.value == "classification" else None)
                X_inference_raw = self.splits["X_test"]
                display(HTML(f"<div style='color:#6366f1;font-size:0.85em;'>Mode inférence — train={len(X_tr):,}, val={len(X_val):,}, inference={len(X_inference_raw):,}</div>"))
            else:
                X_tr = self.splits["X_train"]; y_tr = self.splits["y_train"]
                X_val = self.splits["X_test"]; y_val = self.splits["y_test"]
                X_inference_raw = None
                display(HTML(f"<div style='color:#10b981;font-size:0.85em;'>Mode évaluation — train={len(X_tr):,}, test={len(X_val):,}</div>"))
            fill = self._resolve_fill(X_tr)
            X_val_aligned, report_val = self._align_and_report(X_tr, X_val, "X_val", fill)
            display(HTML(f"<div style='font-size:0.82em;margin:4px 0;'>{report_val}</div>"))
            X_inference = None
            if X_inference_raw is not None:
                X_inference, report_inf = self._align_and_report(X_tr, X_inference_raw, "X_inference", fill)
                display(HTML(f"<div style='font-size:0.82em;margin:4px 0;'>{report_inf}</div>"))
            self.state.feature_columns = list(X_tr.columns)
            import joblib
            models_dir = pathlib.Path("models"); models_dir.mkdir(exist_ok=True)
            if not hasattr(self.state, "predictions"):   self.state.predictions = {}
            if not hasattr(self.state, "trained_models"): self.state.trained_models = {}
            for name, info in selected.items():
                cfg = info["config"]
                ModelClass = _dynamic_import(f"{cfg['module']}.{cfg['class_name']}")
                if not ModelClass:
                    display(HTML(f"<div style='color:#ef4444;'>Erreur import : {name}</div>")); continue
                display(HTML(f"<hr style='border:1px solid #f1f5f9;'><b style='color:#1e293b;'>Entraînement : {name}</b>"))
                params = cfg.get("params", {}).copy()
                if hasattr(self.state, "imbalance_config") and target_col:
                    if self.state.imbalance_config.get(target_col, {}).get("method") == "class_weights":
                        try:
                            if hasattr(ModelClass(), "class_weight"):
                                params["class_weight"] = "balanced"
                        except Exception: pass
                try:
                    model = ModelClass(**params); model.fit(X_tr, y_tr)
                except Exception as e:
                    display(HTML(f"<div style='color:#ef4444;font-size:0.82em;'>Erreur fit : {e}</div>")); continue
                self.state.models[name] = model
                safe_name  = name.replace(" ", "_").replace("/", "_")
                model_path = models_dir / f"{safe_name}.pkl"
                joblib.dump(model, model_path)
                display(HTML(f"<span style='color:#64748b;font-size:0.8em;'>Sauvegardé → {model_path}</span>"))
                from sklearn.metrics import accuracy_score, r2_score, classification_report
                try:
                    y_pred_val = model.predict(X_val_aligned)
                    if self.dd_task.value == "classification":
                        sc = accuracy_score(y_val, y_pred_val)
                        display(HTML(f"<div style='color:#10b981;font-weight:bold;'>Accuracy (val) : {sc:.4f}</div>"))
                        display(HTML(f"<pre style='font-size:0.78em;background:#f8fafc;padding:8px;border-radius:4px;'>{classification_report(y_val, y_pred_val)}</pre>"))
                    else:
                        sc = r2_score(y_val, y_pred_val)
                        display(HTML(f"<div style='color:#10b981;font-weight:bold;'>R² (val) : {sc:.4f}</div>"))
                except Exception as e:
                    display(HTML(f"<div style='color:#ef4444;font-size:0.82em;'>Erreur predict val : {e}</div>"))
                    y_pred_val = None
                pred_entry = {"model": name, "X_train": X_tr, "y_train": y_tr,
                               "X_val": X_val_aligned, "y_val": y_val, "y_pred_val": y_pred_val,
                               "target_col": target_col, "target_pred_col": target_pred_col,
                               "feature_columns": list(X_tr.columns)}
                if X_inference is not None:
                    try:
                        y_pred_inf = model.predict(X_inference)
                        X_inf_out  = X_inference.copy(); X_inf_out[target_pred_col] = y_pred_inf
                        pred_entry["X_inference_with_pred"] = X_inf_out
                        pred_entry["y_pred_inference"]       = y_pred_inf
                        display(HTML(f"<div style='color:#6366f1;font-size:0.82em;'>Prédictions inférence : {len(y_pred_inf):,} lignes → '{target_pred_col}'</div>"))
                    except Exception as e:
                        display(HTML(f"<div style='color:#ef4444;font-size:0.82em;'>Erreur predict inference : {e}</div>"))
                self.state.predictions[name] = pred_entry
                self.state.trained_models[name] = {"model": model, "model_path": str(model_path),
                    "params": params, "target_col": target_col, "target_pred_col": target_pred_col,
                    "inference_mode": inference_mode, "feature_columns": list(X_tr.columns)}
            self.state.log_step("Modeling", "Models Trained",
                                 {"models": list(selected.keys()), "target_col": target_col,
                                  "inference_mode": inference_mode, "feature_columns": self.state.feature_columns})
            display(HTML("<div style='color:#10b981;font-weight:bold;margin-top:10px;'>Entraînement terminé.</div>"))


def runner(state) -> ModelingUI:
    m = ModelingUI(state)
    if hasattr(m, "ui"):
        display(m.ui)
    return m
