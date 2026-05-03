"""Étape 11 — Prédictions & Export (PredictionsUI)."""
from __future__ import annotations
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from ml_pipeline.styles import styles

_PAL  = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#ec4899"]
_GRAY = "#64748b"; _BG = "#f8fafc"; _GRID = "#e2e8f0"


def _is_valid(arr) -> bool:
    if arr is None: return False
    if isinstance(arr, pd.DataFrame) and arr.empty: return False
    if isinstance(arr, pd.Series)    and arr.empty: return False
    if isinstance(arr, np.ndarray)   and arr.size == 0: return False
    return True


def _is_inference_mode(splits: dict) -> bool:
    return not _is_valid(splits.get("y_test")) and _is_valid(splits.get("X_test"))


def _section(title, color="#6366f1") -> widgets.HTML:
    return widgets.HTML(f"<div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;'>"
                        f"<div style='width:4px;height:20px;background:{color};border-radius:2px;'></div>"
                        f"<span style='font-size:0.95em;font-weight:700;color:#1e293b;'>{title}</span></div>")


def _warn(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#92400e;background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")


def _info(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#1e40af;background:#eff6ff;border-left:4px solid #3b82f6;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")


def _success(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#065f46;background:#ecfdf5;border-left:4px solid #10b981;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")


def align_columns(X_ref: pd.DataFrame, X_target: pd.DataFrame, fill_value=0) -> tuple:
    X_out = X_target.copy(); ref_cols = list(X_ref.columns); tgt_set = set(X_out.columns); ref_set = set(ref_cols)
    missing = ref_set - tgt_set; extra = tgt_set - ref_set
    for c in missing:
        fill = fill_value[c] if isinstance(fill_value, pd.Series) and c in fill_value else (fill_value if not isinstance(fill_value, pd.Series) else 0)
        X_out[c] = fill
    if extra: X_out.drop(columns=list(extra), inplace=True)
    X_out = X_out[ref_cols]
    parts = []
    if missing: parts.append(f"<b style='color:#f59e0b;'>{len(missing)} col(s) manquantes → remplies :</b> {', '.join(sorted(missing))}")
    if extra:   parts.append(f"<b style='color:#64748b;'>{len(extra)} col(s) en trop → supprimées :</b> {', '.join(sorted(extra))}")
    if not parts: parts.append("<span style='color:#10b981;'>Colonnes identiques à X_train.</span>")
    return X_out, "<br>".join(parts)


def _resolve_feature_columns(state, model_name):
    pred = getattr(state, "predictions", {}).get(model_name, {})
    if pred.get("feature_columns"): return pred["feature_columns"]
    tm = getattr(state, "trained_models", {}).get(model_name, {})
    if tm.get("feature_columns"): return tm["feature_columns"]
    if hasattr(state, "feature_columns") and state.feature_columns: return state.feature_columns
    X_tr = getattr(state, "data_splits", {}).get("X_train")
    if _is_valid(X_tr) and hasattr(X_tr, "columns"): return list(X_tr.columns)
    return None


class PredictionsUI:
    """Interface de génération et export des prédictions finales."""

    def __init__(self, state):
        self.state       = state
        self.splits      = getattr(state, "data_splits", {})
        self.models      = getattr(state, "models", {})
        self.predictions = getattr(state, "predictions", {})
        self.config      = getattr(state, "config", {})
        if not self.models:
            self.ui = styles.error_msg("Aucun modèle trouvé. Exécutez d'abord la cellule Modeling."); return
        self._build_ui()

    def _detect_task(self) -> tuple[str, str]:
        y = self.splits.get("y_train"); prob, sub = "classification", "binary"
        if _is_valid(y):
            n = y.nunique() if hasattr(y, "nunique") else len(np.unique(y))
            if n > 20: prob, sub = "regression", "continuous"
            elif n > 2: prob, sub = "classification", "multiclass"
        return prob, sub

    def _best_model_name(self) -> str:
        if hasattr(self.state, "best_model_name") and self.state.best_model_name in self.models:
            return self.state.best_model_name
        return list(self.models.keys())[-1]

    def _build_ui(self) -> None:
        self.task, self.subtask = self._detect_task()
        inference = _is_inference_mode(self.splits); best_name = self._best_model_name()
        header  = widgets.HTML(styles.card_html("Prédictions & Export", "Génération finale des prédictions — Export CSV", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(align_items="center", margin="0 0 12px 0", padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.dd_model = widgets.Dropdown(options=list(self.models.keys()), value=best_name, description="Modèle :", style={"description_width": "initial"}, layout=widgets.Layout(width="380px"))
        best_badge = widgets.HTML(f"<span style='background:#d1fae5;color:#065f46;border-radius:4px;padding:4px 10px;font-size:0.8em;font-weight:700;'>Best : {best_name}</span>")
        ds_options = [("X_test (split courant)", "__X_test__")]
        for k, v in getattr(self.state, "data_raw", {}).items():
            if isinstance(v, pd.DataFrame) and len(v) > 0:
                ds_options.append((f"data_raw['{k}']  ({len(v):,} lignes)", f"raw::{k}"))
        self.dd_source = widgets.Dropdown(options=ds_options, value="__X_test__", description="Source données :", style={"description_width": "initial"}, layout=widgets.Layout(width="420px"))
        self.dd_fill = widgets.Dropdown(options=[("0 (recommandé)", "zero"), ("Médiane X_train", "median"), ("Moyenne X_train", "mean")], value="zero", description="Fill colonnes manquantes :", style={"description_width": "initial"}, layout=widgets.Layout(width="380px"))
        self.txt_id_col = widgets.Text(value="id", description="Colonne ID (opt.) :", placeholder="laisser vide si inexistante", style={"description_width": "initial"}, layout=widgets.Layout(width="300px"))
        self.txt_pred_col = widgets.Text(value=self.splits.get("target_pred_col", f"{self.splits.get('target','target')}_pred"), description="Nom colonne prédite :", style={"description_width": "initial"}, layout=widgets.Layout(width="300px"))
        self.txt_output_file = widgets.Text(value="submission.csv", description="Nom fichier export :", style={"description_width": "initial"}, layout=widgets.Layout(width="300px"))
        self.chk_proba = widgets.Checkbox(value=(self.task == "classification"), description="Exporter aussi les probabilités (si classif.)", layout=widgets.Layout(width="380px"))
        self.chk_include_features = widgets.Checkbox(value=False, description="Inclure toutes les features dans l'export", layout=widgets.Layout(width="380px"))
        self.slider_threshold = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01, description="Seuil décision :", readout_format=".2f", style={"description_width": "initial"}, layout=widgets.Layout(width="360px", display="flex" if self.task == "classification" else "none"))
        self.btn_preview = widgets.Button(description="Aperçu (50 lignes)", button_style="info", layout=widgets.Layout(width="180px"))
        self.btn_predict = widgets.Button(description="Générer & Exporter", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="200px"))
        self.btn_preview.on_click(self._run_preview); self.btn_predict.on_click(self._run_predict)
        self.output = widgets.Output()
        mode_banner = styles.help_box(
            "<b>Mode inférence :</b> y_test absent — prédictions générées sur X_test et exportées directement." if inference
            else "<b>Mode évaluation :</b> y_test disponible — les prédictions seront comparées au ground truth.",
            "#f59e0b" if inference else "#10b981")
        self.ui = widgets.VBox([
            top_bar, mode_banner,
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Modèle</b>"),
            widgets.HBox([self.dd_model, best_badge], layout=widgets.Layout(gap="12px", align_items="center")),
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Source des données</b>"),
            self.dd_source, self.dd_fill,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Options d'export</b>"),
            widgets.HBox([self.txt_id_col, self.txt_pred_col, self.txt_output_file], layout=widgets.Layout(gap="12px", flex_wrap="wrap")),
            self.slider_threshold, self.chk_proba, self.chk_include_features,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HBox([self.btn_preview, self.btn_predict], layout=widgets.Layout(gap="12px")),
            self.output,
        ], layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb", padding="18px", border_radius="10px", background_color="#ffffff"))

    def _resolve_X(self):
        src = self.dd_source.value
        if src == "__X_test__":
            X_raw = self.splits.get("X_test"); y_true = self.splits.get("y_test")
        elif src.startswith("raw::"):
            X_raw = self.state.data_raw.get(src[len("raw::"):])
            y_true = None
        else:
            X_raw = None; y_true = None
        if not _is_valid(X_raw): return None, None, None
        id_col = self.txt_id_col.value.strip(); id_series = None
        if id_col and isinstance(X_raw, pd.DataFrame) and id_col in X_raw.columns:
            id_series = X_raw[id_col].copy(); X_raw = X_raw.drop(columns=[id_col])
        return X_raw, y_true, id_series

    def _resolve_fill_value(self, X_train):
        mode = self.dd_fill.value
        if mode == "median": return X_train.median()
        if mode == "mean":   return X_train.mean()
        return 0

    def _get_X_train_ref(self, model_name):
        pred = self.predictions.get(model_name, {})
        return pred.get("X_train", self.splits.get("X_train"))

    def _predict(self, model, X_aligned, threshold=0.5):
        y_proba = None
        if self.task == "classification" and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_aligned)
            y_pred = (y_proba[:, 1] >= threshold).astype(int) if y_proba.shape[1] == 2 else np.argmax(y_proba, axis=1)
        else:
            y_pred = model.predict(X_aligned)
        return y_pred, y_proba

    def _run_preview(self, b) -> None:
        with self.output:
            clear_output(wait=True)
            model_name = self.dd_model.value; model = self.models.get(model_name)
            if model is None: display(_warn(f"Modèle '{model_name}' introuvable.")); return
            X_raw, y_true, id_series = self._resolve_X()
            if not _is_valid(X_raw): display(_warn("Source de données invalide.")); return
            X_tr = self._get_X_train_ref(model_name); feat = _resolve_feature_columns(self.state, model_name)
            if not _is_valid(X_tr) or feat is None: display(_warn("Impossible de retrouver les colonnes d'entraînement.")); return
            X_ref = X_tr[feat] if hasattr(X_tr, "__getitem__") else X_tr
            fill = self._resolve_fill_value(X_ref); X_aligned, report = align_columns(X_ref, X_raw, fill)
            display(HTML(f"<div style='font-size:0.82em;margin:4px 0;'>{report}</div>"))
            display(_section(f"Aperçu (50 lignes) — {model_name}", "#6366f1"))
            threshold = self.slider_threshold.value; y_pred, y_proba = self._predict(model, X_aligned.head(50), threshold)
            preview_df = X_aligned.head(50).copy() if self.chk_include_features.value else pd.DataFrame(index=X_aligned.head(50).index)
            pred_col = self.txt_pred_col.value or "y_pred"; preview_df[pred_col] = y_pred
            if y_proba is not None and self.chk_proba.value:
                classes = getattr(model, "classes_", np.arange(y_proba.shape[1]))
                for i, c in enumerate(classes): preview_df[f"proba_{c}"] = y_proba[:50, i]
            if id_series is not None: preview_df.insert(0, self.txt_id_col.value, id_series.values[:50])
            if y_true is not None and _is_valid(y_true):
                y_true_arr = np.array(y_true)[:50]; preview_df["y_true"] = y_true_arr; preview_df["correct"] = (y_pred == y_true_arr)
            display(preview_df.style.set_table_styles([{"selector":"thead th","props":[("background-color","#f1f5f9"),("font-size","0.82em"),("padding","6px 10px")]},{"selector":"td","props":[("font-size","0.82em"),("padding","4px 10px")]}]).format(precision=4, na_rep="—"))

    def _run_predict(self, b) -> None:
        with self.output:
            clear_output(wait=True)
            model_name = self.dd_model.value; model = self.models.get(model_name)
            if model is None: display(_warn(f"Modèle '{model_name}' introuvable.")); return
            X_raw, y_true, id_series = self._resolve_X()
            if not _is_valid(X_raw): display(_warn("Source de données invalide ou vide.")); return
            X_tr = self._get_X_train_ref(model_name); feat = _resolve_feature_columns(self.state, model_name)
            if not _is_valid(X_tr) or feat is None: display(_warn("Impossible de retrouver les colonnes d'entraînement.")); return
            X_ref = X_tr[feat] if hasattr(X_tr, "__getitem__") else X_tr
            fill = self._resolve_fill_value(X_ref); X_aligned, report = align_columns(X_ref, X_raw, fill)
            display(_section(f"Génération des prédictions — {model_name}", "#6366f1"))
            display(HTML(f"<div style='font-size:0.82em;margin:4px 0;'>{report}</div>"))
            display(_info(f"Prédiction sur <b>{len(X_aligned):,}</b> lignes…"))
            threshold = self.slider_threshold.value
            try: y_pred, y_proba = self._predict(model, X_aligned, threshold)
            except Exception as e: display(_warn(f"Erreur lors de la prédiction : {e}")); return
            out_df = X_aligned.copy() if self.chk_include_features.value else pd.DataFrame(index=X_aligned.index)
            pred_col = self.txt_pred_col.value or "y_pred"; out_df[pred_col] = y_pred
            if y_proba is not None and self.chk_proba.value:
                classes = getattr(model, "classes_", np.arange(y_proba.shape[1]))
                for i, c in enumerate(classes): out_df[f"proba_{c}"] = y_proba[:, i]
            if id_series is not None: out_df.insert(0, self.txt_id_col.value, id_series.values)
            if _is_valid(y_true): self._show_eval_metrics(model, X_aligned, y_true, y_pred, y_proba)
            fname = self.txt_output_file.value.strip() or "submission.csv"
            if not fname.endswith(".csv"): fname += ".csv"
            out_df.to_csv(fname, index=False)
            display(_success(f"<b>Export terminé !</b><br>Fichier : <b>{fname}</b> | {len(out_df):,} lignes | {len(out_df.columns)} colonnes"))
            self.state.final_predictions = {"model_name": model_name, "y_pred": y_pred, "y_proba": y_proba, "output_df": out_df, "output_file": fname, "n_rows": len(out_df), "feature_cols": feat}
            self.state.log_step("Predictions", "Final Predictions Generated", {"model": model_name, "n_rows": len(out_df), "output_file": fname})
            display(_section("Aperçu du fichier exporté (10 premières lignes)", "#3b82f6"))
            display(out_df.head(10).style.set_table_styles([{"selector":"thead th","props":[("background-color","#f1f5f9"),("font-size","0.82em"),("padding","6px 10px")]},{"selector":"td","props":[("font-size","0.82em"),("padding","4px 10px")]}]).format(precision=4, na_rep="—"))

    def _show_eval_metrics(self, model, X_aligned, y_true, y_pred, y_proba) -> None:
        from sklearn import metrics as skm
        display(_section("Métriques de validation (y_true disponible)", "#8b5cf6"))
        cards_html = ""
        if self.task == "classification":
            met_list = [("Accuracy", skm.accuracy_score(y_true, y_pred), _PAL[1]),
                        ("F1 (macro)", skm.f1_score(y_true, y_pred, average="macro", zero_division=0), _PAL[0]),
                        ("Precision", skm.precision_score(y_true, y_pred, average="macro", zero_division=0), _PAL[2]),
                        ("Recall", skm.recall_score(y_true, y_pred, average="macro", zero_division=0), _PAL[3])]
            if y_proba is not None and y_proba.shape[1] == 2:
                try: met_list.append(("ROC-AUC", skm.roc_auc_score(y_true, y_proba[:,1]), _PAL[4]))
                except Exception: pass
            for name, val, color in met_list:
                cards_html += (f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-top:3px solid {color};border-radius:6px;padding:10px 14px;text-align:center;min-width:110px;'>"
                               f"<div style='font-size:0.7em;text-transform:uppercase;color:#94a3b8;margin-bottom:4px;'>{name}</div>"
                               f"<div style='font-size:1.3em;font-weight:700;color:#1e293b;'>{val:.4f}</div></div>")
            display(HTML(f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;'>{cards_html}</div>"))
            display(HTML(f"<pre style='font-size:0.78em;background:#f8fafc;padding:10px;border-radius:6px;border:1px solid #e2e8f0;'>{skm.classification_report(y_true, y_pred)}</pre>"))
        else:
            r2 = skm.r2_score(y_true, y_pred); mae = skm.mean_absolute_error(y_true, y_pred); rmse = np.sqrt(skm.mean_squared_error(y_true, y_pred))
            for name, val, color in [("R²", r2, _PAL[1]), ("MAE", mae, _PAL[2]), ("RMSE", rmse, _PAL[3])]:
                cards_html += (f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-top:3px solid {color};border-radius:6px;padding:10px 14px;text-align:center;min-width:110px;'>"
                               f"<div style='font-size:0.7em;text-transform:uppercase;color:#94a3b8;margin-bottom:4px;'>{name}</div>"
                               f"<div style='font-size:1.3em;font-weight:700;color:#1e293b;'>{val:.4f}</div></div>")
            display(HTML(f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;'>{cards_html}</div>"))


def runner(state) -> PredictionsUI:
    ui = PredictionsUI(state)
    if hasattr(ui, "ui"):
        display(ui.ui)
    return ui
