"""Étape 9 — Évaluation (EvaluationUI).

Réexporte EvaluationUI depuis archives/cell13.py refactorisé.
"""
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
_GRAY = "#64748b"
_BG   = "#f8fafc"
_GRID = "#e2e8f0"
_MAX_LC_SAMPLES = 5_000


def _is_valid(arr) -> bool:
    if arr is None: return False
    if isinstance(arr, pd.DataFrame) and arr.empty: return False
    if isinstance(arr, pd.Series)    and arr.empty: return False
    if isinstance(arr, np.ndarray)   and arr.size == 0: return False
    return True


def _is_inference_mode(splits: dict) -> bool:
    return not _is_valid(splits.get("y_test")) and _is_valid(splits.get("X_test"))


def _resolve_eval_data(splits, predictions, model_name):
    pred = (predictions or {}).get(model_name, {})
    Xv = pred.get("X_val"); yv = pred.get("y_val")
    if not _is_valid(Xv) or not _is_valid(yv):
        Xv = splits.get("X_test"); yv = splits.get("y_test")
    return Xv, yv


def _fig(w=10, h=5, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    ax.grid(color=_GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(_GRID)
    if title: ax.set_title(title, fontsize=11, fontweight="bold", color="#1e293b", pad=10)
    return fig, ax


def _multi_fig(rows, cols, w=14, h=4):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h*rows))
    fig.patch.set_facecolor(_BG)
    axes_flat = np.array(axes).flatten() if rows*cols > 1 else [axes]
    for ax in axes_flat:
        ax.set_facecolor(_BG); ax.grid(color=_GRID, linewidth=0.8, zorder=0)
        ax.spines[["top","right"]].set_visible(False); ax.spines[["left","bottom"]].set_color(_GRID)
    return fig, axes_flat


def _metric_card(name, value, color="#6366f1") -> str:
    return (f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-top:3px solid {color};"
            f"border-radius:6px;padding:12px 16px;text-align:center;min-width:120px;'>"
            f"<div style='font-size:0.7em;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:6px;'>{name}</div>"
            f"<div style='font-size:1.5em;font-weight:700;color:#1e293b;'>{value}</div></div>")


def _section(title, color="#6366f1") -> widgets.HTML:
    return widgets.HTML(f"<div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;'>"
                        f"<div style='width:4px;height:20px;background:{color};border-radius:2px;'></div>"
                        f"<span style='font-size:0.95em;font-weight:700;color:#1e293b;'>{title}</span></div>")


def _warn(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#92400e;background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")


def _info(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#1e40af;background:#eff6ff;border-left:4px solid #3b82f6;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")


def _compute_metrics(model, X_eval, y_eval, task, subtask, cfg_metrics) -> dict:
    from sklearn import metrics as skm
    results = {}
    if task == "classification":
        y_pred = model.predict(X_eval)
        sub_cfg = cfg_metrics.get("classification", {}).get(subtask, cfg_metrics.get("classification", {}).get("binary", []))
        for m in sub_cfg:
            try:
                fn = getattr(skm, m["func"])
                kwargs = m.get("kwargs", {})
                if m["func"] in ("roc_auc_score", "log_loss"):
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_eval)
                        val = fn(y_eval, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob, **kwargs)
                    else: continue
                else:
                    val = fn(y_eval, y_pred, **kwargs)
                results[m["name"]] = val
            except Exception as e:
                results[m["name"]] = f"ERR: {e}"
    elif task == "regression":
        y_pred = model.predict(X_eval)
        for m in cfg_metrics.get("regression", []):
            try:
                fn = getattr(skm, m["func"])
                val = fn(y_eval, y_pred)
                if m.get("post") == "np.sqrt": val = np.sqrt(val)
                results[m["name"]] = val
            except Exception as e:
                results[m["name"]] = f"ERR: {e}"
    return results


def _plot_confusion_matrix(model, X_eval, y_eval, model_name, ax=None):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_pred = model.predict(X_eval)
    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    if ax is None:
        fig, ax = _fig(5, 4, f"Confusion Matrix — {model_name}")
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=10, fontweight="bold", color="#1e293b")
    ax.set_facecolor(_BG)
    return ax.figure


def _plot_roc_curves(models_dict, X_eval, y_eval):
    from sklearn.metrics import roc_curve, auc
    fig, ax = _fig(7, 5, "ROC Curves")
    for i, (name, model) in enumerate(models_dict.items()):
        try:
            if not hasattr(model, "predict_proba"): continue
            y_prob = model.predict_proba(X_eval)
            scores = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.max(axis=1)
            fpr, tpr, _ = roc_curve(y_eval, scores)
            ax.plot(fpr, tpr, color=_PAL[i % len(_PAL)], lw=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
        except Exception: pass
    ax.plot([0,1],[0,1],"--",color=_GRAY,lw=1); ax.set_xlabel("FPR",color=_GRAY); ax.set_ylabel("TPR",color=_GRAY)
    ax.legend(fontsize=9); ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); plt.tight_layout(); return fig


def _plot_residuals(model, X_eval, y_eval, model_name):
    y_pred = model.predict(X_eval); y_arr = np.array(y_eval); residuals = y_arr - y_pred
    fig, axes = _multi_fig(1, 3, w=14, h=4)
    axes[0].scatter(y_pred, residuals, alpha=0.5, color=_PAL[0], s=20)
    axes[0].axhline(0, color=_PAL[3], lw=1.5, ls="--"); axes[0].set_xlabel("Predicted",color=_GRAY); axes[0].set_ylabel("Residuals",color=_GRAY); axes[0].set_title(f"Residuals vs Predicted — {model_name}",fontsize=10,fontweight="bold",color="#1e293b")
    _mn = min(y_arr.min(), y_pred.min()); _mx = max(y_arr.max(), y_pred.max())
    axes[1].scatter(y_arr, y_pred, alpha=0.5, color=_PAL[1], s=20); axes[1].plot([_mn,_mx],[_mn,_mx],"--",color=_PAL[3],lw=1.5); axes[1].set_xlabel("Actual",color=_GRAY); axes[1].set_ylabel("Predicted",color=_GRAY); axes[1].set_title("Actual vs Predicted",fontsize=10,fontweight="bold",color="#1e293b")
    axes[2].hist(residuals, bins=30, color=_PAL[2], edgecolor="white", alpha=0.85); axes[2].set_xlabel("Residual",color=_GRAY); axes[2].set_ylabel("Count",color=_GRAY); axes[2].set_title("Residual Distribution",fontsize=10,fontweight="bold",color="#1e293b")
    plt.tight_layout(); return fig


def _plot_feature_importance(model, feature_names, model_name, top_n=20):
    importance = None
    if hasattr(model, "feature_importances_"): importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_; importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if importance is None: return None
    idx = np.argsort(importance)[-top_n:]
    fig, ax = _fig(8, max(4, len(idx)*0.35), f"Feature Importance — {model_name}")
    colors = [_PAL[0] if v > np.median(importance[idx]) else _PAL[2] for v in importance[idx]]
    ax.barh(np.array(feature_names)[idx], importance[idx], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Importance", color=_GRAY); ax.tick_params(labelsize=8); plt.tight_layout(); return fig


def _plot_learning_curve(model, X_train, y_train, model_name, task, max_samples=_MAX_LC_SAMPLES, cv=3, n_points=6):
    from sklearn.model_selection import learning_curve
    scoring = "accuracy" if task == "classification" else "r2"
    X_lc, y_lc = X_train, y_train; n = len(X_lc)
    if n > max_samples:
        rng = np.random.RandomState(42); idx = rng.choice(n, max_samples, replace=False)
        X_lc = X_lc.iloc[idx] if hasattr(X_lc, "iloc") else X_lc[idx]
        y_lc = y_lc.iloc[idx] if hasattr(y_lc, "iloc") else y_lc[idx]
    try:
        train_sz, train_sc, val_sc = learning_curve(model, X_lc, y_lc, cv=cv, n_jobs=1, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, n_points))
    except Exception: return None
    fig, ax = _fig(8, 4.5, f"Learning Curve — {model_name}")
    ax.fill_between(train_sz, train_sc.mean(1)-train_sc.std(1), train_sc.mean(1)+train_sc.std(1), alpha=0.15, color=_PAL[0])
    ax.fill_between(train_sz, val_sc.mean(1)-val_sc.std(1), val_sc.mean(1)+val_sc.std(1), alpha=0.15, color=_PAL[1])
    ax.plot(train_sz, train_sc.mean(1), "o-", color=_PAL[0], lw=2, label=f"Train ({train_sc.mean(1)[-1]:.3f})")
    ax.plot(train_sz, val_sc.mean(1), "s-", color=_PAL[1], lw=2, label=f"Val CV ({val_sc.mean(1)[-1]:.3f})")
    ax.set_xlabel("Training examples", color=_GRAY); ax.set_ylabel(scoring.upper(), color=_GRAY)
    ax.legend(fontsize=9); ax.set_ylim(bottom=max(0, ax.get_ylim()[0])); plt.tight_layout(); return fig


def _plot_metric_comparison(all_metrics, metric_name):
    names = list(all_metrics.keys())
    pairs = [(n, all_metrics[n].get(metric_name)) for n in names if isinstance(all_metrics[n].get(metric_name), (int, float))]
    if not pairs: return None
    names, vals = zip(*pairs)
    fig, ax = _fig(7, 3.5, f"Model Comparison — {metric_name}")
    bars = ax.bar(names, vals, color=_PAL[:len(vals)], edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#1e293b")
    ax.set_ylabel(metric_name, color=_GRAY); ax.set_ylim(0, min(1.1, max(vals)*1.15))
    plt.xticks(rotation=20, ha="right", fontsize=9); plt.tight_layout(); return fig


class EvaluationUI:
    """Interface d'évaluation et d'interprétabilité des modèles."""

    def __init__(self, state):
        self.state       = state
        self.config      = getattr(state, "config", {})
        self.splits      = getattr(state, "data_splits", {})
        self.models      = getattr(state, "models", {})
        self.predictions = getattr(state, "predictions", {})
        self.cfg_metrics = self.config.get("metrics", {})
        self.cfg_exp     = self.config.get("explainability", {})
        self.inference_mode = _is_inference_mode(self.splits)
        if not self.models:
            self.ui = styles.error_msg("Aucun modèle trouvé. Exécutez d'abord la cellule Modeling."); return
        if self.inference_mode:
            any_val = any(_is_valid(p.get("X_val")) and _is_valid(p.get("y_val")) for p in self.predictions.values())
            if not any_val:
                self.ui = styles.error_msg("Mode inférence : aucune donnée de validation interne trouvée."); return
        else:
            if not _is_valid(self.splits.get("X_test")) or not _is_valid(self.splits.get("y_test")):
                self.ui = styles.error_msg("Mode évaluation : X_test ou y_test manquant."); return
        self._build_ui()

    def _build_ui(self) -> None:
        header  = widgets.HTML(styles.card_html("Evaluation", "Evaluation & Interpretability", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        model_names = list(self.models.keys())
        self.dd_model = widgets.Dropdown(options=["-- All --"]+model_names, description="Model:", layout=widgets.Layout(width="280px"))
        self.task, self.subtask = self._detect_task()
        task_badge = widgets.HTML(f"<span style='background:#ede9fe;color:#6d28d9;border-radius:4px;padding:4px 10px;font-size:0.8em;font-weight:700;'>Task: {self.task} / {self.subtask}</span>")
        mode_banner = styles.help_box(
            "<b>Mode inférence :</b> évaluation sur la partition de validation interne." if self.inference_mode
            else "<b>Mode évaluation :</b> évaluation sur le test set fourni.",
            "#f59e0b" if self.inference_mode else "#10b981")
        self.tab = widgets.Tab()
        self.out_metrics = widgets.Output()
        tab0 = widgets.VBox([styles.help_box("<b>Metrics :</b> toutes les métriques configurées.", "#6366f1"), self.out_metrics])
        self.out_clf_plots = widgets.Output()
        self.btn_clf = widgets.Button(description="Generate plots", button_style="primary", layout=widgets.Layout(width="200px", margin="8px 0"))
        self.btn_clf.on_click(self._on_clf_plots)
        tab1 = widgets.VBox([styles.help_box("<b>Classification :</b> Confusion matrix, ROC, Precision-Recall.", "#10b981"), self.btn_clf, self.out_clf_plots])
        self.out_reg_plots = widgets.Output()
        self.btn_reg = widgets.Button(description="Generate plots", button_style="primary", layout=widgets.Layout(width="200px", margin="8px 0"))
        self.btn_reg.on_click(self._on_reg_plots)
        tab2 = widgets.VBox([styles.help_box("<b>Regression :</b> Residuals, Actual vs Predicted.", "#f59e0b"), self.btn_reg, self.out_reg_plots])
        self.out_fi = widgets.Output()
        self.btn_fi = widgets.Button(description="Feature Importance", button_style="primary", layout=widgets.Layout(width="200px", margin="4px 0 12px 0"))
        self.btn_fi.on_click(self._on_fi_plots)
        self.btn_lc = widgets.Button(description="Learning Curve", button_style="warning", layout=widgets.Layout(width="200px"))
        self.btn_lc.on_click(self._on_lc_plots)
        self.lc_samples = widgets.BoundedIntText(value=min(len(self.splits.get("X_train", [])), _MAX_LC_SAMPLES), min=200, max=max(len(self.splits.get("X_train", [])), 200), step=500, description="Max lignes :", style={"description_width": "initial"}, layout=widgets.Layout(width="220px"))
        self.lc_cv = widgets.BoundedIntText(value=3, min=2, max=10, description="CV folds :", style={"description_width": "initial"}, layout=widgets.Layout(width="160px"))
        self.lc_points = widgets.BoundedIntText(value=6, min=3, max=12, description="Nb points :", style={"description_width": "initial"}, layout=widgets.Layout(width="160px"))
        self.lc_n_est = widgets.BoundedIntText(value=30, min=5, max=200, step=5, description="n_estimators :", style={"description_width": "initial"}, layout=widgets.Layout(width="190px"))
        tab3 = widgets.VBox([styles.help_box("<b>Feature Importance</b> et <b>Learning Curve</b>.", "#3b82f6"),
            widgets.HBox([self.btn_fi], layout=widgets.Layout(margin="4px 0 12px 0")),
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:0 0 10px 0;'>"),
            widgets.HBox([self.lc_samples, self.lc_cv, self.lc_points, self.lc_n_est], layout=widgets.Layout(gap="12px", align_items="center", flex_wrap="wrap")),
            widgets.HBox([self.btn_lc], layout=widgets.Layout(margin="8px 0")), self.out_fi])
        self.out_exp = widgets.Output()
        exp_types = [e["name"] for e in self.cfg_exp.get("tabular", [])] or ["SHAP (Tree/Kernel)", "LIME (Tabular)"]
        self.dd_explainer = widgets.Dropdown(options=exp_types, description="Explainer:", layout=widgets.Layout(width="280px"))
        self.btn_exp = widgets.Button(description="Run explanation", button_style="primary", layout=widgets.Layout(width="220px", margin="8px 0"))
        self.btn_exp.on_click(self._on_explainability)
        tab4 = widgets.VBox([styles.help_box("<b>Interpretability :</b> SHAP et LIME.", "#8b5cf6"), widgets.HBox([self.dd_explainer, self.btn_exp]), self.out_exp])
        self.out_compare = widgets.Output()
        self.btn_compare = widgets.Button(description="Compare models", button_style="primary", layout=widgets.Layout(width="220px", margin="8px 0"))
        self.btn_compare.on_click(self._on_compare)
        tab5 = widgets.VBox([styles.help_box("<b>Comparison :</b> un graphe par métrique.", "#ec4899"), self.btn_compare, self.out_compare])
        all_tabs   = [tab0, tab1, tab2, tab3, tab4, tab5]
        all_titles = ["Metrics", "Clf — Curves", "Reg — Residuals", "Feature Importance", "Interpretability", "Comparison"]
        if self.task == "regression":   all_tabs.pop(1); all_titles.pop(1)
        elif self.task == "classification": all_tabs.pop(2); all_titles.pop(2)
        self.tab.children = all_tabs
        for i, t in enumerate(all_titles): self.tab.set_title(i, t)
        controls = widgets.HBox([self.dd_model, task_badge], layout=widgets.Layout(align_items="center", gap="16px", margin="0 0 10px 0"))
        self.ui = widgets.VBox([top_bar, mode_banner, controls, self.tab],
            layout=widgets.Layout(width="100%", max_width="1100px", border="1px solid #e5e7eb",
                                   padding="18px", border_radius="10px", background_color="#ffffff"))
        self._compute_all_metrics(); self._render_metrics()

    def _detect_task(self) -> tuple[str, str]:
        y_train = self.splits.get("y_train"); prob_type = "classification"; subtask = "binary"
        if _is_valid(y_train):
            n = y_train.nunique() if hasattr(y_train, "nunique") else len(np.unique(y_train))
            if n > 20: prob_type, subtask = "regression", "continuous"
            elif n > 2: prob_type, subtask = "classification", "multiclass"
        if hasattr(self.state, "business_context") and self.state.business_context.get("domain"):
            dv = self.state.business_context["domain"]
            for d in self.config.get("domains", {}).get("supported", []):
                if d["value"] == dv and d.get("task"):
                    prob_type = d["task"]; subtask = d.get("subtask") or subtask; break
        return prob_type, subtask

    def _selected_models(self) -> dict:
        sel = self.dd_model.value
        return self.models if sel == "-- All --" else {sel: self.models[sel]}

    def _compute_all_metrics(self) -> None:
        self._all_metrics = {}
        for name, model in self.models.items():
            X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, name)
            if not _is_valid(X_eval) or not _is_valid(y_eval):
                self._all_metrics[name] = {"ERROR": "evaluation data unavailable"}; continue
            try:
                self._all_metrics[name] = _compute_metrics(model, X_eval, y_eval, self.task, self.subtask, self.cfg_metrics)
            except Exception as e:
                self._all_metrics[name] = {"ERROR": str(e)}

    def _render_metrics(self) -> None:
        with self.out_metrics:
            clear_output(wait=True)
            for model_name, metrics in self._all_metrics.items():
                valid = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                cards_html = "".join(_metric_card(k, f"{v:.4f}", _PAL[i % len(_PAL)]) for i, (k, v) in enumerate(valid.items()))
                display(HTML(f"<div style='margin-bottom:16px;'><div style='font-size:0.85em;font-weight:700;color:#6d28d9;margin-bottom:8px;text-transform:uppercase;'>{model_name}</div>"
                              f"<div style='display:flex;flex-wrap:wrap;gap:8px;'>{cards_html}</div></div>"))

    def _on_clf_plots(self, b) -> None:
        with self.out_clf_plots:
            clear_output(wait=True)
            sel = self._selected_models()
            first_name = list(sel.keys())[0]
            X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, first_name)
            if not _is_valid(X_eval) or not _is_valid(y_eval):
                display(_warn("Evaluation data unavailable.")); return
            display(_section("Confusion Matrix", "#6366f1"))
            n = len(sel)
            if n == 1:
                name, model = list(sel.items())[0]
                try: plt.show(_plot_confusion_matrix(model, X_eval, y_eval, name))
                except Exception as e: display(_warn(f"Unavailable: {e}"))
            else:
                cols = min(3, n); rows = (n+cols-1)//cols
                fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows)); fig.patch.set_facecolor(_BG)
                axes_flat = np.array(axes).flatten() if rows*cols > 1 else [axes]
                for ax, (name, model) in zip(axes_flat, sel.items()):
                    Xe, ye = _resolve_eval_data(self.splits, self.predictions, name)
                    if _is_valid(Xe) and _is_valid(ye):
                        try: _plot_confusion_matrix(model, Xe, ye, name, ax=ax)
                        except Exception: ax.set_title(f"{name}: ERR", fontsize=9, color="#ef4444")
                    else: ax.set_title(f"{name}: data unavailable", fontsize=9, color="#ef4444")
                for ax in axes_flat[len(sel):]: ax.set_visible(False)
                plt.tight_layout(); plt.show()
            display(_section("ROC Curves", "#10b981"))
            try: plt.show(_plot_roc_curves(sel, X_eval, y_eval))
            except Exception as e: display(_warn(f"ROC unavailable: {e}"))

    def _on_reg_plots(self, b) -> None:
        with self.out_reg_plots:
            clear_output(wait=True)
            for name, model in self._selected_models().items():
                X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, name)
                if not _is_valid(X_eval) or not _is_valid(y_eval):
                    display(_warn(f"Evaluation data unavailable for {name}.")); continue
                display(_section(f"Residuals — {name}", "#f59e0b"))
                try: plt.show(_plot_residuals(model, X_eval, y_eval, name))
                except Exception as e: display(_warn(f"Residuals unavailable: {e}"))

    def _on_fi_plots(self, b) -> None:
        with self.out_fi:
            clear_output(wait=True)
            X_train = self.splits.get("X_train")
            if not _is_valid(X_train): display(_warn("X_train missing.")); return
            feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]
            for name, model in self._selected_models().items():
                display(_section(f"Feature Importance — {name}", "#3b82f6"))
                fig = _plot_feature_importance(model, feature_names, name)
                if fig: plt.show()
                else: display(_warn(f"{name} does not expose feature_importances_ or coef_."))

    def _on_lc_plots(self, b) -> None:
        with self.out_fi:
            clear_output(wait=True)
            X_train = self.splits.get("X_train"); y_train = self.splits.get("y_train")
            if not _is_valid(X_train) or not _is_valid(y_train): display(_warn("X_train or y_train missing.")); return
            max_samples = self.lc_samples.value; cv_folds = self.lc_cv.value
            n_points = self.lc_points.value; n_est_fast = self.lc_n_est.value
            for name, model in self._selected_models().items():
                display(_section(f"Learning Curve — {name}", "#6366f1"))
                from sklearn.base import clone as sk_clone
                fast_model = sk_clone(model)
                if hasattr(fast_model, "n_estimators"): fast_model.set_params(n_estimators=min(n_est_fast, fast_model.n_estimators))
                if hasattr(fast_model, "n_jobs"): fast_model.set_params(n_jobs=1)
                fig = _plot_learning_curve(fast_model, X_train, y_train, name, self.task, max_samples=max_samples, cv=cv_folds, n_points=n_points)
                if fig: plt.show()
                else: display(_warn(f"Learning curve unavailable for {name}."))

    def _on_explainability(self, b) -> None:
        with self.out_exp:
            clear_output(wait=True)
            X_train = self.splits.get("X_train"); explainer_name = self.dd_explainer.value
            if not _is_valid(X_train): display(_warn("X_train invalid.")); return
            feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]
            for model_name, model in self._selected_models().items():
                X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, model_name)
                if not _is_valid(X_eval): display(_warn(f"Evaluation data unavailable for {model_name}.")); continue
                display(_section(f"{explainer_name} — {model_name}", "#8b5cf6"))
                if "SHAP" in explainer_name:
                    display(_info("Computing SHAP values..."))
                    try:
                        import shap
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_eval[:min(100, len(X_eval))])
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5)); fig.patch.set_facecolor(_BG)
                        plt.sca(axes[0]); shap.plots.beeswarm(shap_values, max_display=15, show=False)
                        plt.sca(axes[1]); shap.plots.bar(shap_values, max_display=15, show=False)
                        plt.tight_layout(); clear_output(wait=True); display(_section(f"SHAP — {model_name}", "#8b5cf6")); plt.show()
                    except ImportError: display(_warn("SHAP not available — pip install shap"))
                    except Exception as e: display(_warn(f"SHAP error: {e}"))
                elif "LIME" in explainer_name:
                    display(_info("Computing LIME explanation..."))
                    try:
                        import lime.lime_tabular
                        mode = "classification" if self.task == "classification" else "regression"
                        exp_obj = lime.lime_tabular.LimeTabularExplainer(X_train.values if hasattr(X_train, "values") else X_train, feature_names=feature_names, mode=mode)
                        predict_fn = model.predict_proba if hasattr(model, "predict_proba") and self.task == "classification" else model.predict
                        exp = exp_obj.explain_instance(X_eval.iloc[0].values if hasattr(X_eval, "iloc") else X_eval[0], predict_fn, num_features=15)
                        fig = exp.as_pyplot_figure(); fig.patch.set_facecolor(_BG); plt.tight_layout()
                        clear_output(wait=True); display(_section(f"LIME — {model_name}", "#8b5cf6")); plt.show()
                    except ImportError: display(_warn("LIME not available — pip install lime"))
                    except Exception as e: display(_warn(f"LIME error: {e}"))

    def _on_compare(self, b) -> None:
        with self.out_compare:
            clear_output(wait=True)
            if len(self._all_metrics) < 2: display(_warn("Comparison requires at least 2 trained models.")); return
            all_metric_names = set()
            for m in self._all_metrics.values(): all_metric_names.update([k for k, v in m.items() if isinstance(v, (int, float))])
            for metric in sorted(all_metric_names):
                display(_section(f"Comparison — {metric}", "#ec4899"))
                fig = _plot_metric_comparison(self._all_metrics, metric)
                if fig: plt.show()
            display(_section("Summary table", "#1e293b"))
            rows = [{"Model": name, **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()}} for name, metrics in self._all_metrics.items()]
            if rows:
                df_summary = pd.DataFrame(rows).set_index("Model")
                display(df_summary.style.set_properties(**{"font-size": "0.85em"}))
            self.state.log_step("Evaluation", "Metrics Computed", {"models": list(self._all_metrics.keys()), "task": self.task})


def runner(state) -> EvaluationUI:
    ev = EvaluationUI(state)
    if hasattr(ev, "ui"):
        display(ev.ui)
    return ev
