"""Étape 7 — Split & Balancing (SplitBalancingUI)."""
from __future__ import annotations
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml_pipeline.styles import styles

_COLORS = {"before": "#60a5fa", "after": "#34d399", "text": "#1e293b",
           "muted": "#64748b", "border": "#e2e8f0", "warning": "#f59e0b",
           "danger": "#ef4444", "success": "#10b981"}


def _bar_chart(ax, series, color: str, title: str) -> None:
    vc = series.value_counts().sort_index()
    ax.bar([str(l) for l in vc.index], vc.values, color=color, alpha=0.8, width=0.6)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6, color=_COLORS["text"])
    ax.set_xlabel(series.name, fontsize=8, color=_COLORS["muted"])
    ax.set_ylabel("Count", fontsize=8, color=_COLORS["muted"])
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color=_COLORS["border"], linestyle="--", linewidth=0.3, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(_COLORS["border"])


def _simulate_balance(y_series, method: str) -> pd.Series:
    if method == "none":
        return y_series.copy()
    counts = y_series.value_counts()
    if method == "undersample":
        min_size = counts.min()
        return pd.concat([y_series[y_series == cls].sample(min_size, random_state=42) for cls in counts.index])
    max_size = counts.max()
    return pd.concat([y_series[y_series == cls].sample(max_size, replace=True, random_state=42) for cls in counts.index])


class SplitBalancingUI:
    """Interface de sélection Train/Test et balancing des classes."""

    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("[ERROR] Configuration non chargée.")
            return
        self.dfs = {k: v for k, v in state.data_encoded.items() if isinstance(v, pd.DataFrame)}
        balancing_cfg = state.config.get("split", {}).get("balancing", {}).get("methods", [])
        if not balancing_cfg:
            # fallback sur l'ancienne clé
            balancing_cfg = state.config.get("balancing", [])
        self.balancing_methods = [(b["label"], b["value"]) for b in balancing_cfg]
        # index rapide label→value et value→label pour les Dropdowns
        self._bal_val_to_label = {b["value"]: b["label"] for b in balancing_cfg}
        self._bal_label_to_val = {b["label"]: b["value"] for b in balancing_cfg}
        self._build_ui()

    def _build_ui(self) -> None:
        if not self.dfs:
            self.ui = styles.error_msg("Aucun DataFrame encodé disponible.")
            return
        self.train_ds_selector = widgets.Dropdown(options=list(self.dfs.keys()), description="Train Dataset:", style={"description_width": "initial"})
        self.test_ds_selector  = widgets.Dropdown(options=["<None>"]+list(self.dfs.keys()), description="Test Dataset:", style={"description_width": "initial"})
        target_opts = list(self.dfs[self.train_ds_selector.value].columns) if self.dfs else []
        default_target = self.state.business_context.get("target")
        self.target_selector = widgets.Dropdown(
            options=target_opts, description="Target:",
            value=default_target if default_target in target_opts else (target_opts[0] if target_opts else None),
            style={"description_width": "initial"})
        self.col_config_container = widgets.VBox()
        self.grid_output    = widgets.Output()
        self.balance_output = widgets.Output()
        self.btn_preview  = widgets.Button(description="Preview Balancing", layout=widgets.Layout(width="140px"))
        self.btn_validate = widgets.Button(description="Validate & Balance", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="140px"))
        self.btn_preview.on_click(self._do_preview)
        self.btn_validate.on_click(self._do_balance)
        self.train_ds_selector.observe(self._on_train_ds_change, names="value")
        header = widgets.HTML(styles.card_html("Dataset Balancing", "Preview and validate class balancing", ""))
        self.ui = widgets.VBox([
            header,
            widgets.HBox([self.train_ds_selector, self.test_ds_selector, self.target_selector],
                          layout=widgets.Layout(margin="0 0 16px 0")),
            self.col_config_container,
            widgets.HBox([self.btn_preview, self.btn_validate],
                          layout=widgets.Layout(margin="10px 0", gap="12px")),
            self.grid_output, self.balance_output,
        ], layout=widgets.Layout(width="100%", max_width="1000px",
                                  border="1px solid #e2e8f0", padding="16px", border_radius="8px"))
        self._build_col_config()

    def _build_col_config(self) -> None:
        ds_key = self.train_ds_selector.value
        df = self.dfs[ds_key]
        self.row_widgets = {}
        headers = widgets.HBox([
            widgets.HTML("<div style='width:180px;font-weight:600;color:#475569'>Column</div>"),
            widgets.HTML("<div style='width:140px;font-weight:600;color:#475569'>Dominant Class</div>"),
            widgets.HTML("<div style='width:100px;font-weight:600;color:#475569'>Significant?</div>"),
            widgets.HTML("<div style='width:200px;font-weight:600;color:#475569'>Method</div>"),
        ], layout=widgets.Layout(border_bottom="1px solid #e2e8f0", padding="0 0 6px 0", margin="0 0 8px 0"))
        rows = [headers]
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique <= 20 and not pd.api.types.is_float_dtype(df[col]):
                vc = df[col].value_counts(normalize=True)
                if len(vc) < 2:
                    continue
                max_pct = vc.max() * 100
                is_imbalanced = max_pct > 75
                color  = _COLORS["danger"] if max_pct > 90 else _COLORS["warning"] if is_imbalanced else _COLORS["success"]
                status = "Severe Imbalance" if max_pct > 90 else "Moderate Imbalance" if is_imbalanced else "Balanced"
                lbl_col  = widgets.HTML(f"<div style='width:180px;font-weight:500;color:#1e293b'>{col}</div>")
                lbl_dist = widgets.HTML(f"<div style='width:140px;color:{color};font-size:12px'>{status} ({max_pct:.1f}%)</div>")
                chk_signif = widgets.Checkbox(value=False, indent=False, layout=widgets.Layout(width="100px"))
                dd_method  = widgets.Dropdown(options=self.balancing_methods,
                                               value=self._bal_val_to_label.get("smote", self.balancing_methods[0][0]) if max_pct > 90
                                                     else self._bal_val_to_label.get("none", self.balancing_methods[0][0]),
                                               layout=widgets.Layout(width="190px"))
                self.row_widgets[col] = {"signif": chk_signif, "method": dd_method}
                rows.append(widgets.HBox([lbl_col, lbl_dist, chk_signif, dd_method],
                                          layout=widgets.Layout(padding="4px 0", border_bottom="1px solid #f1f5f9")))
        if len(rows) == 1:
            rows.append(widgets.HTML("<i style='color:#64748b'>No categorical columns detected.</i>"))
        self.col_config_container.children = rows

    def _on_train_ds_change(self, change) -> None:
        if change["new"] in self.dfs:
            self.target_selector.options = list(self.dfs[change["new"]].columns)
            with self.grid_output: clear_output()
            with self.balance_output: clear_output()
            self._build_col_config()

    def _do_preview(self, _=None) -> None:
        ds_key = self.train_ds_selector.value
        df = self.dfs[ds_key]
        cols_to_show = [c for c in self.row_widgets if c in df.columns]
        if not cols_to_show:
            return
        n = len(cols_to_show)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3*n))
        if n == 1: axes = axes.reshape(1, -1)
        for i, col in enumerate(cols_to_show):
            method = self.row_widgets[col]["method"].value
            y_orig = df[col].dropna()
            y_sim  = _simulate_balance(y_orig, method)
            _bar_chart(axes[i, 0], y_orig, _COLORS["before"], f"{col} - Before")
            _bar_chart(axes[i, 1], y_sim,  _COLORS["after"],  f"{col} - After ({method})")
        plt.suptitle("Preview: Before / After Balancing", fontsize=11, fontweight="bold", y=0.98)
        plt.tight_layout()
        with self.grid_output:
            clear_output(wait=True); plt.show(); plt.close(fig)

    def _do_balance(self, _=None) -> None:
        with self.balance_output:
            clear_output(wait=True)
        train_key  = self.train_ds_selector.value
        test_key   = self.test_ds_selector.value
        target_col = self.target_selector.value
        if not target_col or target_col not in self.dfs[train_key].columns:
            with self.balance_output: print("Invalid target.")
            return
        df_train = self.dfs[train_key].dropna(subset=[target_col]).copy()
        X_train  = df_train.drop(columns=[target_col])
        y_train  = df_train[target_col]
        y_train_before = y_train.copy()
        X_test, y_test = None, None
        if test_key != "<None>":
            if target_col in self.dfs[test_key].columns:
                df_test = self.dfs[test_key].dropna(subset=[target_col]).copy()
                X_test  = df_test.drop(columns=[target_col])
                y_test  = df_test[target_col]
            else:
                X_test = self.dfs[test_key].copy()
        is_clf = y_train.nunique() <= 20 and not pd.api.types.is_float_dtype(y_train)
        imbalance_config = {col: {"significant": wd["signif"].value, "method": wd["method"].value}
                            for col, wd in self.row_widgets.items()}
        self.state.imbalance_config = imbalance_config
        method = imbalance_config.get(target_col, {}).get("method", "none")
        balance_applied = False
        if method not in ("none", "class_weights") and is_clf:
            try:
                if method == "smote":
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(random_state=42)
                elif method == "oversample":
                    from imblearn.over_sampling import RandomOverSampler
                    sampler = RandomOverSampler(random_state=42)
                elif method == "undersample":
                    from imblearn.under_sampling import RandomUnderSampler
                    sampler = RandomUnderSampler(random_state=42)
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balance_applied = True
            except ImportError:
                if method == "undersample":
                    min_size = y_train_before.value_counts().min()
                    df_comb = pd.concat([X_train, y_train_before], axis=1)
                    df_res  = df_comb.groupby(target_col, group_keys=False).apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)
                else:
                    max_size = y_train_before.value_counts().max()
                    df_comb = pd.concat([X_train, y_train_before], axis=1)
                    df_res  = df_comb.groupby(target_col, group_keys=False).apply(lambda x: x.sample(max_size, replace=True, random_state=42)).reset_index(drop=True)
                X_train = df_res.drop(columns=[target_col])
                y_train = df_res[target_col]
                balance_applied = True
        self.state.data_splits = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "target": target_col, "train_dataset": train_key, "test_dataset": test_key,
            "target_pred_col": f"{target_col}_pred" if (test_key != "<None>" and target_col not in self.dfs.get(test_key, {}).get("columns", [])) else None
        }
        self.state.log_step("Dataset Config", "Train/Test Selection & Balance",
                             {"target": target_col, "balancing": method, "train_dataset": train_key})
        with self.balance_output:
            clear_output(wait=True)
            display(HTML(
                f"<div style='padding:12px;border:1px solid #e2e8f0;border-radius:6px;"
                f"background:#f8fafc;color:#1e293b;font-size:13px;max-width:600px;margin-top:8px'>"
                f"<b>Configuration sauvegardée</b><br>"
                f"X_train: <b>{X_train.shape[0]:,}</b> × <b>{X_train.shape[1]:,}</b><br>"
                + (f"X_test: <b>{X_test.shape[0]:,}</b> × <b>{X_test.shape[1]:,}</b><br>" if X_test is not None else "")
                + f"Target: <b>{target_col}</b> | Method: <b>{method}</b></div>"))


def runner(state) -> SplitBalancingUI:
    ui = SplitBalancingUI(state)
    if hasattr(ui, "ui"):
        display(ui.ui)
    return ui
