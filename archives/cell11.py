import sys
import os
import pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    from ml_pipeline.cell_0d_styles import styles
except ImportError:
    if 'styles' in globals():
        styles = globals()['styles']
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("styles", os.path.join(os.path.dirname(__file__), "cell_0d_styles.py"))
        styles = importlib.util.module_from_spec(spec)
        sys.modules["styles"] = styles
        spec.loader.exec_module(styles)

# Minimal color palette
COLORS = {
    "before": "#60a5fa",
    "after": "#34d399",
    "text": "#1e293b",
    "muted": "#64748b",
    "border": "#e2e8f0",
    "bg": "#ffffff",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "success": "#10b981"
}

def _bar_chart(ax, series, color, title):
    """Simple bar chart for categorical distribution."""
    vc = series.value_counts().sort_index()
    labels = [str(l) for l in vc.index]
    values = vc.values
    ax.bar(labels, values, color=color, alpha=0.8, width=0.6)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6, color=COLORS["text"])
    ax.set_xlabel(series.name, fontsize=8, color=COLORS["muted"])
    ax.set_ylabel("Count", fontsize=8, color=COLORS["muted"])
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis="y", color=COLORS["border"], linestyle="--", linewidth=0.3, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])

def _simulate_balance(y_series, method):
    """Simulate balancing for preview."""
    if method == "none":
        return y_series.copy()
    counts = y_series.value_counts()
    if method == "undersample":
        min_size = counts.min()
        return pd.concat([y_series[y_series == cls].sample(min_size, random_state=42) for cls in counts.index])
    elif method in ("oversample", "smote"):
        max_size = counts.max()
        return pd.concat([y_series[y_series == cls].sample(max_size, replace=True, random_state=42) for cls in counts.index])
    return y_series.copy()

class SplitBalancingUI:
    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("[ERROR] Configuration not loaded. Please run Cell 1a (Config) first.")
            return
        self.dfs = {k: v for k, v in state.data_encoded.items() if isinstance(v, pd.DataFrame)}
        balancing_cfg = state.config.get("balancing", [])
        self.balancing_methods = [(b["label"], b["value"]) for b in balancing_cfg]
        self._build_ui()

    def _apply_method_to_all(self, method):
        """Apply a balancing method to all columns."""
        for col, wd in self.row_widgets.items():
            wd["method"].value = method
        print(f"Applied '{method}' to all columns.")

    def _build_ui(self):
        if not self.dfs:
            self.ui = styles.error_msg("No encoded DataFrame available.")
            return

        # Dataset selectors
        self.train_ds_selector = widgets.Dropdown(
            options=list(self.dfs.keys()),
            description="Train Dataset:",
            style={"description_width": "initial"}
        )
        test_options = ["<None>"] + list(self.dfs.keys())
        self.test_ds_selector = widgets.Dropdown(
            options=test_options,
            description="Test Dataset:",
            style={"description_width": "initial"}
        )
        target_opts = list(self.dfs[self.train_ds_selector.value].columns) if self.dfs else []
        default_target = self.state.business_context.get("target")
        self.target_selector = widgets.Dropdown(
            options=target_opts,
            description="Target:",
            value=default_target if default_target in target_opts else (target_opts[0] if target_opts else None),
            style={"description_width": "initial"}
        )

        # Containers
        self.col_config_container = widgets.VBox()
        self.grid_output = widgets.Output()
        self.balance_output = widgets.Output()

        # Quick action buttons
        self.btn_apply_none = widgets.Button(
            description="Apply None to All",
            button_style="",
            layout=widgets.Layout(width="140px"),
            tooltip="Set all columns to 'None' (no balancing)"
        )
        self.btn_apply_smote = widgets.Button(
            description="Apply SMOTE to All",
            button_style="",
            layout=widgets.Layout(width="140px"),
            tooltip="Set all columns to 'SMOTE'"
        )
        self.btn_apply_oversample = widgets.Button(
            description="Apply Oversample to All",
            button_style="",
            layout=widgets.Layout(width="140px"),
            tooltip="Set all columns to 'Oversample'"
        )
        self.btn_apply_undersample = widgets.Button(
            description="Apply Undersample to All",
            button_style="",
            layout=widgets.Layout(width="140px"),
            tooltip="Set all columns to 'Undersample'"
        )

        self.btn_apply_none.on_click(lambda b: self._apply_method_to_all("none"))
        self.btn_apply_smote.on_click(lambda b: self._apply_method_to_all("smote"))
        self.btn_apply_oversample.on_click(lambda b: self._apply_method_to_all("oversample"))
        self.btn_apply_undersample.on_click(lambda b: self._apply_method_to_all("undersample"))

        # Main buttons
        self.btn_preview = widgets.Button(
            description="Preview Balancing",
            button_style="",
            layout=widgets.Layout(width="140px")
        )
        self.btn_validate = widgets.Button(
            description="Validate & Balance",
            button_style=styles.BTN_PRIMARY,
            layout=widgets.Layout(width="140px")
        )
        self.btn_preview.on_click(self._do_preview)
        self.btn_validate.on_click(self._do_balance)

        # Observers
        self.train_ds_selector.observe(self._on_train_ds_change, names="value")

        # Explanation box
        explanation_text = """
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px;margin-bottom:12px;'>
            <b style='color:#1e293b;'>What is Class Balancing?</b><br>
            <span style='color:#475569;font-size:13px;'>
                <b>Class:</b> A category or label in your dataset (e.g., "Yes/No", "Cat/Dog/Bird").<br>
                <b>Imbalanced Data:</b> When one class has significantly more samples than others (e.g., 95% "No" vs 5% "Yes").<br><br>

                <b>When to Balance:</b><br>
                - One class dominates (>75% of data)<br>
                - Minority class is important for your task<br>
                - You're using accuracy as a metric (can be misleading with imbalanced data)<br><br>

                <b>When NOT to Balance:</b><br>
                - Classes are naturally imbalanced in reality (e.g., fraud detection where 99% are normal transactions)<br>
                - You have very little data (undersampling would remove too many samples)<br>
                - You're using proper metrics (precision/recall/F1) and class weights<br>
                - The imbalance reflects real-world distribution you want to preserve<br><br>

                <b>Methods:</b><br>
                - <b>SMOTE:</b> Creates synthetic samples (good for small datasets)<br>
                - <b>Oversample:</b> Duplicates minority samples (risk of overfitting)<br>
                - <b>Undersample:</b> Reduces majority samples (loses data)<br>
                - <b>Class Weights:</b> Adjusts algorithm weights (keeps original data)
            </span>
        </div>
        """

        # Layout
        header = widgets.HTML(styles.card_html(
            "Dataset Balancing Configuration",
            "Preview and validate class balancing",
            ""
        ))

        self.ui = widgets.VBox([
            header,
            widgets.HTML(explanation_text),
            widgets.HBox(
                [self.train_ds_selector, self.test_ds_selector, self.target_selector],
                layout=widgets.Layout(margin="0 0 16px 0")
            ),
            widgets.HTML("<b style='color:#1e293b;margin-bottom:8px;display:block'>Quick Actions:</b>"),
            widgets.HBox(
                [self.btn_apply_none, self.btn_apply_smote, self.btn_apply_oversample, self.btn_apply_undersample],
                layout=widgets.Layout(margin="0 0 12px 0", gap="8px")
            ),
            self.col_config_container,
            widgets.HBox(
                [self.btn_preview, self.btn_validate],
                layout=widgets.Layout(margin="10px 0", gap="12px")
            ),
            self.grid_output,
            self.balance_output,
        ], layout=widgets.Layout(
            width="100%", max_width="1000px",
            border="1px solid #e2e8f0",
            padding="16px",
            border_radius="8px",
        ))

        self._build_col_config()

    def _build_col_config(self):
        ds_key = self.train_ds_selector.value
        df = self.dfs[ds_key]
        self.row_widgets = {}

        headers = widgets.HBox([
            widgets.HTML("<div style='width:180px;font-weight:600;color:#475569'>Column</div>"),
            widgets.HTML("<div style='width:140px;font-weight:600;color:#475569'>Dominant Class</div>"),
            widgets.HTML("<div style='width:100px;font-weight:600;color:#475569'>Significant?</div>"),
            widgets.HTML("<div style='width:200px;font-weight:600;color:#475569'>Method</div>"),
        ], layout=widgets.Layout(
            border_bottom="1px solid #e2e8f0",
            padding="0 0 6px 0",
            margin="0 0 8px 0"
        ))

        rows = [headers]

        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique <= 20 and not pd.api.types.is_float_dtype(df[col]):
                vc = df[col].value_counts(normalize=True)
                if len(vc) < 2:
                    continue
                max_pct = vc.max() * 100
                is_imbalanced = max_pct > 75
                color = COLORS["danger"] if max_pct > 90 else COLORS["warning"] if is_imbalanced else COLORS["success"]
                status = "Severe Imbalance" if max_pct > 90 else "Moderate Imbalance" if is_imbalanced else "Balanced"

                lbl_col = widgets.HTML(f"<div style='width:180px;font-weight:500;color:#1e293b'>{col}</div>")
                lbl_dist = widgets.HTML(f"<div style='width:140px;color:{color};font-size:12px'>{status} ({max_pct:.1f}%)</div>")
                chk_signif = widgets.Checkbox(value=False, indent=False, layout=widgets.Layout(width="100px"))
                dd_method = widgets.Dropdown(
                    options=self.balancing_methods,
                    value="smote" if max_pct > 90 else "none",
                    layout=widgets.Layout(width="190px")
                )

                self.row_widgets[col] = {"signif": chk_signif, "method": dd_method}
                rows.append(widgets.HBox(
                    [lbl_col, lbl_dist, chk_signif, dd_method],
                    layout=widgets.Layout(padding="4px 0", border_bottom="1px solid #f1f5f9")
                ))

        if len(rows) == 1:
            rows.append(widgets.HTML("<i style='color:#64748b'>No categorical columns detected.</i>"))

        self.col_config_container.children = rows

    def _on_train_ds_change(self, change):
        if change["new"] in self.dfs:
            self.target_selector.options = list(self.dfs[change["new"]].columns)
            with self.grid_output:
                clear_output()
            with self.balance_output:
                clear_output()
            self._build_col_config()

    def _do_preview(self, _=None):
        ds_key = self.train_ds_selector.value
        df = self.dfs[ds_key]

        cols_to_show = [col for col in self.row_widgets.keys() if col in df.columns]
        if not cols_to_show:
            return

        n = len(cols_to_show)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(cols_to_show):
            method = self.row_widgets[col]["method"].value
            y_orig = df[col].dropna()
            y_sim = _simulate_balance(y_orig, method)

            _bar_chart(axes[i, 0], y_orig, COLORS["before"], f"{col} - Before")
            _bar_chart(axes[i, 1], y_sim, COLORS["after"], f"{col} - After ({method})")

        plt.suptitle("Preview: Before / After Balancing", fontsize=11, fontweight="bold", y=0.98)
        plt.tight_layout()
        with self.grid_output:
            clear_output(wait=True)
            plt.show()
            plt.close(fig)

    def _do_balance(self, _=None):
        with self.balance_output:
            clear_output(wait=True)

        train_key = self.train_ds_selector.value
        test_key = self.test_ds_selector.value
        target_col = self.target_selector.value

        if not target_col or target_col not in self.dfs[train_key].columns:
            with self.balance_output:
                print("Invalid target for the training dataset.")
            return

        df_train = self.dfs[train_key].dropna(subset=[target_col]).copy()
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        y_train_before = y_train.copy()

        X_test, y_test = None, None
        if test_key != "<None>":
            if target_col in self.dfs[test_key].columns:
                df_test = self.dfs[test_key].dropna(subset=[target_col]).copy()
                X_test = df_test.drop(columns=[target_col])
                y_test = df_test[target_col]
            else:
                X_test = self.dfs[test_key].copy()
                print(f"Target {target_col} not found in Test Dataset ({test_key}). "
                      f"Test will be used for inference only. Column {target_col}_pred will be generated later.")

        is_classification = y_train.nunique() <= 20 and not pd.api.types.is_float_dtype(y_train)

        imbalance_config = {}
        for col, wd in self.row_widgets.items():
            imbalance_config[col] = {
                "significant": wd["signif"].value,
                "method": wd["method"].value
            }
        self.state.imbalance_config = imbalance_config

        method = imbalance_config.get(target_col, {}).get("method", "none")
        balance_applied = False

        if method not in ("none", "class_weights") and is_classification:
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
                    df_res = df_comb.groupby(target_col, group_keys=False).apply(
                        lambda x: x.sample(min_size, random_state=42)
                    ).reset_index(drop=True)
                    X_train = df_res.drop(columns=[target_col])
                    y_train = df_res[target_col]
                else:
                    max_size = y_train_before.value_counts().max()
                    df_comb = pd.concat([X_train, y_train_before], axis=1)
                    df_res = df_comb.groupby(target_col, group_keys=False).apply(
                        lambda x: x.sample(max_size, replace=True, random_state=42)
                    ).reset_index(drop=True)
                    X_train = df_res.drop(columns=[target_col])
                    y_train = df_res[target_col]
                balance_applied = True

        self.state.data_splits = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "target": target_col,
            "train_dataset": train_key,
            "test_dataset": test_key,
            "target_pred_col": (
                f"{target_col}_pred"
                if (test_key != "<None>" and target_col not in self.dfs[test_key].columns)
                else None
            )
        }
        self.state.log_step("Dataset Config", "Train/Test Selection & Balance", {
            "target": target_col,
            "balancing": method,
            "train_dataset": train_key,
            "test_dataset": test_key,
            "target_pred_col": self.state.data_splits["target_pred_col"]
        })

        # Display confirmation grid
        cols_to_show = list(self.row_widgets.keys())
        n = len(cols_to_show)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(cols_to_show):
            y_col = df_train[col].dropna() if col != target_col else y_train_before
            m = self.row_widgets[col]["method"].value
            y_after_col = (
                y_train if (col == target_col and balance_applied)
                else _simulate_balance(y_col, m) if m != "none"
                else y_col
            )

            _bar_chart(axes[i, 0], y_col, COLORS["before"], f"{col} - Before")
            _bar_chart(axes[i, 1], y_after_col, COLORS["after"], f"{col} - After")

            if (col == target_col and balance_applied) or (col != target_col and m != "none"):
                axes[i, 1].text(
                    0.98, 0.97, "APPLIED",
                    transform=axes[i, 1].transAxes,
                    ha="right", va="top",
                    fontsize=8, color=COLORS["after"], fontweight="bold"
                )

        plt.suptitle("Confirmation: Before / After Balancing", fontsize=11, fontweight="bold", y=0.98)
        plt.tight_layout()

        with self.grid_output:
            clear_output(wait=True)
            plt.show()
            plt.close(fig)

        with self.balance_output:
            clear_output(wait=True)
            display(HTML(
                f"<div style='padding:12px;border:1px solid #e2e8f0;border-radius:6px;"
                f"background:#f8fafc;color:#1e293b;font-size:13px;max-width:600px;margin-top:8px'>"
                f"<b>Configuration saved</b><br>"
                f"X_train: <b>{X_train.shape[0]:,}</b> × <b>{X_train.shape[1]:,}</b><br>"
                + (f"X_test: <b>{X_test.shape[0]:,}</b> × <b>{X_test.shape[1]:,}</b><br>" if X_test is not None else "")
                + f"Target: <b>{target_col}</b> | Method: <b>{method}</b>"
                f"</div>"
            ))

def runner(state):
    ui = SplitBalancingUI(state)
    display(ui.ui)
    return ui

try:
    runner(state)
except NameError:
    pass