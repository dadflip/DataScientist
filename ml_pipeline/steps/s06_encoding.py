"""Étape 6 — Encodage & Outliers (UltimateEncoder)."""
from __future__ import annotations
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml_pipeline.styles import styles


def _dynamic_import(import_string: str):
    if not import_string:
        return None
    import importlib
    parts = import_string.split(".")
    module = importlib.import_module(".".join(parts[:-1]))
    return getattr(module, parts[-1])


class UltimateEncoder:
    """Interface d'encodage des variables + gestion des outliers."""

    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("Configuration non chargée.")
            return
        self.config = state.config.get("encoding", {})
        self.all_datasets = getattr(state, "data_cleaned", {})
        if not hasattr(state, "meta"):
            state.meta = {}
        self.meta = state.meta
        cleaning_config = state.config.get("cleaning", {})
        self.outlier_options = [(opt["label"], opt["value"])
                                for opt in cleaning_config.get("outliers", [])]
        if not any(v == "none" for _, v in self.outlier_options):
            self.outlier_options.insert(0, ("Do nothing", "none"))
        self.datasets    = {k: v.copy() for k, v in self.all_datasets.items() if isinstance(v, pd.DataFrame)}
        self.non_tabular = {k: v for k, v in self.all_datasets.items() if not isinstance(v, pd.DataFrame)}
        self._sync_metadata()
        self._build_ui()

    def _sync_metadata(self) -> None:
        for ds_name, df in self.datasets.items():
            if ds_name not in self.meta:
                self.meta[ds_name] = {}
            for col in df.columns:
                if col not in self.meta[ds_name]:
                    s = df[col]; n_unq = s.nunique()
                    if pd.api.types.is_datetime64_any_dtype(s): kind = "datetime"
                    elif pd.api.types.is_bool_dtype(s) or n_unq == 2: kind = "binary"
                    elif pd.api.types.is_numeric_dtype(s):
                        kind = "id_like" if n_unq / max(len(s), 1) > 0.95 else "numeric"
                    else:
                        kind = "categorical" if n_unq < 100 else "text"
                    self.meta[ds_name][col] = {"kind": kind}

    def _calc_outliers(self, df_col, o_act: str) -> tuple[int, int]:
        if not pd.api.types.is_numeric_dtype(df_col):
            return 0, len(df_col)
        s = df_col.dropna()
        if len(s) == 0:
            return 0, 0
        if o_act == "clip_iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()), len(s)
        if o_act == "drop_zscore":
            std = s.std()
            if std > 0:
                return int((((s - s.mean()) / std).abs() > 3).sum()), len(s)
        return 0, len(s)

    def _apply_outliers(self, df: pd.DataFrame, row_widgets: dict) -> pd.DataFrame:
        df_new = df.copy()
        for col, wd in row_widgets.items():
            o_act = wd["outlier_dd"].value
            flag_it = wd.get("flag_cb", widgets.Checkbox(value=False)).value
            if o_act == "none" or col not in df_new.columns:
                continue
            if flag_it:
                if o_act == "clip_iqr":
                    q1, q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
                    iqr = q3 - q1
                    df_new[f"{col}_is_outlier"] = ((df_new[col] < q1-1.5*iqr) | (df_new[col] > q3+1.5*iqr)).astype(int)
                elif o_act == "drop_zscore":
                    std = df_new[col].std()
                    if std > 0:
                        df_new[f"{col}_is_outlier"] = (((df_new[col] - df_new[col].mean()) / std).abs() > 3).astype(int)
            if o_act == "clip_iqr":
                q1, q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
                iqr = q3 - q1
                df_new[col] = df_new[col].clip(lower=q1-1.5*iqr, upper=q3+1.5*iqr)
            elif o_act == "drop_zscore":
                std = df_new[col].std()
                if std > 0:
                    df_new = df_new[(((df_new[col] - df_new[col].mean()) / std).abs() <= 3)]
        return df_new

    def _get_encoded_df(self) -> pd.DataFrame | None:
        if not getattr(self, "current_ds", None):
            return None
        df = self.datasets[self.current_ds].copy()
        if not hasattr(self, "_tab_enc_widgets"):
            return df
        for col, wd in self._tab_enc_widgets.items():
            enc_value = wd["dd"].value; kind = wd["kind"]
            options_config = self.config.get("tabular", {}).get(kind, [])
            opt_info = next((o for o in options_config if o["value"] == enc_value), None)
            if col not in df.columns:
                continue
            if enc_value == "drop":
                df.drop(columns=[col], inplace=True)
            elif opt_info and "code" in opt_info and opt_info["code"]:
                loc_env = {"df": df, "col": col, "params": opt_info.get("params", {})}
                try:
                    exec(opt_info["code"], globals(), loc_env)
                    df = loc_env["df"]
                except Exception:
                    pass
        return df

    def _build_ui(self) -> None:
        tabs_children = []
        if self.datasets:
            self.ds_selector = widgets.Dropdown(options=list(self.datasets.keys()), description="Dataset:")
            self.outlier_timing = widgets.Dropdown(
                options=["Before Encoding", "After Encoding"], value="Before Encoding",
                description="Outliers:", layout=widgets.Layout(width="350px"))
            help_text = styles.help_box(
                "<b>Encoding & Outliers</b> — configurez l'encodage par colonne et le timing des outliers.<br>"
                "<b>Before Encoding</b> (recommandé) : les outliers sont traités avant la mise à l'échelle.<br>"
                "<b>After Encoding</b> : utile si l'encodage génère de nouvelles métriques numériques.",
                "#10b981")
            selectors = widgets.HBox([self.ds_selector, self.outlier_timing],
                                      layout=widgets.Layout(margin="0 0 10px 0", gap="20px"))
            self.ds_selector.observe(self._on_tab_ds_change, names="value")
            self.outlier_timing.observe(self._build_outliers_table, names="value")
            self.current_ds = self.ds_selector.value
            self.enc_container     = widgets.VBox()
            self.outlier_container = widgets.VBox()
            self.outlier_plot_out  = widgets.Output()
            self.tab_out           = widgets.Output()
            btn_apply = widgets.Button(description="Execute Pipeline",
                                        button_style=styles.BTN_PRIMARY,
                                        layout=styles.LAYOUT_BTN_LARGE)
            btn_apply.on_click(self._apply_tabular)
            self._on_tab_ds_change(None)
            tabs_children.append(widgets.VBox([selectors, help_text, self.enc_container,
                                               self.outlier_container, self.outlier_plot_out,
                                               widgets.HBox([btn_apply], layout=widgets.Layout(margin="20px 0 10px 0")),
                                               self.tab_out]))
        else:
            tabs_children.append(widgets.HTML("<div style='padding:16px;'>Aucune donnée tabulaire nettoyée.</div>"))
        tabs_children.append(widgets.HTML("<div style='padding:16px;'>Données non-tabulaires transmises directement.</div>"))
        tabs = widgets.Tab(children=tabs_children)
        tabs.set_title(0, "Tabular"); tabs.set_title(1, "Non-Tabular")
        header  = widgets.HTML(styles.card_html("Encode", "Encoding & Outliers", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.ui = widgets.VBox(
            [top_bar, tabs],
            layout=widgets.Layout(width="100%", max_width="1000px",
                                   border="1px solid #e5e7eb", padding="18px",
                                   border_radius="10px", background_color="#ffffff"))

    def _on_tab_ds_change(self, change) -> None:
        if change:
            self.current_ds = change["new"]
        if not self.current_ds:
            return
        self._build_enc_table()
        self._build_outliers_table()

    def _build_enc_table(self) -> None:
        df = self.datasets[self.current_ds]
        self._tab_enc_widgets = {}
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px;font-weight:bold;color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:250px;font-weight:bold;color:#475569;'>Encoding Action</div>"),
        ], layout=widgets.Layout(border_bottom="2px solid #cbd5e1", padding="0 0 5px 0", margin="0 0 10px 0"))
        rows = [headers]
        for col in df.columns:
            kind = self.meta.get(self.current_ds, {}).get(col, {}).get("kind", "categorical")
            tabular_config = self.config.get("tabular", self.config)
            options_config = tabular_config.get(kind, [])
            opts = [(o["label"], o["value"]) for o in options_config]
            if not opts:
                opts = [("Passthrough", "none"), ("Drop column", "drop")]
            if not any(o[1] == "none" for o in opts):
                opts.insert(0, ("Passthrough", "none"))
            if not any(o[1] == "drop" for o in opts):
                opts.append(("Drop column", "drop"))
            defaults = {"id_like": "drop", "categorical": next((o[1] for o in opts if "onehot" in o[1]), opts[0][1]),
                        "numeric": next((o[1] for o in opts if o[1] in ("std","minmax","robust")), opts[0][1]),
                        "datetime": next((o[1] for o in opts if "extract" in o[1] or "epoch" in o[1]), opts[0][1]),
                        "binary": next((o[1] for o in opts if o[1] in ("label","bool_map")), opts[0][1])}
            default_val = defaults.get(kind, "none")
            if not any(o[1] == default_val for o in opts):
                default_val = opts[0][1]
            lbl_col = widgets.HTML(f"<div style='width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-top:4px;' title='{col}'><b>{col}</b> <span style='color:#94a3b8;font-size:0.8em;'>[{kind}]</span></div>")
            enc_dd = widgets.Dropdown(options=opts, value=default_val, layout=widgets.Layout(width="240px"))
            self._tab_enc_widgets[col] = {"dd": enc_dd, "kind": kind}
            rows.append(widgets.HBox([lbl_col, enc_dd], layout=widgets.Layout(padding="4px 0", border_bottom="1px solid #f1f5f9")))
        self.enc_container.children = [widgets.HTML("<h4 style='color:#3b82f6;'>Encoding Rules</h4>")] + rows

    def _build_outliers_table(self, change=None) -> None:
        if not getattr(self, "current_ds", None):
            return
        is_after = self.outlier_timing.value == "After Encoding"
        df = self._get_encoded_df() if is_after else self.datasets[self.current_ds]
        self._tab_outlier_widgets = {}
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px;font-weight:bold;color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:200px;font-weight:bold;color:#475569;'>Outlier Action</div>"),
            widgets.HTML("<div style='width:150px;font-weight:bold;color:#475569;'>Detected</div>"),
            widgets.HTML("<div style='width:150px;font-weight:bold;color:#475569;'>Create Indicator</div>"),
        ], layout=widgets.Layout(border_bottom="2px solid #cbd5e1", padding="0 0 5px 0", margin="0 0 10px 0"))
        rows = [headers]
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            lbl_col  = widgets.HTML(f"<div style='width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-top:4px;'><b>{col}</b></div>")
            out_dd   = widgets.Dropdown(options=self.outlier_options, value="none", layout=widgets.Layout(width="180px", margin="0 20px 0 0"))
            lbl_ratio = widgets.HTML("<div style='width:150px;color:#64748b;padding-top:4px;'>-</div>")
            flag_cb  = widgets.Checkbox(value=False, description="Significatif", indent=False, layout=styles.LAYOUT_BTN_STD, disabled=True)
            def _update_ratio(change, df_col=df[col], lbl=lbl_ratio, cb=flag_cb):
                if change["new"] == "none":
                    lbl.value = "<div style='width:150px;color:#64748b;padding-top:4px;'>-</div>"
                    cb.disabled = True; cb.value = False; return
                cb.disabled = False
                n_out, t_out = self._calc_outliers(df_col, change["new"])
                pct = (n_out/t_out)*100 if t_out > 0 else 0
                color = "#ef4444" if pct > 5 else "#f59e0b" if pct > 1 else "#10b981"
                lbl.value = f"<div style='width:150px;color:{color};padding-top:4px;'>{n_out}/{t_out} ({pct:.1f}%)</div>"
            out_dd.observe(_update_ratio, names="value")
            self._tab_outlier_widgets[col] = {"outlier_dd": out_dd, "flag_cb": flag_cb}
            rows.append(widgets.HBox([lbl_col, out_dd, lbl_ratio, flag_cb],
                                      layout=widgets.Layout(padding="4px 0", border_bottom="1px solid #f1f5f9", align_items="center")))
        step_title = "Outliers Rules (Applied After Encoding)" if is_after else "Outliers Rules (Applied Before Encoding)"
        self.outlier_container.children = [widgets.HTML(f"<h4 style='color:#eab308;margin-top:20px;'>{step_title}</h4>")] + rows

    def _apply_tabular(self, b) -> None:
        with self.tab_out:
            clear_output()
            if not getattr(self, "current_ds", None):
                return
            df = self.datasets[self.current_ds].copy()
            timing = self.outlier_timing.value
            if timing == "Before Encoding":
                df = self._apply_outliers(df, self._tab_outlier_widgets)
            for col, wd in self._tab_enc_widgets.items():
                enc_value = wd["dd"].value; kind = wd["kind"]
                options_config = self.config.get("tabular", {}).get(kind, [])
                opt_info = next((o for o in options_config if o["value"] == enc_value), None)
                if col not in df.columns:
                    continue
                if enc_value == "drop":
                    df.drop(columns=[col], inplace=True)
                elif opt_info and "code" in opt_info and opt_info["code"]:
                    loc_env = {"df": df, "col": col, "params": opt_info.get("params", {})}
                    try:
                        exec(opt_info["code"], globals(), loc_env)
                        df = loc_env["df"]
                    except Exception as e:
                        print(f"[Error] {enc_value} on {col}: {e}")
            if timing == "After Encoding":
                df = self._apply_outliers(df, self._tab_outlier_widgets)
            self.state.data_encoded[self.current_ds] = df
            self.state.log_step("Data Encoding", "Tabular Encoded",
                                 {"dataset": self.current_ds, "outliers_timing": timing})
            display(styles.info_msg(
                f"Encodage appliqué sur '{self.current_ds}'.<br>"
                f"Original : {self.datasets[self.current_ds].shape} → Final : {df.shape}"))


def runner(state) -> UltimateEncoder:
    encoder = UltimateEncoder(state)
    if hasattr(encoder, "ui"):
        display(encoder.ui)
    return encoder
