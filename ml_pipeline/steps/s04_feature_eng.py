"""Étape 4 — Feature Engineering (FeatureEngUI).

Ce module réexporte FeatureEngUI depuis archives/cell8.py refactorisé.
La classe est volumineuse (~1200 lignes) ; elle est intégrée ici directement.
"""
from __future__ import annotations
import math
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles

_TAB_CSS_INJECTED = False


def _inject_tab_css() -> None:
    global _TAB_CSS_INJECTED
    if _TAB_CSS_INJECTED:
        return
    _TAB_CSS_INJECTED = True
    display(HTML("""<style>
    .fe-tabs .jupyter-widgets-tab-bar {
        display:flex !important; flex-wrap:wrap !important;
        border-bottom:2px solid #e2e8f0 !important; gap:2px !important; }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab {
        flex-shrink:0 !important; white-space:nowrap !important;
        font-size:0.80em !important; padding:5px 12px !important;
        border-radius:6px 6px 0 0 !important; border:1px solid #e2e8f0 !important;
        border-bottom:none !important; background:#f8fafc !important;
        color:#64748b !important; font-weight:500 !important; }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab.p-mod-current {
        background:#ffffff !important; color:#1e293b !important;
        border-color:#cbd5e1 !important; border-bottom-color:#ffffff !important;
        font-weight:700 !important; }
    </style>"""))


class FeatureEngUI:
    """Interface Feature Engineering — 10 onglets : Preview, Math, Conditions, Formula, Text, Date, Binning, Viz, Dashboard, Manage."""

    def __init__(self, state):
        self.state = state
        if not self.state.config:
            self.ui = styles.error_msg("[ERROR] Configuration non chargée.")
            return
        self.tabular_datasets = {k: v for k, v in self.state.data_raw.items()
                                  if isinstance(v, pd.DataFrame)}
        if not self.tabular_datasets:
            self.ui = styles.error_msg("Aucun dataset tabulaire disponible.")
            return
        self.current_ds = list(self.tabular_datasets.keys())[0]
        _inject_tab_css()
        self._build_ui()

    def _get_df(self) -> pd.DataFrame:
        return self.tabular_datasets[self.current_ds]

    def _propagate_to_state(self) -> None:
        ds = self.current_ds
        df = self.tabular_datasets[ds]
        self.state.data_raw[ds] = df
        if hasattr(self.state, "data_cleaned"):
            self.state.data_cleaned[ds] = df.copy()
        if not hasattr(self.state, "meta"):
            self.state.meta = {}
        if ds not in self.state.meta:
            self.state.meta[ds] = {}
        meta_ds = self.state.meta[ds]
        for col in list(meta_ds.keys()):
            if col not in df.columns:
                del meta_ds[col]
        for col in df.columns:
            if col not in meta_ds:
                s = df[col]; n_unq = s.nunique()
                if pd.api.types.is_datetime64_any_dtype(s): kind = "datetime"
                elif pd.api.types.is_bool_dtype(s) or n_unq == 2: kind = "binary"
                elif pd.api.types.is_numeric_dtype(s):
                    kind = "id_like" if n_unq / max(len(s), 1) > 0.95 else "numeric"
                else:
                    kind = "categorical" if n_unq < 100 else "text"
                meta_ds[col] = {"kind": kind}

    def _sync_targets(self, op_fn, op_label: str) -> None:
        if not self.sync_check.value:
            return
        targets = list(self.sync_datasets.value)
        lines = []
        for ds_name in targets:
            if ds_name == self.current_ds:
                continue
            df_target = self.tabular_datasets.get(ds_name)
            if df_target is None:
                lines.append(f"<span style='color:#ef4444;'>⚠ '{ds_name}' introuvable.</span>")
                continue
            try:
                df_result = op_fn(df_target.copy())
                if df_result is not None:
                    self.tabular_datasets[ds_name] = df_result
                    self.state.data_raw[ds_name] = df_result
                    if hasattr(self.state, "data_cleaned"):
                        self.state.data_cleaned[ds_name] = df_result.copy()
                    lines.append(f"<span style='color:#10b981;'>✓ '{ds_name}' — {op_label}</span>")
                else:
                    lines.append(f"<span style='color:#94a3b8;'>– '{ds_name}' — aucune modification.</span>")
            except Exception as e:
                lines.append(f"<span style='color:#ef4444;'>⚠ '{ds_name}' — {e}</span>")
        if lines:
            with self.sync_status:
                clear_output(wait=True)
                display(HTML("<div style='font-size:0.79em;padding:4px 0;'><b style='color:#6366f1;'>Sync :</b> "
                              + "  ·  ".join(lines) + "</div>"))

    def _build_ui(self) -> None:
        header  = widgets.HTML(styles.card_html("Feature Engineering", "Advanced Variable Laboratory", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.ds_dd = widgets.Dropdown(options=list(self.tabular_datasets.keys()),
                                       value=self.current_ds, description="Dataset:",
                                       layout=styles.LAYOUT_DD_LONG)
        self.ds_dd.observe(self._on_ds_change, names="value")
        other_ds = [k for k in self.tabular_datasets if k != self.current_ds]
        self.sync_datasets = widgets.SelectMultiple(
            options=other_ds, value=other_ds, description="Sync vers :",
            layout=widgets.Layout(width="280px", height=f"{max(36, min(120, len(other_ds)*24))}px"),
            disabled=not bool(other_ds))
        self.sync_check = widgets.Checkbox(value=bool(other_ds),
                                            description="Répliquer sur d'autres datasets",
                                            layout=widgets.Layout(width="280px"),
                                            disabled=not bool(other_ds))
        self.sync_check.observe(lambda c: setattr(self.sync_datasets, "disabled", not c["new"]), names="value")
        self.sync_status = widgets.Output()
        self._sync_box = widgets.VBox([
            widgets.HBox([self.sync_check, self.sync_datasets],
                          layout=widgets.Layout(align_items="flex-start", gap="12px")),
            self.sync_status],
            layout=widgets.Layout(padding="8px 10px", margin="6px 0",
                                   border="1px solid #e0e7ff", border_radius="6px",
                                   background_color="#f5f3ff"))
        self.tabs = widgets.Tab()
        self._build_preview_tab()
        self._build_math_tab()
        self._build_condition_tab()
        self._build_formula_tab()
        self._build_text_tab()
        self._build_date_tab()
        self._build_binning_tab()
        self._build_viz_tab()
        self._build_dashboard_tab()
        self._build_manage_tab()
        self.tabs.children = [self.tab_preview, self.tab_math, self.tab_condition,
                               self.tab_formula, self.tab_text, self.tab_date,
                               self.tab_binning, self.tab_viz, self.tab_dashboard, self.tab_manage]
        for i, t in enumerate(["Data Preview", "Math & Logic", "Conditions", "Custom Formula",
                                "Text Operations", "Date / Time", "Binning",
                                "Visualization", "Target Dashboard", "Manage Columns"]):
            self.tabs.set_title(i, t)
        self.tabs.add_class("fe-tabs")
        self.ui = widgets.VBox(
            [top_bar, self.ds_dd, self._sync_box,
             widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:10px 0;'>"),
             self.tabs],
            layout=widgets.Layout(width="100%", max_width="1000px",
                                   border="1px solid #e5e7eb", padding="18px",
                                   border_radius="10px", background_color="#ffffff"))

    def _refresh_columns(self) -> None:
        cols = list(self._get_df().columns)
        for w in (self.math_col1, self.text_col, self.date_col, self.bin_col,
                  self.viz_x, self.viz_y, self.dash_target, self.cond_col):
            w.options = cols
        self.math_col2.options = ["(None)", "(Constant)"] + cols
        self.viz_hue.options   = ["(None)"] + cols
        self.dash_target.options = ["(None)"] + cols
        self.dash_features.options = cols
        self.cond_then_col.options = ["(Constant)"] + cols
        self.cond_else_col.options = ["(Constant)"] + cols
        self._refresh_formula_col_list()
        if hasattr(self, "manage_col"):
            self.manage_col.options = cols
        self._refresh_preview_col_selector()

    def _on_ds_change(self, change) -> None:
        if change["new"]:
            self.current_ds = change["new"]
            other_ds = [k for k in self.tabular_datasets if k != self.current_ds]
            self.sync_datasets.options = other_ds
            self.sync_datasets.value   = other_ds
            self.sync_check.disabled   = not bool(other_ds)
            self.sync_datasets.disabled = not (bool(other_ds) and self.sync_check.value)
            self._refresh_columns()
            self.viz_out.clear_output()
            if hasattr(self, "dash_out"):
                self.dash_out.clear_output()
            self._render_preview()

    def _notify(self, out_widget, msg: str, is_error: bool = False) -> None:
        color = "#ef4444" if is_error else "#10b981"
        tag   = "[ERROR]" if is_error else "[OK]"
        with out_widget:
            clear_output(wait=True)
            display(HTML(f"<div style='color:{color};font-weight:bold;font-size:0.85em;'>{tag} {msg}</div>"))

    # ── Preview tab ───────────────────────────────────────────────────────────
    def _build_preview_tab(self) -> None:
        df = self._get_df(); cols = list(df.columns)
        self.preview_rows = widgets.IntSlider(value=20, min=5, max=500, step=5, description="Rows:", layout=widgets.Layout(width="320px"))
        self.preview_search = widgets.Text(placeholder='Filter: col==value or col>0', description="Filter:", layout=widgets.Layout(width="480px"))
        self.preview_col_select = widgets.SelectMultiple(options=cols, value=cols[:min(len(cols), 30)], description="Columns:", layout=widgets.Layout(width="320px", height="90px"))
        self.preview_col_all_btn = widgets.Button(description="All cols", button_style="info", layout=widgets.Layout(width="90px", height="28px"))
        self.preview_col_new_btn = widgets.Button(description="New cols only", layout=widgets.Layout(width="110px", height="28px"))
        self.preview_highlight_new = widgets.Checkbox(value=True, description="Highlight new cols", layout=widgets.Layout(width="170px"))
        self.preview_show_stats = widgets.Checkbox(value=False, description="Show quick stats", layout=widgets.Layout(width="150px"))
        self.preview_sort_col = widgets.Dropdown(options=["(none)"] + cols, value="(none)", description="Sort by:", layout=widgets.Layout(width="220px"))
        self.preview_sort_asc = widgets.Checkbox(value=True, description="Ascending", layout=widgets.Layout(width="110px"))
        self.preview_refresh_btn = widgets.Button(description="Refresh", button_style="primary", layout=widgets.Layout(width="100px", height="32px"))
        self.preview_out = widgets.Output()
        self._preview_original_cols = set(cols)
        self.preview_col_all_btn.on_click(lambda _: setattr(self.preview_col_select, "value", list(self.preview_col_select.options)))
        self.preview_col_new_btn.on_click(lambda _: setattr(self.preview_col_select, "value",
            [c for c in self._get_df().columns if c not in self._preview_original_cols] or list(self.preview_col_select.options)))
        self.preview_refresh_btn.on_click(lambda _: self._render_preview())
        for w in (self.preview_rows, self.preview_search, self.preview_sort_col, self.preview_sort_asc,
                  self.preview_highlight_new, self.preview_show_stats, self.preview_col_select):
            w.observe(lambda c: self._render_preview(), names="value")
        self.tab_preview = widgets.VBox([
            styles.help_box("<b>Live table view</b> — filtre, tri, highlight des nouvelles colonnes.", "#0ea5e9"),
            widgets.HBox([self.preview_rows, self.preview_sort_col, self.preview_sort_asc], layout=widgets.Layout(align_items="center", gap="10px", margin="0 0 6px 0")),
            widgets.HBox([self.preview_search], layout=widgets.Layout(margin="0 0 6px 0")),
            widgets.HBox([self.preview_col_select, widgets.VBox([self.preview_col_all_btn, self.preview_col_new_btn], layout=widgets.Layout(gap="4px", margin="0 0 0 6px"))], layout=widgets.Layout(align_items="flex-start", margin="0 0 6px 0")),
            widgets.HBox([self.preview_highlight_new, self.preview_show_stats, self.preview_refresh_btn], layout=widgets.Layout(align_items="center", gap="10px")),
            self.preview_out], layout=widgets.Layout(padding="10px"))
        self._render_preview()

    def _refresh_preview_col_selector(self) -> None:
        df = self._get_df(); cols = list(df.columns)
        current_val = list(self.preview_col_select.value)
        self.preview_col_select.options = cols
        still_valid = [c for c in current_val if c in cols]
        new_cols = [c for c in cols if c not in self._preview_original_cols]
        merged = still_valid + [c for c in new_cols if c not in still_valid]
        self.preview_col_select.value = merged if merged else cols[:min(len(cols), 30)]
        self.preview_sort_col.options = ["(none)"] + cols

    def _render_preview(self) -> None:
        df = self._get_df().copy()
        selected_cols = list(self.preview_col_select.value) or list(df.columns)
        filter_expr = self.preview_search.value.strip()
        filter_error = None
        if filter_expr:
            try: df = df.query(filter_expr)
            except Exception:
                try: df = df[df.eval(filter_expr)]
                except Exception as e: filter_error = str(e)
        sort_col = self.preview_sort_col.value
        if sort_col and sort_col != "(none)" and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=self.preview_sort_asc.value)
        n_rows = self.preview_rows.value
        view = df[selected_cols].head(n_rows)
        new_cols = set(self._get_df().columns) - self._preview_original_cols
        with self.preview_out:
            clear_output(wait=True)
            meta_parts = [f"<b>{len(df):,}</b> rows", f"<b>{len(self._get_df().columns)}</b> cols",
                          f"<b>{len(new_cols)}</b> new"]
            if filter_error: meta_parts.append(f"<span style='color:#ef4444;'>filter error: {filter_error}</span>")
            display(HTML("<div style='font-size:0.82em;color:#64748b;margin-bottom:8px;padding:6px 10px;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;'>" + "  ·  ".join(meta_parts) + "</div>"))
            ts = [{"selector": "thead th", "props": [("background-color","#f1f5f9"),("font-size","0.8em"),("padding","6px 10px")]},
                  {"selector": "td", "props": [("font-size","0.82em"),("padding","4px 10px"),("max-width","200px"),("overflow","hidden"),("text-overflow","ellipsis"),("white-space","nowrap")]}]
            if self.preview_highlight_new.value and new_cols:
                def _hl(col):
                    return ["background-color:#fef9c3;font-weight:600;" if col.name in new_cols else "" for _ in col]
                display(view.style.apply(_hl, axis=0).set_table_styles(ts).format(precision=4, na_rep="—"))
            else:
                display(view.style.set_table_styles(ts).format(precision=4, na_rep="—"))
            if len(df) > n_rows:
                display(HTML(f"<div style='font-size:0.78em;color:#94a3b8;margin-top:8px;text-align:right;'>Showing {n_rows} of {len(df):,} rows</div>"))

    # ── Math tab ──────────────────────────────────────────────────────────────
    def _build_math_tab(self) -> None:
        cols = list(self._get_df().columns)
        math_ops = self.state.config.get("feature_engineering", {}).get("math_operations", ["+","-","*","/","log(A)","exp(A)","sqrt(A)","A^2","Abs(A)","Modulo"])
        self.math_col1  = widgets.Dropdown(options=cols, description="Col A:", layout=styles.LAYOUT_DD)
        self.math_op    = widgets.Dropdown(options=math_ops, value=math_ops[0], description="Op:", layout=styles.LAYOUT_BTN_LARGE)
        self.math_col2  = widgets.Dropdown(options=["(None)","(Constant)"]+cols, description="Col B:", layout=styles.LAYOUT_DD)
        self.math_const = widgets.FloatText(value=0, description="Const:", layout=widgets.Layout(width="200px", display="none"))
        self.math_new_col = widgets.Text(description="New Name:", placeholder="result_col", layout=styles.LAYOUT_DD)
        self.math_btn   = widgets.Button(description="Apply Math", button_style=styles.BTN_PRIMARY)
        self.math_out   = widgets.Output()
        def _on_op(c):
            hide = c["new"] in ("log(A)","exp(A)","sqrt(A)","A^2","Abs(A)")
            self.math_col2.layout.display = "none" if hide else "block"
            self.math_const.layout.display = "none" if hide else ("block" if self.math_col2.value == "(Constant)" else "none")
        self.math_op.observe(_on_op, names="value")
        self.math_col2.observe(lambda c: setattr(self.math_const.layout, "display", "block" if c["new"] == "(Constant)" else "none"), names="value")
        self.math_btn.on_click(self._apply_math)
        self.tab_math = widgets.VBox([
            styles.help_box("Opérations mathématiques sur colonnes numériques.", "#8b5cf6"),
            widgets.HBox([self.math_col1, self.math_op, self.math_col2, self.math_const]),
            widgets.HBox([self.math_new_col, self.math_btn]),
            self.math_out], layout=widgets.Layout(padding="10px"))

    def _apply_math(self, _) -> None:
        df = self._get_df(); col1 = self.math_col1.value; col2 = self.math_col2.value
        op = self.math_op.value; new_name = self.math_new_col.value or f"{col1}_new"
        const_val = self.math_const.value
        try:
            v1 = pd.to_numeric(df[col1], errors="coerce")
            if op in ("log(A)","exp(A)","sqrt(A)","A^2","Abs(A)"):
                res = {"log(A)": np.log1p(v1), "exp(A)": np.exp(v1), "sqrt(A)": np.sqrt(np.maximum(v1,0)), "A^2": v1**2, "Abs(A)": np.abs(v1)}[op]
            else:
                v2 = const_val if col2 == "(Constant)" else pd.to_numeric(df[col2], errors="coerce")
                res = {"+": v1+v2, "-": v1-v2, "*": v1*v2, "/": v1/v2.replace(0,np.nan) if hasattr(v2,"replace") else v1/v2, "A^B/C": v1**v2, "Modulo": v1%v2}[op]
            self.tabular_datasets[self.current_ds][new_name] = res
            self.state.log_step("Feature Eng", "Math applied", {"op": op, "new_col": new_name})
            self._propagate_to_state()
            def _sync(df_t):
                v = pd.to_numeric(df_t[col1], errors="coerce") if col1 in df_t.columns else (_ for _ in ()).throw(KeyError(col1))
                df_t[new_name] = res  # simplified sync
                return df_t
            self._sync_targets(_sync, f"Math {op} → '{new_name}'")
            self._refresh_columns(); self._notify(self.math_out, f"Created '{new_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.math_out, str(e), True)

    # ── Condition tab ─────────────────────────────────────────────────────────
    def _build_condition_tab(self) -> None:
        cols = list(self._get_df().columns)
        ops_list = ["==","!=",">",">=","<","<=","isin","not isin","is null","is not null","contains (str)","startswith","endswith"]
        self.cond_col      = widgets.Dropdown(options=cols, description="Column:", layout=styles.LAYOUT_DD)
        self.cond_op       = widgets.Dropdown(options=ops_list, description="Operator:", layout=widgets.Layout(width="200px"))
        self.cond_val      = widgets.Text(description="Value:", placeholder="e.g. -1 or val1,val2", layout=widgets.Layout(width="280px"))
        self.cond_then_col = widgets.Dropdown(options=["(Constant)"]+cols, value="(Constant)", description="THEN col:", layout=styles.LAYOUT_DD)
        self.cond_then_val = widgets.Text(value="1", description="THEN val:", layout=widgets.Layout(width="160px"))
        self.cond_else_col = widgets.Dropdown(options=["(Constant)"]+cols, value="(Constant)", description="ELSE col:", layout=styles.LAYOUT_DD)
        self.cond_else_val = widgets.Text(value="0", description="ELSE val:", layout=widgets.Layout(width="160px"))
        self.cond_combine  = widgets.Dropdown(options=["AND","OR"], value="AND", description="Combine:", layout=widgets.Layout(width="150px"))
        self.cond_extra_rows: list = []
        self.cond_extra_box = widgets.VBox([])
        add_btn = widgets.Button(description="+ Add condition", button_style="info", layout=widgets.Layout(width="150px", height="28px"))
        rem_btn = widgets.Button(description="- Remove last", button_style="warning", layout=widgets.Layout(width="130px", height="28px"))
        def _add(_):
            r = (widgets.Dropdown(options=cols, layout=widgets.Layout(width="200px")),
                 widgets.Dropdown(options=ops_list, layout=widgets.Layout(width="180px")),
                 widgets.Text(placeholder="value", layout=widgets.Layout(width="220px")))
            self.cond_extra_rows.append(r)
            self.cond_extra_box.children = [widgets.HBox(list(r)) for r in self.cond_extra_rows]
        def _rem(_):
            if self.cond_extra_rows:
                self.cond_extra_rows.pop()
                self.cond_extra_box.children = [widgets.HBox(list(r)) for r in self.cond_extra_rows]
        add_btn.on_click(_add); rem_btn.on_click(_rem)
        self.cond_map_text = widgets.Textarea(placeholder="jan:1, feb:2\n(leave empty to use IF/THEN)", description="Value Map:", layout=widgets.Layout(width="460px", height="80px"))
        self.cond_new_col  = widgets.Text(description="New col:", placeholder="flag_col", layout=styles.LAYOUT_DD)
        self.cond_btn      = widgets.Button(description="Apply Condition", button_style=styles.BTN_PRIMARY)
        self.cond_out      = widgets.Output()
        self.cond_btn.on_click(self._apply_condition)
        self.tab_condition = widgets.VBox([
            styles.help_box("<b>IF/THEN/ELSE</b> — flags binaires ou mapping de valeurs.", "#f97316"),
            widgets.HBox([self.cond_col, self.cond_op, self.cond_val, self.cond_combine]),
            widgets.HBox([add_btn, rem_btn]), self.cond_extra_box,
            widgets.HBox([self.cond_then_col, self.cond_then_val, self.cond_else_col, self.cond_else_val]),
            self.cond_map_text, widgets.HBox([self.cond_new_col, self.cond_btn]), self.cond_out],
            layout=widgets.Layout(padding="10px"))

    def _parse_cond_value(self, raw: str):
        raw = raw.strip()
        try: return int(raw)
        except ValueError: pass
        try: return float(raw)
        except ValueError: pass
        return raw

    def _build_mask(self, df, col, op, raw_val):
        s = df[col]
        if op == "is null":     return s.isna()
        if op == "is not null": return s.notna()
        val = self._parse_cond_value(raw_val)
        if op == "==":  return s == val
        if op == "!=":  return s != val
        if op == ">":   return pd.to_numeric(s, errors="coerce") > float(val)
        if op == ">=":  return pd.to_numeric(s, errors="coerce") >= float(val)
        if op == "<":   return pd.to_numeric(s, errors="coerce") < float(val)
        if op == "<=":  return pd.to_numeric(s, errors="coerce") <= float(val)
        if op == "isin":     return s.astype(str).isin([v.strip() for v in raw_val.split(",")])
        if op == "not isin": return ~s.astype(str).isin([v.strip() for v in raw_val.split(",")])
        if op == "contains (str)": return s.astype(str).str.contains(raw_val, na=False)
        if op == "startswith":     return s.astype(str).str.startswith(raw_val, na=False)
        if op == "endswith":       return s.astype(str).str.endswith(raw_val, na=False)
        raise ValueError(f"Unknown operator: {op}")

    def _resolve_value(self, df, col_dd_val, const_txt_val):
        return self._parse_cond_value(const_txt_val) if col_dd_val == "(Constant)" else df[col_dd_val]

    def _apply_condition(self, _) -> None:
        df = self._get_df().copy(); new_name = self.cond_new_col.value or "new_flag"
        try:
            map_raw = self.cond_map_text.value.strip()
            if map_raw:
                col = self.cond_col.value
                mapping = {}
                for pair in map_raw.replace("\n", ",").split(","):
                    if ":" not in pair: continue
                    k, v = pair.split(":", 1)
                    mapping[self._parse_cond_value(k.strip())] = self._parse_cond_value(v.strip())
                df[new_name] = df[col].map(mapping)
            else:
                mask = self._build_mask(df, self.cond_col.value, self.cond_op.value, self.cond_val.value)
                combine = self.cond_combine.value
                for (r_col, r_op, r_val) in self.cond_extra_rows:
                    extra = self._build_mask(df, r_col.value, r_op.value, r_val.value)
                    mask = (mask & extra) if combine == "AND" else (mask | extra)
                then_val = self._resolve_value(df, self.cond_then_col.value, self.cond_then_val.value)
                else_val = self._resolve_value(df, self.cond_else_col.value, self.cond_else_val.value)
                df[new_name] = np.where(mask, then_val, else_val)
            self.tabular_datasets[self.current_ds] = df
            self.state.log_step("Feature Eng", "Condition applied", {"new_col": new_name})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.cond_out, f"Created '{new_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.cond_out, traceback.format_exc(), True)

    # ── Formula tab ───────────────────────────────────────────────────────────
    def _build_formula_tab(self) -> None:
        cols = list(self._get_df().columns)
        self._formula_col_html = widgets.HTML(self._formula_col_list_html(cols))
        SNIPPETS = {
            "Binary flag (col != value)": "new_col = (col != -1).astype(int)",
            "Ratio A / B (safe)":         "new_col = col_a / col_b.replace(0, np.nan)",
            "Zscore normalize":           "new_col = (col - col.mean()) / col.std()",
            "Min-max normalize":          "new_col = (col - col.min()) / (col.max() - col.min())",
            "Log1p transform":            "new_col = np.log1p(col)",
            "Interaction A x B":          "new_col = col_a * col_b",
            "Conditional value":          "new_col = np.where(col > 0, col * 2, 0)",
            "Map string values":          "new_col = col.map({'yes': 1, 'no': 0})",
            "Extract year from date":     "new_col = pd.to_datetime(col).dt.year",
            "Count occurrences":          "new_col = col.map(col.value_counts())",
        }
        self._snippets = SNIPPETS
        self.formula_snippet = widgets.Dropdown(options=["-- pick a snippet --"]+list(SNIPPETS.keys()), description="Snippets:", layout=widgets.Layout(width="400px"))
        self.formula_insert_btn = widgets.Button(description="Insert", button_style="info", layout=widgets.Layout(width="80px", height="28px"))
        self.formula_editor = widgets.Textarea(placeholder="# Write Python expressions.\n# Column names are pd.Series variables.\n# np, pd, math available.\nnew_col = col_a / col_b.replace(0, np.nan)", layout=widgets.Layout(width="100%", height="180px"))
        self.formula_preview_btn = widgets.Button(description="Preview (10 rows)", layout=widgets.Layout(width="160px", height="32px"))
        self.formula_apply_btn   = widgets.Button(description="Apply to Dataset", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="160px", height="32px"))
        self.formula_out = widgets.Output()
        def _insert(_):
            key = self.formula_snippet.value
            if key in SNIPPETS:
                cur = self.formula_editor.value
                self.formula_editor.value = cur + ("\n" if cur and not cur.endswith("\n") else "") + SNIPPETS[key] + "\n"
        self.formula_insert_btn.on_click(_insert)
        self.formula_preview_btn.on_click(lambda _: self._run_formula(preview=True))
        self.formula_apply_btn.on_click(lambda _: self._run_formula(preview=False))
        self.tab_formula = widgets.VBox([
            styles.help_box("<b>Expressions Python libres</b> — colonnes disponibles comme variables pd.Series.", "#0ea5e9"),
            widgets.HTML("<b style='font-size:0.82em;color:#6b7280;'>Colonnes disponibles :</b>"),
            self._formula_col_html,
            widgets.HBox([self.formula_snippet, self.formula_insert_btn], layout=widgets.Layout(align_items="center", gap="8px", margin="6px 0")),
            self.formula_editor,
            widgets.HBox([self.formula_preview_btn, self.formula_apply_btn], layout=widgets.Layout(gap="10px", margin="8px 0 0 0")),
            self.formula_out], layout=widgets.Layout(padding="10px"))

    def _formula_col_list_html(self, cols: list) -> str:
        pills = "".join(f"<span style='display:inline-block;background:#ede9fe;color:#5b21b6;border-radius:4px;padding:2px 8px;margin:2px 3px;font-size:0.78em;font-family:monospace;'>{c}</span>" for c in cols)
        return f"<div style='line-height:2;margin-bottom:6px;'>{pills}</div>"

    def _refresh_formula_col_list(self) -> None:
        self._formula_col_html.value = self._formula_col_list_html(list(self._get_df().columns))

    def _run_formula(self, preview: bool = False) -> None:
        df = self._get_df().copy(); code = self.formula_editor.value.strip()
        if not code:
            self._notify(self.formula_out, "Formula editor is empty.", True); return
        ns = {col: df[col].copy() for col in df.columns}
        ns.update({"np": np, "pd": pd, "math": math, "df": df})
        with self.formula_out:
            clear_output(wait=True)
            try: exec(compile(code, "<formula>", "exec"), ns)
            except Exception:
                display(HTML(f"<pre style='color:#ef4444;font-size:0.82em;'>{traceback.format_exc()}</pre>")); return
        new_or_mod = {k: v for k, v in ns.items()
                      if not k.startswith("_") and k not in ("np","pd","math","df")
                      and isinstance(v, pd.Series) and (k not in df.columns or not df[k].equals(v))}
        if not new_or_mod:
            self._notify(self.formula_out, "No new/modified columns detected.", True); return
        if preview:
            with self.formula_out:
                display(HTML(f"<b style='color:#0ea5e9;'>Preview — {len(new_or_mod)} column(s) :</b>"))
                display(pd.DataFrame({k: v for k, v in new_or_mod.items()}).head(10))
        else:
            for k, v in new_or_mod.items():
                self.tabular_datasets[self.current_ds][k] = v.values if isinstance(v, pd.Series) else v
            self.state.log_step("Feature Eng", "Custom Formula applied", {"columns": list(new_or_mod.keys())})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.formula_out, f"Applied {len(new_or_mod)} column(s) : {', '.join(new_or_mod)}")
            self._render_preview()

    # ── Text tab ──────────────────────────────────────────────────────────────
    def _build_text_tab(self) -> None:
        cols = list(self._get_df().columns)
        text_ops = self.state.config.get("feature_engineering", {}).get("text_operations", ["Lowercase","Uppercase","Length","Extract Regex","Replace","Split & Keep N"])
        self.text_col     = widgets.Dropdown(options=cols, description="Col:", layout=styles.LAYOUT_DD)
        self.text_op      = widgets.Dropdown(options=text_ops, value=text_ops[0], description="Op:", layout=styles.LAYOUT_BTN_LARGE)
        self.text_arg1    = widgets.Text(description="Regex/Find/N:", layout=widgets.Layout(width="300px", display="none"))
        self.text_arg2    = widgets.Text(description="Replace/Sep:", layout=widgets.Layout(width="300px", display="none"))
        self.text_new_col = widgets.Text(description="New Name:", placeholder="text_result", layout=styles.LAYOUT_DD)
        self.text_btn     = widgets.Button(description="Apply Text Op", button_style=styles.BTN_PRIMARY)
        self.text_out     = widgets.Output()
        def _on_op(c):
            hide = c["new"] in ("Lowercase","Uppercase","Length")
            self.text_arg1.layout.display = "none" if hide else "block"
            self.text_arg2.layout.display = "none" if hide or c["new"] == "Extract Regex" else "block"
        self.text_op.observe(_on_op, names="value")
        self.text_btn.on_click(self._apply_text)
        self.tab_text = widgets.VBox([
            styles.help_box("Opérations sur colonnes texte.", "#06b6d4"),
            widgets.HBox([self.text_col, self.text_op, self.text_arg1, self.text_arg2]),
            widgets.HBox([self.text_new_col, self.text_btn]), self.text_out],
            layout=widgets.Layout(padding="10px"))

    def _apply_text(self, _) -> None:
        df = self._get_df(); col = self.text_col.value; op = self.text_op.value
        new_name = self.text_new_col.value or f"{col}_txt"
        try:
            s = df[col].astype(str)
            res = {"Lowercase": s.str.lower(), "Uppercase": s.str.upper(), "Length": s.str.len(),
                   "Extract Regex": s.str.extract(f"({self.text_arg1.value})", expand=False),
                   "Replace": s.str.replace(self.text_arg1.value, self.text_arg2.value, regex=True),
                   "Split & Keep N": s.str.split(self.text_arg2.value).str[int(self.text_arg1.value)]}.get(op)
            if res is None: raise ValueError(f"Unknown op: {op}")
            self.tabular_datasets[self.current_ds][new_name] = res
            self.state.log_step("Feature Eng", "Text Op applied", {"op": op, "new_col": new_name})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.text_out, f"Created '{new_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.text_out, str(e), True)

    # ── Date tab ──────────────────────────────────────────────────────────────
    def _build_date_tab(self) -> None:
        cols = list(self._get_df().columns)
        date_ops = self.state.config.get("feature_engineering", {}).get("date_operations", ["Year","Month","Day","DayOfWeek","Hour","Minute","IsWeekend"])
        self.date_col     = widgets.Dropdown(options=cols, description="Date Col:", layout=styles.LAYOUT_DD)
        self.date_extract = widgets.SelectMultiple(options=date_ops, value=date_ops[:2], description="Extract:", layout=styles.LAYOUT_DD)
        self.date_btn     = widgets.Button(description="Extract Features", button_style=styles.BTN_PRIMARY)
        self.date_out     = widgets.Output()
        self.date_btn.on_click(self._apply_date)
        self.tab_date = widgets.VBox([
            styles.help_box("Extraire des features temporelles depuis une colonne date.", "#f59e0b"),
            widgets.HBox([self.date_col, self.date_extract]), self.date_btn, self.date_out],
            layout=widgets.Layout(padding="10px"))

    def _apply_date(self, _) -> None:
        df = self._get_df(); col = self.date_col.value; features = self.date_extract.value
        try:
            s = pd.to_datetime(df[col], errors="coerce"); created = []
            mapping = {"Year": s.dt.year, "Month": s.dt.month, "Day": s.dt.day,
                       "DayOfWeek": s.dt.dayofweek, "Hour": s.dt.hour, "Minute": s.dt.minute,
                       "IsWeekend": (s.dt.dayofweek >= 5).astype(int)}
            for feat in features:
                if feat in mapping:
                    df[f"{col}_{feat}"] = mapping[feat]; created.append(f"{col}_{feat}")
            self.tabular_datasets[self.current_ds] = df
            self.state.log_step("Feature Eng", "Date Features extracted", {"col": col, "features": created})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.date_out, f"Created {len(created)} columns"); self._render_preview()
        except Exception as e:
            self._notify(self.date_out, str(e), True)

    # ── Binning tab ───────────────────────────────────────────────────────────
    def _build_binning_tab(self) -> None:
        cols = list(self._get_df().columns)
        bin_ops = self.state.config.get("feature_engineering", {}).get("binning_strategies", ["Equal Width (Cut)","Equal Frequency (Qcut)","Custom Edges"])
        self.bin_col     = widgets.Dropdown(options=cols, description="Num Col:", layout=styles.LAYOUT_DD)
        self.bin_method  = widgets.Dropdown(options=bin_ops, value=bin_ops[0], description="Method:", layout=styles.LAYOUT_BTN_LARGE)
        self.bin_bins    = widgets.Text(value="5", description="Bins/Edges:", layout=styles.LAYOUT_DD)
        self.bin_labels  = widgets.Checkbox(value=False, description="Numeric labels", layout=styles.LAYOUT_DD)
        self.bin_new_col = widgets.Text(description="New Name:", placeholder="binned_col", layout=styles.LAYOUT_DD)
        self.bin_btn     = widgets.Button(description="Apply Binning", button_style=styles.BTN_PRIMARY)
        self.bin_out     = widgets.Output()
        self.bin_btn.on_click(self._apply_binning)
        self.tab_binning = widgets.VBox([
            styles.help_box("Discrétiser des variables continues en bins.", "#10b981"),
            widgets.HBox([self.bin_col, self.bin_method, self.bin_bins]),
            widgets.HBox([self.bin_labels, self.bin_new_col, self.bin_btn]), self.bin_out],
            layout=widgets.Layout(padding="10px"))

    def _apply_binning(self, _) -> None:
        df = self._get_df(); col = self.bin_col.value; method = self.bin_method.value
        bins_val = self.bin_bins.value; new_name = self.bin_new_col.value or f"{col}_bin"
        try:
            s = pd.to_numeric(df[col], errors="coerce"); lbls = False if self.bin_labels.value else None
            if method == "Equal Width (Cut)":       res = pd.cut(s, bins=int(bins_val), labels=lbls)
            elif method == "Equal Frequency (Qcut)": res = pd.qcut(s, q=int(bins_val), labels=lbls, duplicates="drop")
            elif method == "Custom Edges":
                edges = [float(x.strip()) for x in bins_val.split(",")]
                res = pd.cut(s, bins=edges, labels=lbls)
            self.tabular_datasets[self.current_ds][new_name] = res
            self.state.log_step("Feature Eng", "Binning applied", {"method": method, "new_col": new_name})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.bin_out, f"Created '{new_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.bin_out, str(e), True)

    # ── Viz tab ───────────────────────────────────────────────────────────────
    def _build_viz_tab(self) -> None:
        cols = list(self._get_df().columns)
        viz_ops = self.state.config.get("feature_engineering", {}).get("viz_types", ["auto","scatter","line","bar","box","violin","hist","kde"])
        self.viz_x    = widgets.Dropdown(options=cols, description="X:", layout=widgets.Layout(width="250px"))
        self.viz_y    = widgets.Dropdown(options=cols, description="Y:", layout=widgets.Layout(width="250px"))
        self.viz_hue  = widgets.Dropdown(options=["(None)"]+cols, description="Hue:", layout=widgets.Layout(width="250px"))
        self.viz_kind = widgets.Dropdown(options=viz_ops, value="auto", description="Type:", layout=styles.LAYOUT_BTN_LARGE)
        self.viz_btn  = widgets.Button(description="Plot", button_style="info")
        self.viz_out  = widgets.Output()
        self.viz_btn.on_click(self._apply_viz)
        self.tab_viz = widgets.VBox([
            styles.help_box("Visualiser rapidement les nouvelles variables.", "#6366f1"),
            widgets.HBox([self.viz_x, self.viz_y, self.viz_hue, self.viz_kind, self.viz_btn]),
            self.viz_out], layout=widgets.Layout(padding="10px"))

    def _apply_viz(self, _) -> None:
        df = self._get_df(); x = self.viz_x.value; y = self.viz_y.value
        hue = None if self.viz_hue.value == "(None)" else self.viz_hue.value
        kind = self.viz_kind.value
        with self.viz_out:
            clear_output(wait=True)
            plt.figure(figsize=(9, 5))
            try:
                if kind == "auto":
                    kind = "scatter" if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]) else ("box" if pd.api.types.is_numeric_dtype(df[y]) else "bar")
                {"scatter": lambda: sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.7),
                 "line":    lambda: sns.lineplot(data=df, x=x, y=y, hue=hue),
                 "bar":     lambda: sns.barplot(data=df, x=x, y=y, hue=hue),
                 "box":     lambda: sns.boxplot(data=df, x=x, y=y, hue=hue),
                 "violin":  lambda: sns.violinplot(data=df, x=x, y=y, hue=hue),
                 "hist":    lambda: sns.histplot(data=df, x=x, hue=hue, kde=True),
                 "kde":     lambda: sns.kdeplot(data=df, x=x, hue=hue, fill=True)}.get(kind, lambda: None)()
                plt.title(f"{kind.capitalize()} : {y} vs {x}"); plt.tight_layout(); plt.show()
            except Exception as e:
                print(f"[ERROR] {e}")

    # ── Dashboard tab ─────────────────────────────────────────────────────────
    def _build_dashboard_tab(self) -> None:
        cols = list(self._get_df().columns)
        target_val = "(None)"
        if hasattr(self.state, "business_context") and self.state.business_context.get("target") in cols:
            target_val = self.state.business_context["target"]
        self.dash_target   = widgets.Dropdown(options=["(None)"]+cols, value=target_val, description="Target:", layout=styles.LAYOUT_DD)
        self.dash_features = widgets.SelectMultiple(options=cols, description="Features:", layout=widgets.Layout(width="300px", height="150px"))
        self.dash_btn      = widgets.Button(description="Plot Dashboard", button_style=styles.BTN_PRIMARY)
        self.dash_out      = widgets.Output()
        self.dash_btn.on_click(self._apply_dashboard)
        self.tab_dashboard = widgets.VBox([
            styles.help_box("Analyse multi-features vs Target (1 à 4 features).", "#be185d"),
            widgets.HBox([self.dash_target, self.dash_features, self.dash_btn]),
            self.dash_out], layout=widgets.Layout(padding="10px"))

    def _apply_dashboard(self, _) -> None:
        df = self._get_df(); target = self.dash_target.value; features = list(self.dash_features.value)[:4]
        if target == "(None)" or not features:
            self._notify(self.dash_out, "Sélectionnez une target et au moins une feature.", True); return
        with self.dash_out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, len(features), figsize=(len(features)*6, 5))
            if len(features) == 1: axes = [axes]
            target_vals = df[target].dropna().unique()
            is_binary = len(target_vals) == 2
            rate_series = (df[target] == sorted(target_vals)[-1]).astype(int) if is_binary else pd.to_numeric(df[target], errors="coerce")
            overall_rate = rate_series.mean()
            for ax, feat in zip(axes, features):
                is_num = pd.api.types.is_numeric_dtype(df[feat]) and df[feat].nunique() >= 20
                if is_num:
                    if is_binary: sns.histplot(data=df, x=feat, hue=target, multiple="layer", alpha=0.5, ax=ax)
                    else: sns.scatterplot(data=df, x=feat, y=target, ax=ax, alpha=0.5)
                    ax.set_title(f"{feat} vs {target}")
                else:
                    df_t = pd.DataFrame({"f": df[feat].astype(str), "r": rate_series}).dropna()
                    if df[feat].nunique() > 30:
                        top = df_t["f"].value_counts().nlargest(15).index
                        df_t = df_t[df_t["f"].isin(top)]
                    rate_by_cat = df_t.groupby("f")["r"].mean().sort_values()
                    ax.barh(range(len(rate_by_cat)), rate_by_cat.values, color="#a78bfa")
                    ax.set_yticks(range(len(rate_by_cat))); ax.set_yticklabels(rate_by_cat.index)
                    if overall_rate is not None: ax.axvline(overall_rate, color="#f28859", linestyle="--", label="Overall mean"); ax.legend()
                    ax.set_title(f"Target Rate by {feat}")
            plt.tight_layout(); plt.show()

    # ── Manage tab ────────────────────────────────────────────────────────────
    def _build_manage_tab(self) -> None:
        cols = list(self._get_df().columns)
        enc_cfg = self.state.config.get("encoding", {})
        tabular_types = list(enc_cfg.get("tabular", enc_cfg).keys()) or ["numeric","categorical","binary","datetime","text","id_like"]
        manage_ops = self.state.config.get("feature_engineering", {}).get("manage_actions", ["Set Type (Meta)","Duplicate","Delete"])
        self.manage_col      = widgets.Dropdown(options=cols, description="Column:", layout=styles.LAYOUT_DD)
        self.manage_action   = widgets.Dropdown(options=manage_ops, value=manage_ops[0], description="Action:", layout=widgets.Layout(width="250px"))
        self.manage_type     = widgets.Dropdown(options=tabular_types, value=tabular_types[0], description="Type:", layout=styles.LAYOUT_BTN_LARGE)
        self.manage_new_name = widgets.Text(description="New Name:", placeholder="for duplication", layout=widgets.Layout(width="200px", display="none"))
        self.manage_btn      = widgets.Button(description="Apply Action", button_style="warning")
        self.manage_out      = widgets.Output()
        def _on_action(c):
            self.manage_new_name.layout.display = "block" if c["new"] == "Duplicate" else "none"
            self.manage_type.layout.display = "block" if c["new"] == "Set Type (Meta)" else "none"
        self.manage_action.observe(_on_action, names="value")
        self.manage_btn.on_click(self._apply_manage)
        self.tab_manage = widgets.VBox([
            styles.help_box("Gérer les colonnes : type, duplication, suppression.", "#eab308"),
            widgets.HBox([self.manage_col, self.manage_action, self.manage_type, self.manage_new_name]),
            widgets.HBox([self.manage_btn]), self.manage_out],
            layout=widgets.Layout(padding="10px"))

    def _apply_manage(self, _) -> None:
        df = self._get_df(); col = self.manage_col.value; action = self.manage_action.value
        try:
            if action == "Delete":
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                    self.tabular_datasets[self.current_ds] = df
                    self.state.log_step("Feature Eng", "Column deleted", {"col": col})
                    self._propagate_to_state(); self._refresh_columns()
                    self._notify(self.manage_out, f"Deleted '{col}'"); self._render_preview()
            elif action == "Duplicate":
                new_name = self.manage_new_name.value or f"{col}_copy"
                if new_name in df.columns:
                    self._notify(self.manage_out, f"'{new_name}' already exists.", True); return
                df[new_name] = df[col].copy()
                self.tabular_datasets[self.current_ds] = df
                self.state.log_step("Feature Eng", "Column duplicated", {"col": col, "new_col": new_name})
                self._propagate_to_state(); self._refresh_columns()
                self._notify(self.manage_out, f"Duplicated '{col}' → '{new_name}'"); self._render_preview()
            elif action == "Set Type (Meta)":
                new_type = self.manage_type.value; ds = self.current_ds
                if not hasattr(self.state, "meta"): self.state.meta = {}
                if ds not in self.state.meta: self.state.meta[ds] = {}
                if col not in self.state.meta[ds]: self.state.meta[ds][col] = {}
                self.state.meta[ds][col]["kind"] = new_type
                self.state.log_step("Feature Eng", "Type overridden", {"col": col, "new_type": new_type})
                self._notify(self.manage_out, f"Set type of '{col}' → '{new_type}'")
        except Exception as e:
            self._notify(self.manage_out, str(e), True)


def runner(state) -> FeatureEngUI:
    fe = FeatureEngUI(state)
    if hasattr(fe, "ui"):
        display(fe.ui)
    return fe
