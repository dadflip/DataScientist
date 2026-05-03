import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

try:
    from ml_pipeline.cell_0d_styles import styles
except ImportError:
    if 'styles' in globals():
        styles = globals()['styles']
    else:
        print("styles not found in globals or via ml_pipeline, applying fallback")
        import importlib.util
        import sys
        import os
        spec = importlib.util.spec_from_file_location("styles", os.path.join(os.path.dirname(__file__), "cell_0d_styles.py"))
        styles = importlib.util.module_from_spec(spec)
        sys.modules["styles"] = styles
        spec.loader.exec_module(styles)

class AdvancedCleaner:
    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("[ERROR] Configuration not loaded. Please run Cell 1a (Config) first.")
            return
        self.config = state.config.get("cleaning", {})
        self.missing_options = [(opt["label"], opt["value"]) for opt in self.config.get("missing", [])]
        if not any(val == 'none' for _, val in self.missing_options):
            self.missing_options.insert(0, ("Do nothing", "none"))
        self.raw_datasets = state.data_raw
        self.original_datasets = {k: v.copy() for k, v in self.raw_datasets.items() if isinstance(v, pd.DataFrame)}
        self.current_datasets = {k: v.copy() for k, v in self.original_datasets.items()}
        self.current_ds = None
        self.row_widgets = {}
        if not hasattr(state, 'meta'):
            state.meta = {}
        self.meta = state.meta
        self._sync_metadata()
        self._build_ui()
    def _sync_metadata(self):
        for ds_name, df in self.original_datasets.items():
            orig_key = ds_name
            if orig_key not in self.meta:
                self.meta[orig_key] = {}
            for col in df.columns:
                if col not in self.meta[orig_key]:
                    s = df[col]
                    n_unq = s.nunique()
                    if pd.api.types.is_datetime64_any_dtype(s): kind = 'datetime'
                    elif pd.api.types.is_bool_dtype(s) or n_unq == 2: kind = 'binary'
                    elif pd.api.types.is_numeric_dtype(s):
                        kind = 'id_like' if n_unq / max(len(s), 1) > 0.95 else 'numeric'
                    else:
                        kind = 'categorical' if n_unq < 100 else 'text'
                    self.meta[orig_key][col] = {'kind': kind}
    def _build_ui(self):
        if not self.original_datasets:
            self.ui = styles.error_msg("No tabular data available for cleaning.")
            for k, v in self.raw_datasets.items():
                if not isinstance(v, pd.DataFrame):
                    self.state.data_cleaned[k] = v
            return
        header = widgets.HTML(styles.card_html("Clean", "Missing & Non-Significant Values Handling", ""))
        top_bar = widgets.HBox(
            [header],
            layout=widgets.Layout(
                align_items='center',
                margin='0 0 12px 0',
                padding='0 0 10px 0',
                border_bottom='2px solid #ede9fe'
            )
        )
        self.ds_selector = widgets.Dropdown(options=list(self.original_datasets.keys()), description='Dataset:', layout=styles.LAYOUT_DD_LONG)
        self.ds_selector.observe(self.on_ds_change, names='value')
        if list(self.original_datasets.keys()):
            self.current_ds = list(self.original_datasets.keys())[0]
        help_text = styles.help_box(
            "<b>Cleaning suggestions</b> are based on the global configuration.<br>"
            "<ul>"
            "<li>Use <i>Null representations</i> to convert non-significant values (e.g. <code>-1, unknown, ?</code>) to NaN before cleaning.</li>"
            "<li>If missing > 50%: we suggest 'Drop Column'.</li>"
            "<li>If missing < 5%: we suggest 'Drop Rows'.</li>"
            "<li>For Numeric columns with missing values: 'Mean' or 'Median' based on skew.</li>"
            "</ul>",
            "#10b981"
        )
        self.table_container = widgets.VBox(layout=widgets.Layout(width='100%', border='1px solid #e2e8f0', border_radius='6px', padding='10px', background_color='#f8fafc', margin='0 0 15px 0'))
        self.btn_apply = widgets.Button(description='Execute Cleaning', button_style=styles.BTN_PRIMARY, icon='magic', layout=styles.LAYOUT_BTN_LARGE)
        self.btn_apply.style.button_color = '#10b981'
        self.btn_apply.on_click(self._execute_cleaning)
        self.btn_reset = widgets.Button(description='Reset to Raw', button_style=styles.BTN_WARNING, icon='undo', layout=styles.LAYOUT_BTN_STD)
        self.btn_reset.on_click(self._reset_cleaning)
        self.out_logs = widgets.Output()
        main_content = widgets.VBox([
            self.ds_selector,
            help_text,
            self.table_container,
            widgets.HBox([self.btn_apply, self.btn_reset], layout=widgets.Layout(gap='15px')),
            self.out_logs
        ])
        self.ui = widgets.VBox(
            [top_bar, main_content],
            layout=widgets.Layout(
                width='100%', max_width='1000px',
                border='1px solid #e5e7eb',
                padding='18px',
                border_radius='10px',
                background_color='#ffffff'
            )
        )
        self.on_ds_change(None)
    def on_ds_change(self, change):
        if change and change.new:
            self.current_ds = change.new
        self._build_table()
    def _auto_suggest_missing(self, col, df, meta_info):
        pct_miss = meta_info.get('pct_miss', df[col].isna().mean() * 100)
        is_num = meta_info.get('kind') in ['numeric', 'timeseries'] or pd.api.types.is_numeric_dtype(df[col])
        if pct_miss == 0:
            return 'none'
        elif pct_miss > 50:
            return 'drop_cols'
        elif pct_miss < 5:
            return 'drop_rows'
        elif is_num:
            return 'median'
        else:
            return 'mode'
    def _build_table(self):
        if not self.current_ds: return
        df = self.original_datasets[self.current_ds]
        ds_key_meta = self.current_ds
        ds_meta = self.meta.get(ds_key_meta, {})
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px; font-weight:bold; color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:100px; font-weight:bold; color:#475569; text-align:right; padding-right:15px;'>% Missing</div>"),
            widgets.HTML("<div style='width:220px; font-weight:bold; color:#475569;'>Null representations (,-separated)</div>"),
            widgets.HTML("<div style='width:250px; font-weight:bold; color:#475569;'>Missing Action</div>")
        ], layout=widgets.Layout(border_bottom='2px solid #cbd5e1', padding='0 0 5px 0', margin='0 0 10px 0'))
        rows = [headers]
        self.row_widgets = {}
        for col in df.columns:
            col_meta = ds_meta.get(col, {})
            pct_miss = col_meta.get('pct_miss', df[col].isna().mean() * 100)
            is_num = col_meta.get('kind', '') in ['numeric'] or pd.api.types.is_numeric_dtype(df[col])
            sug_miss = self._auto_suggest_missing(col, df, col_meta)
            col_missing_options = []
            for lbl, val in self.missing_options:
                num_only_opts = ['mean', 'median', 'interpolate_linear', 'interpolate_time', 'knn', 'mice', 'zero']
                if not is_num and val in num_only_opts:
                    continue
                col_missing_options.append((lbl, val))
            if not any(sug_miss == val for _, val in col_missing_options): sug_miss = 'none'
            lbl_col = widgets.HTML(f"<div style='width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-top:4px;' title='{col}'><b>{col}</b> <span style='color:#94a3b8; font-size:0.8em;'>[{col_meta.get('kind', 'unk')}]</span></div>")
            color_miss = "#ef4444" if pct_miss > 20 else "#f59e0b" if pct_miss > 0 else "#22c55e"
            lbl_miss = widgets.HTML(f"<div style='width:100px; text-align:right; padding-right:15px; color:{color_miss}; padding-top:4px;'>{pct_miss:.1f}%</div>")
            txt_nulls = widgets.Text(placeholder="e.g. -1, unknown, ?", layout=widgets.Layout(width='200px', margin='0 20px 0 0'))
            dd_m = widgets.Dropdown(options=col_missing_options, value=sug_miss, layout=widgets.Layout(width='240px'))
            if pct_miss == 0:
                dd_m.disabled = True
            def _on_nulls_change(change, c=col, d=dd_m, l=lbl_miss, o_opts=col_missing_options, df_c=df[col].copy(), meta=col_meta):
                null_str = change['new'].strip()
                temp_s = df_c.copy()
                if null_str:
                    reps = [r.strip() for r in null_str.split(',') if r.strip()]
                    to_replace = []
                    for r in reps:
                        to_replace.append(r)
                        try: to_replace.append(int(r))
                        except ValueError: pass
                        try: to_replace.append(float(r))
                        except ValueError: pass
                    temp_s = temp_s.replace(to_replace, np.nan)
                
                new_pct = temp_s.isna().mean() * 100
                color_miss = "#ef4444" if new_pct > 20 else "#f59e0b" if new_pct > 0 else "#22c55e"
                l.value = f"<div style='width:100px; text-align:right; padding-right:15px; color:{color_miss}; padding-top:4px;'>{new_pct:.1f}%</div>"
                
                if new_pct > 0:
                    d.disabled = False
                    if d.value == 'none' and len(o_opts) > 1:
                        meta_temp = meta.copy()
                        meta_temp['pct_miss'] = new_pct
                        sug = self._auto_suggest_missing(c, pd.DataFrame({c: temp_s}), meta_temp)
                        if any(sug == val for _, val in o_opts):
                            d.value = sug
                        else:
                            d.value = o_opts[1][1]
                else:
                    d.disabled = True
                    d.value = 'none'
            txt_nulls.observe(_on_nulls_change, names='value')
            self.row_widgets[col] = {'is_num': is_num, 'missing': dd_m, 'null_reps': txt_nulls}
            rows.append(widgets.HBox([lbl_col, lbl_miss, txt_nulls, dd_m], layout=widgets.Layout(padding='4px 0', border_bottom='1px solid #f1f5f9')))
        self.table_container.children = rows
    def _reset_cleaning(self, b):
        with self.out_logs:
            from IPython.display import clear_output; clear_output()
            if self.current_ds:
                self.current_datasets[self.current_ds] = self.original_datasets[self.current_ds].copy()
                self._build_table()
                print(f"[INFO] Restored '{self.current_ds}' to original state.")
    def _execute_cleaning(self, b):
        with self.out_logs:
            from IPython.display import clear_output; clear_output()
            df_new = self.original_datasets[self.current_ds].copy()
            params = {}
            ops_count = 0
            for col, wgts in self.row_widgets.items():
                m_act = wgts['missing'].value
                null_str = wgts['null_reps'].value.strip()
                if m_act == 'none' and not null_str: continue
                params[col] = {"missing": m_act, "null_reps": null_str}
                ops_count += 1
                if null_str:
                    reps = [r.strip() for r in null_str.split(',') if r.strip()]
                    to_replace = []
                    for r in reps:
                        to_replace.append(r)
                        try:
                            to_replace.append(int(r))
                        except ValueError:
                            pass
                        try:
                            to_replace.append(float(r))
                        except ValueError:
                            pass
                    df_new[col] = df_new[col].replace(to_replace, np.nan)
                if m_act == 'drop_cols': df_new.drop(columns=[col], inplace=True)
                elif m_act == 'drop_rows': df_new.dropna(subset=[col], inplace=True)
                elif m_act == 'mean' and wgts['is_num']: df_new[col] = df_new[col].fillna(df_new[col].mean())
                elif m_act == 'median' and wgts['is_num']: df_new[col] = df_new[col].fillna(df_new[col].median())
                elif m_act == 'mode': 
                    modes = df_new[col].mode()
                    if not modes.empty: df_new[col] = df_new[col].fillna(modes.iloc[0])
                elif m_act == 'zero': df_new[col] = df_new[col].fillna(0 if wgts['is_num'] else '0')
                elif m_act == 'ffill': df_new[col] = df_new[col].ffill()
                elif m_act == 'bfill': df_new[col] = df_new[col].bfill()
            self.current_datasets[self.current_ds] = df_new
            self.state.data_cleaned[self.current_ds] = df_new
            self.state.log_step("Data Cleaning", "Cleaning Applied", {"dataset": self.current_ds, "operations": params})
            status_html = styles.info_msg(
                f"Successfully applied {ops_count} cleaning operation(s) to '{self.current_ds}'.<br>"
                f"<ul style='margin-bottom:0; font-size:0.9em;'>"
                f"<li>Original rows/cols: {self.original_datasets[self.current_ds].shape[0]} / {self.original_datasets[self.current_ds].shape[1]}</li>"
                f"<li>Cleaned rows/cols: {df_new.shape[0]} / {df_new.shape[1]}</li>"
                f"</ul>"
            )
            display(status_html)
def runner(state):
    cleaner = AdvancedCleaner(state)
    if hasattr(cleaner, 'ui'):
        display(cleaner.ui)
    return cleaner
try:
    runner(state)
except NameError:
    pass