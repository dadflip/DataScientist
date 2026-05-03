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

def dynamic_import(import_string):
    if not import_string: return None
    import importlib
    parts = import_string.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
class UltimateEncoder:
    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("Configuration not loaded. Please run Cell 1a (Config) first.")
            return
        self.config = state.config.get("encoding", {})
        self.all_datasets = getattr(state, 'data_cleaned', {})
        if not hasattr(state, 'meta'):
            state.meta = {}
        self.meta = state.meta
        cleaning_config = state.config.get("cleaning", {})
        self.outlier_options = [(opt["label"], opt["value"]) for opt in cleaning_config.get("outliers", [])]
        if not any(val == 'none' for _, val in self.outlier_options):
            self.outlier_options.insert(0, ("Do nothing", "none"))
        self.datasets = {k: v.copy() for k, v in self.all_datasets.items() if isinstance(v, pd.DataFrame)}
        self.non_tabular = {k: v for k, v in self.all_datasets.items() if not isinstance(v, pd.DataFrame)}
        self._sync_metadata()
        self._build_ui()
    def _sync_metadata(self):
        for ds_name, df in self.datasets.items():
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
    def _auto_suggest_outlier(self, df_col):
        is_num = pd.api.types.is_numeric_dtype(df_col)
        if not is_num:
            return 'none'
        return 'none'
    def _calc_outliers(self, df_col, o_act):
        if not pd.api.types.is_numeric_dtype(df_col):
            return 0, len(df_col)
        df_col = df_col.dropna()
        if len(df_col) == 0: return 0, 0
        if o_act == 'clip_iqr':
            q1, q3 = df_col.quantile(0.25), df_col.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = ((df_col < lower) | (df_col > upper)).sum()
            return n_outliers, len(df_col)
        elif o_act == 'drop_zscore':
            std = df_col.std()
            if std > 0:
                n_outliers = (((df_col - df_col.mean()) / std).abs() > 3).sum()
                return n_outliers, len(df_col)
        return 0, len(df_col)
    def _apply_outliers(self, df, row_widgets):
        df_new = df.copy()
        for col, wd in row_widgets.items():
            o_act = wd['outlier_dd'].value
            flag_it = wd['flag_cb'].value if 'flag_cb' in wd else False
            if o_act != 'none' and col in df_new.columns:
                if flag_it:
                    if o_act == 'clip_iqr':
                        q1, q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        df_new[f"{col}_is_outlier"] = ((df_new[col] < lower) | (df_new[col] > upper)).astype(int)
                    elif o_act == 'drop_zscore':
                        std = df_new[col].std()
                        if std > 0:
                            df_new[f"{col}_is_outlier"] = (((df_new[col] - df_new[col].mean()) / std).abs() > 3).astype(int)
                if o_act == 'clip_iqr':
                    q1, q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
                    iqr = q3 - q1
                    df_new[col] = df_new[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
                elif o_act == 'drop_zscore':
                    std = df_new[col].std()
                    if std > 0: df_new = df_new[((df_new[col] - df_new[col].mean()) / std).abs() <= 3]
        return df_new
    def _get_encoded_df(self):
        if not getattr(self, 'current_ds', None): return None
        df = self.datasets[self.current_ds].copy()
        if not hasattr(self, '_tab_enc_widgets'): return df
        for col, wd in self._tab_enc_widgets.items():
            enc_value = wd['dd'].value
            kind = wd['kind']
            options_config = self.config.get("tabular", {}).get(kind, [])
            opt_info = next((opt for opt in options_config if opt["value"] == enc_value), None)
            if col not in df.columns: continue
            if enc_value == 'drop': 
                df.drop(columns=[col], inplace=True)
            elif opt_info and "code" in opt_info and opt_info["code"] and opt_info["code"] != "# passthrough \u2014 no transformation":
                loc_env = {"df": df, "col": col, "params": opt_info.get("params", {})}
                try:
                    exec(opt_info["code"], globals(), loc_env)
                    df = loc_env["df"]
                except Exception as e:
                    print(f"Error applying {enc_value} to {col}: {e}")
        return df
    def _plot_outliers(self, df_col, col_name, o_act):
        if not hasattr(self, 'outlier_plot_out'): return
        with self.outlier_plot_out:
            from IPython.display import clear_output; clear_output(wait=True)
            import matplotlib.pyplot as plt
            s = df_col.dropna()
            if len(s) == 0:
                print(f"Column '{col_name}' is empty or contains only NaNs.")
                return
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(s, bins=50, color='#3b82f6', alpha=0.6, label='Data Distribution')
            median = s.median()
            ax.axvline(median, color='#10b981', linestyle='-', linewidth=2, label='Median')
            if o_act == 'clip_iqr' or (o_act == 'none' and len(self.outlier_options) > 1 and 'clip_iqr' in [x[1] for x in self.outlier_options]):
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                color = '#ef4444' if o_act == 'clip_iqr' else '#94a3b8'
                label_suffix = '' if o_act == 'clip_iqr' else ' (Suggested)'
                ax.axvline(lower, color=color, linestyle='--', linewidth=2, label=f'Lower Bound (1.5 IQR){label_suffix}')
                ax.axvline(upper, color=color, linestyle='--', linewidth=2, label=f'Upper Bound (1.5 IQR){label_suffix}')
                if o_act == 'clip_iqr':
                    ax.axvspan(s.min(), lower, color=color, alpha=0.1)
                    ax.axvspan(upper, s.max(), color=color, alpha=0.1)
            elif o_act == 'drop_zscore':
                mean, std = s.mean(), s.std()
                if std > 0:
                    lower, upper = mean - 3 * std, mean + 3 * std
                    ax.axvline(lower, color='#f59e0b', linestyle='--', linewidth=2, label='Lower Bound (3 Std)')
                    ax.axvline(upper, color='#f59e0b', linestyle='--', linewidth=2, label='Upper Bound (3 Std)')
                    ax.axvspan(s.min(), lower, color='#f59e0b', alpha=0.1)
                    ax.axvspan(upper, s.max(), color='#f59e0b', alpha=0.1)
            ax.set_title(f"Distribution & Outlier Limits for '{col_name}'", fontsize=12, fontweight='bold', color='#334155')
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()
            plt.show()
    def _build_ui(self):
        tabs_children = []
        if self.datasets:
            self.ds_selector = widgets.Dropdown(options=list(self.datasets.keys()), description='Dataset:')
            self.outlier_timing = widgets.Dropdown(options=['Before Encoding', 'After Encoding'], value='Before Encoding', description='Outliers:', layout=widgets.Layout(width='350px'))
            help_text = styles.help_box(
                "<b style='font-size: 1.05em;'>Encoding & Outliers Guide</b><br>"
                "Configure outliers per column and choose when to apply them using the dropdown above.<br>"
                "<ul style='margin-top: 5px; margin-bottom: 0; padding-left: 20px;'>"
                "<li style='margin-bottom: 4px;'><b>Before or After?</b> It is highly recommended to handle outliers <b>Before</b> encoding. Extreme values severely distort scalers like <i>StandardScaler</i> or <i>MinMax</i> by skewing the mean and compressing the normal data range. Capping them first ensures robust scaling. Applying <i>After</i> is only useful if your encodings generate new unconstrained numeric metrics (e.g. TF-IDF frequencies) that need subsequent regularisation.</li>"
                "<li style='margin-bottom: 4px;'><b>Does encoding handle outliers implicitly?</b> Usually <b>No</b>. Standard scalers and One-Hot encoding leave outliers unmanaged. However, some methods do: <i>Binning</i> pushes outliers into the outermost bins, and <i>RobustScaler</i> naturally resists them by using the IQR instead of min/max.</li>"
                "<li><b>Are outliers significant?</b> Yes! In fraud detection, medical anomalies, or whale customers, extreme values ARE the signal. To preserve this information while still scaling the feature properly, check <b>'Create Indicator'</b>. This will extract the outlier status into a separate <code>{col}_is_outlier</code> boolean column before the original column is clipped or dropped.</li>"
                "</ul>",
                "#10b981"
            )
            selectors = widgets.HBox([self.ds_selector, self.outlier_timing], layout=widgets.Layout(margin='0 0 10px 0', gap='20px'))
            self.ds_selector.observe(self._on_tab_ds_change, names='value')
            self.outlier_timing.observe(self._build_outliers_table, names='value')
            self.current_ds = self.ds_selector.value
            self.enc_container = widgets.VBox()
            self.outlier_container = widgets.VBox()
            self.outlier_plot_out = widgets.Output()
            self.tab_out = widgets.Output()
            btn_apply = widgets.Button(description='Execute Pipeline', button_style=styles.BTN_PRIMARY, icon='cogs', layout=styles.LAYOUT_BTN_LARGE)
            btn_apply.on_click(self._apply_tabular)
            self._on_tab_ds_change(None)
            tabs_children.append(widgets.VBox([selectors, help_text, self.enc_container, self.outlier_container, self.outlier_plot_out, widgets.HBox([btn_apply], layout=widgets.Layout(margin='20px 0 10px 0')), self.tab_out]))
        else:
            tabs_children.append(widgets.HTML("<div style='padding:16px;'>No tabular data out.</div>"))
        tabs_children.append(widgets.HTML("<div style='padding:16px;'>Non-tabular data passed through.</div>"))
        tabs = widgets.Tab(children=tabs_children)
        tabs.set_title(0, 'Tabular')
        tabs.set_title(1, 'Non-Tabular')
        header = widgets.HTML(styles.card_html("Encode", "Encoding & Outliers", ""))
        top_bar = widgets.HBox(
            [header],
            layout=widgets.Layout(
                align_items='center',
                margin='0 0 12px 0',
                padding='0 0 10px 0',
                border_bottom='2px solid #ede9fe'
            )
        )
        self.ui = widgets.VBox(
            [top_bar, tabs],
            layout=widgets.Layout(
                width='100%', max_width='1000px',
                border='1px solid #e5e7eb',
                padding='18px',
                border_radius='10px',
                background_color='#ffffff'
            )
        )
    def _on_tab_ds_change(self, change):
        if change: self.current_ds = change['new']
        if not self.current_ds: return
        self._build_enc_table()
        self._build_outliers_table()
    def _build_enc_table(self):
        df = self.datasets[self.current_ds]
        self._tab_enc_widgets = {}
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px; font-weight:bold; color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:250px; font-weight:bold; color:#475569;'>Encoding Action</div>")
        ], layout=widgets.Layout(border_bottom='2px solid #cbd5e1', padding='0 0 5px 0', margin='0 0 10px 0'))
        rows = [headers]
        for col in df.columns:
            kind = self.meta[self.current_ds][col]['kind'] if self.current_ds in self.meta and col in self.meta[self.current_ds] else 'categorical'
            tabular_config = self.config.get("tabular", self.config)
            options_config = tabular_config.get(kind, [])
            opts = [(opt["label"], opt["value"]) for opt in options_config]
            if not opts:
                opts = [("Passthrough", "none"), ("Drop column", "drop")]
            if not any(o[1] == 'none' for o in opts):
                opts.insert(0, ("Passthrough", "none"))
            if not any(o[1] == 'drop' for o in opts):
                opts.append(("Drop column", "drop"))
            if kind == 'id_like':
                default_val = 'drop'
            elif kind == 'categorical':
                default_val = next((o[1] for o in opts if 'onehot' in o[1]), opts[0][1])
            elif kind == 'numeric':
                default_val = next((o[1] for o in opts if o[1] in ['std', 'minmax', 'robust']), opts[0][1])
            elif kind == 'datetime':
                default_val = next((o[1] for o in opts if 'extract' in o[1] or 'epoch' in o[1]), opts[0][1])
            elif kind == 'binary':
                default_val = next((o[1] for o in opts if o[1] in ['label', 'bool_map']), opts[0][1])
            else:
                default_val = 'none'
            if not any(o[1] == default_val for o in opts):
                default_val = opts[0][1]
            lbl_col = widgets.HTML(f"<div style='width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-top:4px;' title='{col}'><b>{col}</b> <span style='color:#94a3b8; font-size:0.8em;'>[{kind}]</span></div>")
            enc_dd = widgets.Dropdown(options=opts, value=default_val, layout=widgets.Layout(width='240px'))
            def _on_enc_change(change):
                if self.outlier_timing.value == 'After Encoding':
                    self._build_outliers_table()
            enc_dd.observe(_on_enc_change, names='value')
            self._tab_enc_widgets[col] = {'dd': enc_dd, 'kind': kind}
            rows.append(widgets.HBox([lbl_col, enc_dd], layout=widgets.Layout(padding='4px 0', border_bottom='1px solid #f1f5f9')))
        self.enc_container.children = [widgets.HTML("<h4 style='color:#3b82f6;'>Encoding Rules</h4>")] + rows
    def _build_outliers_table(self, change=None):
        if not getattr(self, 'current_ds', None): return
        is_after = self.outlier_timing.value == 'After Encoding'
        df = self._get_encoded_df() if is_after else self.datasets[self.current_ds]
        self._tab_outlier_widgets = {}
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px; font-weight:bold; color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:200px; font-weight:bold; color:#475569;'>Outlier Action</div>"),
            widgets.HTML("<div style='width:150px; font-weight:bold; color:#475569;'>Detected</div>"),
            widgets.HTML("<div style='width:150px; font-weight:bold; color:#475569;'>Create Indicator</div>"),
            widgets.HTML("<div style='width:90px; font-weight:bold; color:#475569;'>Preview</div>")
        ], layout=widgets.Layout(border_bottom='2px solid #cbd5e1', padding='0 0 5px 0', margin='0 0 10px 0'))
        rows = [headers]
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_label = f"{col} (Encoded)" if is_after else col
            lbl_col = widgets.HTML(f"<div style='width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-top:4px;' title='{col}'><b>{col_label}</b></div>")
            sug_out = self._auto_suggest_outlier(df[col])
            if not any(sug_out == val for _, val in self.outlier_options): sug_out = 'none'
            out_dd = widgets.Dropdown(options=self.outlier_options, value=sug_out, layout=widgets.Layout(width='180px', margin='0 20px 0 0'))
            lbl_ratio = widgets.HTML(f"<div style='width:150px; color:#64748b; padding-top:4px;'>-</div>")
            flag_cb = widgets.Checkbox(value=False, description='Significatif', indent=False, layout=styles.LAYOUT_BTN_STD)
            btn_preview = widgets.Button(description='Preview', button_style=styles.BTN_INFO, icon='bar-chart', layout=widgets.Layout(width='80px', padding='0'))
            def _update_ratio(change, df_col=df[col], lbl=lbl_ratio, cb=flag_cb):
                if change['new'] == 'none':
                    lbl.value = f"<div style='width:150px; color:#64748b; padding-top:4px;'>-</div>"
                    cb.disabled = True
                    cb.value = False
                    return
                cb.disabled = False
                n_out, t_out = self._calc_outliers(df_col, change['new'])
                pct = (n_out/t_out)*100 if t_out > 0 else 0
                color = "#ef4444" if pct > 5 else "#f59e0b" if pct > 1 else "#10b981"
                lbl.value = f"<div style='width:150px; color:{color}; padding-top:4px;'>{n_out} / {t_out} ({pct:.1f}%)</div>"
            def _on_preview_click(b, c=col, dd=out_dd):
                is_after_clk = self.outlier_timing.value == 'After Encoding'
                curr_df = self._get_encoded_df() if is_after_clk else self.datasets[self.current_ds]
                self._plot_outliers(curr_df[c], c, dd.value)
            btn_preview.on_click(_on_preview_click)
            out_dd.observe(_update_ratio, names='value')
            if sug_out != 'none':
                _update_ratio({'new': sug_out})
            else:
                flag_cb.disabled = True
            self._tab_outlier_widgets[col] = {'outlier_dd': out_dd, 'flag_cb': flag_cb}
            rows.append(widgets.HBox([lbl_col, out_dd, lbl_ratio, flag_cb, btn_preview], layout=widgets.Layout(padding='4px 0', border_bottom='1px solid #f1f5f9', align_items='center')))
        step_title = "Outliers Rules (Applied After Encoding)" if is_after else "Outliers Rules (Applied Before Encoding)"
        self.outlier_container.children = [widgets.HTML(f"<h4 style='color:#eab308; margin-top:20px;'>{step_title}</h4>")] + rows
    def _apply_tabular(self, b):
        with self.tab_out:
            from IPython.display import clear_output; clear_output()
            if not getattr(self, 'current_ds', None): return
            df = self.datasets[self.current_ds].copy()
            params = {}
            ops_count = 0
            timing = self.outlier_timing.value
            if timing == 'Before Encoding':
                df = self._apply_outliers(df, self._tab_outlier_widgets)
                ops_count += sum(1 for wd in self._tab_outlier_widgets.values() if wd['outlier_dd'].value != 'none')
            for col, wd in self._tab_enc_widgets.items():
                enc_value = wd['dd'].value
                if enc_value != 'none' and enc_value != 'drop':
                    ops_count += 1
                params[col] = {"encoding": enc_value}
                kind = wd['kind']
                options_config = self.config.get("tabular", {}).get(kind, [])
                opt_info = next((opt for opt in options_config if opt["value"] == enc_value), None)
                if col not in df.columns: continue
                if enc_value == 'drop': 
                    df.drop(columns=[col], inplace=True)
                elif opt_info and "code" in opt_info and opt_info["code"] and opt_info["code"] != "# passthrough \u2014 no transformation":
                    loc_env = {"df": df, "col": col, "params": opt_info.get("params", {})}
                    try:
                        exec(opt_info["code"], globals(), loc_env)
                        df = loc_env["df"]
                    except Exception as e:
                        error_msg = f"[Error] applying {enc_value} to {col}: {str(e)}"
                        print(error_msg)
                        import traceback
                        traceback.print_exc()
            if timing == 'After Encoding':
                df = self._apply_outliers(df, self._tab_outlier_widgets)
                ops_count += sum(1 for wd in self._tab_outlier_widgets.values() if wd['outlier_dd'].value != 'none')
            params["_outlier_timing"] = timing
            params["_outliers"] = {k: {"method": v['outlier_dd'].value, "flagged": v['flag_cb'].value if 'flag_cb' in v else False} 
                                   for k, v in self._tab_outlier_widgets.items() if v['outlier_dd'].value != 'none'}            
            self.state.data_encoded[self.current_ds] = df
            self.state.log_step("Data Encoding", "Tabular Encoded with Outliers", {"dataset": self.current_ds, "strategy": params, "outliers_timing": timing})
            status_html = styles.info_msg(
                f"Successfully applied {ops_count} operation(s) to '{self.current_ds}'.<br>"
                f"<ul style='margin-bottom:0; font-size:0.9em;'>"
                f"<li>Original rows/cols: {self.datasets[self.current_ds].shape[0]} / {self.datasets[self.current_ds].shape[1]}</li>"
                f"<li>Final rows/cols: {df.shape[0]} / {df.shape[1]}</li>"
                f"</ul>"
            )
            display(status_html)
            btn_eda = widgets.Button(description=' Visualiser (EDA Post-Encodage)', button_style=styles.BTN_INFO, icon='bar-chart', layout=widgets.Layout(width='300px', margin='15px 0'))
            eda_out = widgets.Output()
            def _show_eda(b):
                with eda_out:
                    from IPython.display import clear_output; clear_output()
                    try:
                        try:
                            UltimateEDA = globals()['UltimateEDA']
                        except KeyError:
                            print("UltimateEDA not found. Please run cell 3 (EDA) first.")
                            return
                        print(f"Loading EDA for {self.current_ds}...")
                        eda = UltimateEDA(self.state)
                        target_ds_key = f"[ENC] {self.current_ds}"
                        if target_ds_key in eda.all_datasets:
                            eda.ds_selector.value = target_ds_key
                        display(eda.ui)
                    except Exception as e:
                        print(f"Error loading EDA: {e}")
            btn_eda.on_click(_show_eda)
            display(btn_eda, eda_out)
    def _apply_non_tabular(self, b):
        for key, val in self.non_tabular.items():
            self.state.data_encoded[key] = val
def runner(state):
    encoder = UltimateEncoder(state)
    display(encoder.ui)
    return encoder
try:
    runner(state)
except NameError:
    pass