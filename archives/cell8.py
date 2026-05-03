import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import math


# ══════════════════════════════════════════════════════════════════════════════
# CSS global injecté une seule fois pour les onglets multi-lignes
# ══════════════════════════════════════════════════════════════════════════════
_TAB_CSS_INJECTED = False

def _inject_tab_css():
    global _TAB_CSS_INJECTED
    if _TAB_CSS_INJECTED:
        return
    _TAB_CSS_INJECTED = True
    display(HTML("""
    <style>
    /* ── onglets Feature Eng : multi-lignes, texte complet ── */
    .fe-tabs .jupyter-widgets-tab-bar {
        display: flex !important;
        flex-wrap: wrap !important;
        border-bottom: 2px solid #e2e8f0 !important;
        gap: 2px !important;
    }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab {
        flex-shrink: 0 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        font-size: 0.80em !important;
        padding: 5px 12px !important;
        border-radius: 6px 6px 0 0 !important;
        border: 1px solid #e2e8f0 !important;
        border-bottom: none !important;
        background: #f8fafc !important;
        color: #64748b !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: background 0.15s, color 0.15s !important;
    }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab.p-mod-current {
        background: #ffffff !important;
        color: #1e293b !important;
        border-color: #cbd5e1 !important;
        border-bottom-color: #ffffff !important;
        font-weight: 700 !important;
    }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab:hover:not(.p-mod-current) {
        background: #f1f5f9 !important;
        color: #334155 !important;
    }
    /* fallback JupyterLab 4 / classic */
    .fe-tabs .p-TabBar-tabLabel {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    </style>
    """))


class FeatureEngUI:
    def __init__(self, state):
        self.state = state
        if not self.state.config:
            self.ui = styles.error_msg("[ERROR] Configuration not loaded. Please run Cell 1a (Config) first.")
            display(self.ui)
            return
        self.tabular_datasets = {k: v for k, v in self.state.data_raw.items() if isinstance(v, pd.DataFrame)}
        if not self.tabular_datasets:
            self.ui = styles.error_msg("No tabular datasets available for Feature Engineering.")
            display(self.ui)
            return
        self.current_ds = list(self.tabular_datasets.keys())[0]
        _inject_tab_css()
        self._build_ui()

    def _get_df(self):
        return self.tabular_datasets[self.current_ds]

    # ──────────────────────────────────────────────────────────────────────────
    # PROPAGATION : synchronise les colonnes FE vers data_raw et data_cleaned
    # ──────────────────────────────────────────────────────────────────────────
    def _propagate_to_state(self):
        """
        Après chaque modification (ajout / suppression de colonne),
        répercute le DataFrame courant dans :
          - state.data_raw[ds]          → source de vérité partagée
          - state.data_cleaned[ds]      → utilisé par la suite du pipeline
          - state.meta[ds]              → mise à jour des métadonnées de colonnes
        """
        ds = self.current_ds
        df = self.tabular_datasets[ds]

        # --- data_raw & data_cleaned ---
        self.state.data_raw[ds] = df
        if hasattr(self.state, 'data_cleaned'):
            self.state.data_cleaned[ds] = df.copy()

        # --- meta : ajouter les nouvelles colonnes, retirer les supprimées ---
        if not hasattr(self.state, 'meta'):
            self.state.meta = {}
        if ds not in self.state.meta:
            self.state.meta[ds] = {}
        meta_ds = self.state.meta[ds]

        for col in list(meta_ds.keys()):
            if col not in df.columns:
                del meta_ds[col]

        for col in df.columns:
            if col not in meta_ds:
                s = df[col]
                n_unq = s.nunique()
                if pd.api.types.is_datetime64_any_dtype(s):
                    kind = 'datetime'
                elif pd.api.types.is_bool_dtype(s) or n_unq == 2:
                    kind = 'binary'
                elif pd.api.types.is_numeric_dtype(s):
                    kind = 'id_like' if n_unq / max(len(s), 1) > 0.95 else 'numeric'
                else:
                    kind = 'categorical' if n_unq < 100 else 'text'
                meta_ds[col] = {'kind': kind}

    def _sync_targets(self, op_fn, op_label: str):
        """
        Rejoue l'opération op_fn(df_target) → df_result sur chaque dataset
        sélectionné dans self.sync_datasets (si self.sync_check.value est True).

        op_fn reçoit un DataFrame (copie du dataset cible) et doit retourner
        le DataFrame modifié (ou None si rien à faire).

        Affiche un rapport dans self.sync_status.
        """
        if not self.sync_check.value:
            return

        targets = list(self.sync_datasets.value)
        if not targets:
            return

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
                    # Propager vers state
                    self.state.data_raw[ds_name] = df_result
                    if hasattr(self.state, 'data_cleaned'):
                        self.state.data_cleaned[ds_name] = df_result.copy()
                    lines.append(
                        f"<span style='color:#10b981;'>✓ '{ds_name}' — {op_label}</span>")
                else:
                    lines.append(
                        f"<span style='color:#94a3b8;'>– '{ds_name}' — aucune modification.</span>")
            except Exception as e:
                lines.append(
                    f"<span style='color:#ef4444;'>⚠ '{ds_name}' — erreur : {e}</span>")

        if lines:
            with self.sync_status:
                clear_output(wait=True)
                display(HTML(
                    "<div style='font-size:0.79em; padding:4px 0;'>"
                    "<b style='color:#6366f1;'>Sync :</b> "
                    + "  ·  ".join(lines) + "</div>"))

    def _build_ui(self):
        header = widgets.HTML(styles.card_html("Feature Engineering", "Advanced Variable Laboratory", ""))
        top_bar = widgets.HBox(
            [header],
            layout=widgets.Layout(
                align_items='center',
                margin='0 0 12px 0',
                padding='0 0 10px 0',
                border_bottom='2px solid #ede9fe'
            )
        )
        self.ds_dd = widgets.Dropdown(
            options=list(self.tabular_datasets.keys()),
            value=self.current_ds,
            description="Dataset:",
            layout=styles.LAYOUT_DD_LONG
        )
        self.ds_dd.observe(self._on_ds_change, names='value')

        # ── Case à cocher : synchronisation multi-datasets ────────────────────
        other_ds = [k for k in self.tabular_datasets if k != self.current_ds]
        self.sync_datasets = widgets.SelectMultiple(
            options=other_ds,
            value=other_ds,          # tous cochés par défaut
            description='Sync vers :',
            layout=widgets.Layout(width='280px', height=f'{max(36, min(120, len(other_ds)*24))}px'),
            disabled=not bool(other_ds),
        )
        self.sync_check = widgets.Checkbox(
            value=bool(other_ds),
            description='Répliquer sur d\'autres datasets',
            layout=widgets.Layout(width='280px'),
            disabled=not bool(other_ds),
        )

        def _on_sync_toggle(change):
            self.sync_datasets.disabled = not change.new
        self.sync_check.observe(_on_sync_toggle, names='value')

        sync_help = widgets.HTML(
            "<div style='font-size:0.78em; color:#6b7280; background:#f8fafc; "
            "border:1px solid #e2e8f0; border-left:3px solid #6366f1; "
            "border-radius:4px; padding:7px 10px; margin:4px 0;'>"
            "<b>Synchronisation automatique</b> — quand cette option est activée, "
            "toute opération effectuée sur le dataset courant (ajout ou suppression de colonne) "
            "est <b>rejouée à l'identique</b> sur les datasets sélectionnés ci-dessus.<br>"
            "Cas d'usage typique : vous travaillez sur <code>train</code>, cochez <code>test</code> "
            "→ les features créées (<code>pdays==0</code>, <code>month_num</code>, etc.) "
            "apparaissent automatiquement dans <code>test</code> avec les mêmes paramètres, "
            "garantissant que les deux jeux ont <b>exactement les mêmes colonnes</b> "
            "avant l'étape d'encodage / modélisation.<br>"
            "<span style='color:#ef4444;'>Attention :</span> si une colonne source n'existe pas "
            "dans le dataset cible (ex. colonne absente de <code>test</code>), "
            "l'opération est ignorée pour ce dataset et un avertissement s'affiche."
            "</div>"
        )

        self.sync_status = widgets.Output()   # zone d'affichage des résultats de sync

        self._sync_box = widgets.VBox([
            widgets.HBox([self.sync_check, self.sync_datasets],
                          layout=widgets.Layout(align_items='flex-start', gap='12px')),
            sync_help,
            self.sync_status,
        ], layout=widgets.Layout(
            padding='8px 10px',
            margin='6px 0',
            border='1px solid #e0e7ff',
            border_radius='6px',
            background_color='#f5f3ff',
        ))

        # ── Onglets sans emojis, titres complets ──────────────────────────────
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

        self.tabs.children = [
            self.tab_preview,
            self.tab_math,
            self.tab_condition,
            self.tab_formula,
            self.tab_text,
            self.tab_date,
            self.tab_binning,
            self.tab_viz,
            self.tab_dashboard,
            self.tab_manage,
        ]
        # Titres sans emojis, texte complet
        titles = [
            "Data Preview",
            "Math & Logic",
            "Conditions",
            "Custom Formula",
            "Text Operations",
            "Date / Time",
            "Binning",
            "Visualization",
            "Target Dashboard",
            "Manage Columns",
        ]
        for i, t in enumerate(titles):
            self.tabs.set_title(i, t)

        # Ajout de la classe CSS pour le wrapping des onglets
        self.tabs.add_class("fe-tabs")

        main_content = widgets.VBox([
            self.ds_dd,
            self._sync_box,
            widgets.HTML("<hr style='border:1px solid #f1f5f9; margin:10px 0;'>"),
            self.tabs
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

    def _refresh_columns(self):
        cols = list(self._get_df().columns)
        self.math_col1.options = cols
        self.math_col2.options = ['(None)', '(Constant)'] + cols
        self.text_col.options = cols
        self.date_col.options = cols
        self.bin_col.options = cols
        self.viz_x.options = cols
        self.viz_y.options = cols
        self.viz_hue.options = ['(None)'] + cols
        self.dash_target.options = ['(None)'] + cols
        self.dash_features.options = cols
        self.cond_col.options = cols
        self.cond_then_col.options = ['(Constant)'] + cols
        self.cond_else_col.options = ['(Constant)'] + cols
        self._refresh_formula_col_list()
        if hasattr(self, 'manage_col'):
            self.manage_col.options = cols
        self._refresh_preview_col_selector()

    def _on_ds_change(self, change):
        if change.new:
            self.current_ds = change.new
            # Rafraîchir les options de sync (exclure le dataset courant)
            other_ds = [k for k in self.tabular_datasets if k != self.current_ds]
            self.sync_datasets.options = other_ds
            self.sync_datasets.value   = other_ds          # tout sélectionner par défaut
            self.sync_datasets.layout.height = f'{max(36, min(120, len(other_ds)*24))}px'
            self.sync_check.disabled   = not bool(other_ds)
            self.sync_datasets.disabled = not (bool(other_ds) and self.sync_check.value)
            self._refresh_columns()
            self.viz_out.clear_output()
            if hasattr(self, 'dash_out'):
                self.dash_out.clear_output()
            self._render_preview()

    def _notify(self, out_widget, msg, is_error=False):
        color = "#ef4444" if is_error else "#10b981"
        ext = "[ERROR]" if is_error else "[OK]"
        with out_widget:
            clear_output(wait=True)
            display(HTML(f"<div style='color:{color}; font-weight:bold; font-size:0.85em;'>{ext} {msg}</div>"))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Data Preview
    # ══════════════════════════════════════════════════════════════════════════
    def _build_preview_tab(self):
        df = self._get_df()
        cols = list(df.columns)

        self.preview_rows = widgets.IntSlider(
            value=20, min=5, max=500, step=5,
            description='Rows:', layout=widgets.Layout(width='320px')
        )
        self.preview_search = widgets.Text(
            placeholder='Filter rows: col==value  or  col>0  or  col.str.contains("x")',
            description='Filter:',
            layout=widgets.Layout(width='480px')
        )
        self.preview_col_select = widgets.SelectMultiple(
            options=cols,
            value=cols[:min(len(cols), 30)],
            description='Columns:',
            layout=widgets.Layout(width='320px', height='90px')
        )
        self.preview_col_all_btn = widgets.Button(
            description='All cols', button_style='info',
            layout=widgets.Layout(width='90px', height='28px')
        )
        self.preview_col_new_btn = widgets.Button(
            description='New cols only', button_style='',
            layout=widgets.Layout(width='110px', height='28px')
        )
        self.preview_highlight_new = widgets.Checkbox(
            value=True, description='Highlight new cols',
            layout=widgets.Layout(width='170px')
        )
        self.preview_show_stats = widgets.Checkbox(
            value=False, description='Show quick stats',
            layout=widgets.Layout(width='150px')
        )
        self.preview_sort_col = widgets.Dropdown(
            options=['(none)'] + cols, value='(none)',
            description='Sort by:', layout=widgets.Layout(width='220px')
        )
        self.preview_sort_asc = widgets.Checkbox(
            value=True, description='Ascending',
            layout=widgets.Layout(width='110px')
        )
        self.preview_refresh_btn = widgets.Button(
            description='Refresh', button_style='primary',
            layout=widgets.Layout(width='100px', height='32px')
        )
        self.preview_out = widgets.Output()
        self._preview_original_cols = set(cols)

        def _select_all(_):
            self.preview_col_select.value = list(self.preview_col_select.options)

        def _select_new(_):
            new_cols = [c for c in self._get_df().columns if c not in self._preview_original_cols]
            self.preview_col_select.value = new_cols if new_cols else list(self.preview_col_select.options)

        self.preview_col_all_btn.on_click(_select_all)
        self.preview_col_new_btn.on_click(_select_new)
        self.preview_refresh_btn.on_click(lambda _: self._render_preview())
        self.preview_rows.observe(lambda c: self._render_preview(), names='value')
        self.preview_search.observe(lambda c: self._render_preview(), names='value')
        self.preview_sort_col.observe(lambda c: self._render_preview(), names='value')
        self.preview_sort_asc.observe(lambda c: self._render_preview(), names='value')
        self.preview_highlight_new.observe(lambda c: self._render_preview(), names='value')
        self.preview_show_stats.observe(lambda c: self._render_preview(), names='value')
        self.preview_col_select.observe(lambda c: self._render_preview(), names='value')

        ctrl_row1 = widgets.HBox(
            [self.preview_rows, self.preview_sort_col, self.preview_sort_asc],
            layout=widgets.Layout(align_items='center', gap='10px', margin='0 0 6px 0')
        )
        ctrl_row2 = widgets.HBox([self.preview_search], layout=widgets.Layout(margin='0 0 6px 0'))
        col_sel_row = widgets.HBox(
            [self.preview_col_select,
             widgets.VBox([self.preview_col_all_btn, self.preview_col_new_btn],
                          layout=widgets.Layout(gap='4px', margin='0 0 0 6px'))],
            layout=widgets.Layout(align_items='flex-start', margin='0 0 6px 0')
        )
        ctrl_row3 = widgets.HBox(
            [self.preview_highlight_new, self.preview_show_stats, self.preview_refresh_btn],
            layout=widgets.Layout(align_items='center', gap='10px')
        )

        self.tab_preview = widgets.VBox([
            styles.help_box(
                "<b>Live table view</b> of the dataset. Use <b>Filter</b> to query rows "
                "(any valid pandas <code>.query()</code> expression). "
                "<b>Highlight new cols</b> marks columns added after tab was built. "
                "<b>New cols only</b> filtre rapidement sur les features créées.",
                "#0ea5e9"
            ),
            ctrl_row1, ctrl_row2, col_sel_row, ctrl_row3,
            self.preview_out
        ], layout=widgets.Layout(padding='10px'))

        self._render_preview()

    def _refresh_preview_col_selector(self):
        df = self._get_df()
        cols = list(df.columns)
        current_val = list(self.preview_col_select.value)
        self.preview_col_select.options = cols
        still_valid = [c for c in current_val if c in cols]
        new_cols = [c for c in cols if c not in self._preview_original_cols]
        merged = still_valid + [c for c in new_cols if c not in still_valid]
        self.preview_col_select.value = merged if merged else cols[:min(len(cols), 30)]
        self.preview_sort_col.options = ['(none)'] + cols

    def _render_preview(self):
        df = self._get_df().copy()
        selected_cols = list(self.preview_col_select.value)
        if not selected_cols:
            selected_cols = list(df.columns)

        filter_expr = self.preview_search.value.strip()
        filter_error = None
        if filter_expr:
            try:
                df = df.query(filter_expr)
            except Exception:
                try:
                    mask = df.eval(filter_expr)
                    df = df[mask]
                except Exception as e:
                    filter_error = str(e)

        sort_col = self.preview_sort_col.value
        if sort_col and sort_col != '(none)' and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=self.preview_sort_asc.value)

        n_rows = self.preview_rows.value
        view = df[selected_cols].head(n_rows)
        total_rows = len(df)
        n_cols = len(selected_cols)
        new_cols = set(self._get_df().columns) - self._preview_original_cols
        highlight = self.preview_highlight_new.value

        with self.preview_out:
            clear_output(wait=True)
            meta_parts = [
                f"<b>{total_rows:,}</b> rows",
                f"<b>{len(self._get_df().columns)}</b> total cols",
                f"showing <b>{n_cols}</b> col(s)",
                f"<b>{len(new_cols)}</b> new col(s)",
            ]
            if filter_expr and not filter_error:
                meta_parts.append(f"<span style='color:#0ea5e9;'>filter active ({total_rows:,} rows match)</span>")
            if filter_error:
                meta_parts.append(f"<span style='color:#ef4444;'>filter error: {filter_error}</span>")

            display(HTML(
                "<div style='font-size:0.82em; color:#64748b; margin-bottom:8px; "
                "padding:6px 10px; background:#f8fafc; border-radius:6px; "
                "border:1px solid #e2e8f0;'>"
                + "  ·  ".join(meta_parts) + "</div>"
            ))

            table_styles = [
                {'selector': 'thead th',
                 'props': [('background-color', '#f1f5f9'), ('color', '#374151'),
                           ('font-size', '0.8em'), ('font-weight', '600'),
                           ('border-bottom', '2px solid #e2e8f0'),
                           ('padding', '6px 10px'), ('white-space', 'nowrap')]},
                {'selector': 'td',
                 'props': [('font-size', '0.82em'), ('padding', '4px 10px'),
                           ('border-bottom', '1px solid #f1f5f9'),
                           ('max-width', '200px'), ('overflow', 'hidden'),
                           ('text-overflow', 'ellipsis'), ('white-space', 'nowrap')]},
                {'selector': 'tr:hover td',
                 'props': [('background-color', '#f0f9ff')]},
            ]

            if highlight and new_cols:
                def _style_new(col):
                    if col.name in new_cols:
                        return ['background-color: #fef9c3; font-weight: 600;'] * len(col)
                    return [''] * len(col)
                styled = view.style.apply(_style_new, axis=0).set_table_styles(table_styles).format(precision=4, na_rep='—')
            else:
                styled = view.style.set_table_styles(table_styles).format(precision=4, na_rep='—')
            display(styled)

            if self.preview_show_stats.value:
                display(HTML("<br><b style='font-size:0.85em; color:#374151;'>Quick stats — selected columns :</b>"))
                num_cols_sel = view.select_dtypes(include=np.number).columns.tolist()
                if num_cols_sel:
                    display(view[num_cols_sel].describe().T.style.format(precision=3).set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#f1f5f9'), ('font-size', '0.78em'), ('padding', '4px 8px')]},
                        {'selector': 'td', 'props': [('font-size', '0.78em'), ('padding', '3px 8px')]},
                    ]))
                cat_cols_sel = view.select_dtypes(exclude=np.number).columns.tolist()
                if cat_cols_sel:
                    display(HTML("<br><b style='font-size:0.82em; color:#374151;'>Categorical :</b>"))
                    display(view[cat_cols_sel].describe().T.style.set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#f1f5f9'), ('font-size', '0.78em'), ('padding', '4px 8px')]},
                        {'selector': 'td', 'props': [('font-size', '0.78em'), ('padding', '3px 8px')]},
                    ]))

            if total_rows > n_rows:
                display(HTML(
                    f"<div style='font-size:0.78em; color:#94a3b8; margin-top:8px; text-align:right;'>"
                    f"Showing {n_rows} of {total_rows:,} rows</div>"
                ))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Math & Logic
    # ══════════════════════════════════════════════════════════════════════════
    def _build_math_tab(self):
        df = self._get_df()
        cols = list(df.columns)
        self.math_col1 = widgets.Dropdown(options=cols, description='Col A:', layout=styles.LAYOUT_DD)
        math_ops = self.state.config.get("feature_engineering", {}).get("math_operations", [])
        self.math_op = widgets.Dropdown(
            options=math_ops, value=math_ops[0] if math_ops else '+',
            description='Op:', layout=styles.LAYOUT_BTN_LARGE
        )
        self.math_col2 = widgets.Dropdown(options=['(None)', '(Constant)'] + cols, description='Col B:', layout=styles.LAYOUT_DD)
        self.math_const = widgets.FloatText(value=0, description='Const:', layout=widgets.Layout(width='200px', display='none'))
        self.math_new_col = widgets.Text(description='New Name:', placeholder='result_col', layout=styles.LAYOUT_DD)
        self.math_btn = widgets.Button(description='Apply Math', button_style=styles.BTN_PRIMARY)
        self.math_out = widgets.Output()

        def _on_op_change(change):
            op = change.new
            if op in ['log(A)', 'exp(A)', 'sqrt(A)', 'A^2', 'Abs(A)']:
                self.math_col2.layout.display = 'none'
                self.math_const.layout.display = 'none'
            else:
                self.math_col2.layout.display = 'block'
                if self.math_col2.value == '(Constant)':
                    self.math_const.layout.display = 'block'

        def _on_col2_change(change):
            self.math_const.layout.display = 'block' if change.new == '(Constant)' else 'none'

        self.math_op.observe(_on_op_change, names='value')
        self.math_col2.observe(_on_col2_change, names='value')
        self.math_btn.on_click(self._apply_math)

        self.tab_math = widgets.VBox([
            styles.help_box("Perform mathematical operations on numerical columns. Create ratios, interactions, or transformations (log/sqrt).", "#8b5cf6"),
            widgets.HBox([self.math_col1, self.math_op, self.math_col2, self.math_const]),
            widgets.HBox([self.math_new_col, self.math_btn]),
            self.math_out
        ], layout=widgets.Layout(padding='10px'))

    def _apply_math(self, _):
        df = self._get_df()
        col1 = self.math_col1.value
        col2 = self.math_col2.value
        op = self.math_op.value
        new_name = self.math_new_col.value or f"{col1}_new"
        const_val = self.math_const.value
        try:
            val1 = pd.to_numeric(df[col1], errors='coerce')
            if op in ['log(A)', 'exp(A)', 'sqrt(A)', 'A^2', 'Abs(A)']:
                if op == 'log(A)':    res = np.log1p(val1)
                elif op == 'exp(A)':  res = np.exp(val1)
                elif op == 'sqrt(A)': res = np.sqrt(np.maximum(val1, 0))
                elif op == 'A^2':     res = val1 ** 2
                elif op == 'Abs(A)':  res = np.abs(val1)
            else:
                val2 = const_val if col2 == '(Constant)' else pd.to_numeric(df[col2], errors='coerce')
                if op == '+':        res = val1 + val2
                elif op == '-':      res = val1 - val2
                elif op == '*':      res = val1 * val2
                elif op == '/':      res = val1 / val2.replace(0, np.nan)
                elif op == 'A^B/C':  res = val1 ** val2
                elif op == 'Modulo': res = val1 % val2
            self.tabular_datasets[self.current_ds][new_name] = res
            self.state.log_step("Feature Eng", "Math Operation applied", {"op": op, "new_col": new_name})
            self._propagate_to_state()

            # ── sync ─────────────────────────────────────────────────────────
            def _math_sync(df_t):
                v1 = pd.to_numeric(df_t[col1], errors='coerce') if col1 in df_t.columns else None
                if v1 is None:
                    raise KeyError(f"colonne '{col1}' absente")
                if op in ['log(A)', 'exp(A)', 'sqrt(A)', 'A^2', 'Abs(A)']:
                    if op == 'log(A)':    r = np.log1p(v1)
                    elif op == 'exp(A)':  r = np.exp(v1)
                    elif op == 'sqrt(A)': r = np.sqrt(np.maximum(v1, 0))
                    elif op == 'A^2':     r = v1 ** 2
                    elif op == 'Abs(A)':  r = np.abs(v1)
                else:
                    if col2 == '(Constant)':
                        v2 = const_val
                    elif col2 in df_t.columns:
                        v2 = pd.to_numeric(df_t[col2], errors='coerce')
                    else:
                        raise KeyError(f"colonne '{col2}' absente")
                    if op == '+':        r = v1 + v2
                    elif op == '-':      r = v1 - v2
                    elif op == '*':      r = v1 * v2
                    elif op == '/':      r = v1 / (v2.replace(0, np.nan) if hasattr(v2,'replace') else (v2 or np.nan))
                    elif op == 'A^B/C':  r = v1 ** v2
                    elif op == 'Modulo': r = v1 % v2
                df_t[new_name] = r
                return df_t
            self._sync_targets(_math_sync, f"Math {op} → '{new_name}'")

            self._refresh_columns()
            self._notify(self.math_out, f"Created column '{new_name}'")
            self._render_preview()
        except Exception as e:
            self._notify(self.math_out, str(e), True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Conditions
    # ══════════════════════════════════════════════════════════════════════════
    def _build_condition_tab(self):
        df = self._get_df()
        cols = list(df.columns)

        self.cond_col = widgets.Dropdown(options=cols, description='Column:', layout=styles.LAYOUT_DD)
        self.cond_op = widgets.Dropdown(
            options=['==', '!=', '>', '>=', '<', '<=', 'isin', 'not isin',
                     'is null', 'is not null', 'contains (str)', 'startswith', 'endswith'],
            description='Operator:', layout=widgets.Layout(width='200px')
        )
        self.cond_val = widgets.Text(description='Value:', placeholder='e.g. -1  or  val1,val2', layout=widgets.Layout(width='280px'))
        self.cond_then_col = widgets.Dropdown(options=['(Constant)'] + cols, value='(Constant)', description='THEN col:', layout=styles.LAYOUT_DD)
        self.cond_then_val = widgets.Text(value='1', description='THEN val:', layout=widgets.Layout(width='160px'))
        self.cond_else_col = widgets.Dropdown(options=['(Constant)'] + cols, value='(Constant)', description='ELSE col:', layout=styles.LAYOUT_DD)
        self.cond_else_val = widgets.Text(value='0', description='ELSE val:', layout=widgets.Layout(width='160px'))
        self.cond_combine = widgets.Dropdown(options=['AND', 'OR'], value='AND', description='Combine:', layout=widgets.Layout(width='150px'))
        self.cond_extra_rows = []
        self.cond_extra_box = widgets.VBox([])

        add_cond_btn = widgets.Button(description='+ Add condition', button_style='info', layout=widgets.Layout(width='150px', height='28px'))
        rem_cond_btn = widgets.Button(description='- Remove last',  button_style='warning', layout=widgets.Layout(width='130px', height='28px'))

        def _add_row(_):
            row_col = widgets.Dropdown(options=cols, layout=widgets.Layout(width='200px'))
            row_op  = widgets.Dropdown(
                options=['==', '!=', '>', '>=', '<', '<=', 'isin', 'not isin',
                         'is null', 'is not null', 'contains (str)', 'startswith', 'endswith'],
                layout=widgets.Layout(width='180px')
            )
            row_val = widgets.Text(placeholder='value', layout=widgets.Layout(width='220px'))
            self.cond_extra_rows.append((row_col, row_op, row_val))
            self.cond_extra_box.children = [widgets.HBox([r[0], r[1], r[2]]) for r in self.cond_extra_rows]

        def _rem_row(_):
            if self.cond_extra_rows:
                self.cond_extra_rows.pop()
                self.cond_extra_box.children = [widgets.HBox([r[0], r[1], r[2]]) for r in self.cond_extra_rows]

        add_cond_btn.on_click(_add_row)
        rem_cond_btn.on_click(_rem_row)

        self.cond_map_text = widgets.Textarea(
            placeholder='jan:1, feb:2, mar:3\n(leave empty to use IF/THEN above)',
            description='Value Map:', layout=widgets.Layout(width='460px', height='80px')
        )
        self.cond_new_col = widgets.Text(description='New col:', placeholder='flag_col', layout=styles.LAYOUT_DD)
        self.cond_btn = widgets.Button(description='Apply Condition', button_style=styles.BTN_PRIMARY)
        self.cond_out = widgets.Output()
        self.cond_btn.on_click(self._apply_condition)

        self.tab_condition = widgets.VBox([
            styles.help_box(
                "<b>IF / THEN / ELSE :</b> build binary flags or value columns from one or more conditions.<br>"
                "<b>Combine :</b> AND / OR when multiple condition rows are active.<br>"
                "<b>Value Map :</b> if filled, ignores IF/THEN and maps existing values "
                "(e.g. <code>jan:1, feb:2</code>). Unmapped → NaN.",
                "#f97316"
            ),
            widgets.HTML("<b style='font-size:0.85em;color:#6b7280;'>Primary condition</b>"),
            widgets.HBox([self.cond_col, self.cond_op, self.cond_val, self.cond_combine]),
            widgets.HBox([add_cond_btn, rem_cond_btn]),
            self.cond_extra_box,
            widgets.HTML("<b style='font-size:0.85em;color:#6b7280;'>Result</b>"),
            widgets.HBox([self.cond_then_col, self.cond_then_val, self.cond_else_col, self.cond_else_val]),
            widgets.HTML("<b style='font-size:0.85em;color:#6b7280;'>Value Map (overrides IF/THEN)</b>"),
            self.cond_map_text,
            widgets.HBox([self.cond_new_col, self.cond_btn]),
            self.cond_out
        ], layout=widgets.Layout(padding='10px'))

    def _parse_cond_value(self, raw: str):
        raw = raw.strip()
        try: return int(raw)
        except ValueError: pass
        try: return float(raw)
        except ValueError: pass
        return raw

    def _build_mask(self, df, col, op, raw_val):
        s = df[col]
        if op == 'is null':     return s.isna()
        if op == 'is not null': return s.notna()
        val = self._parse_cond_value(raw_val)
        if op == '==':  return s == val
        if op == '!=':  return s != val
        if op == '>':   return pd.to_numeric(s, errors='coerce') > float(val)
        if op == '>=':  return pd.to_numeric(s, errors='coerce') >= float(val)
        if op == '<':   return pd.to_numeric(s, errors='coerce') < float(val)
        if op == '<=':  return pd.to_numeric(s, errors='coerce') <= float(val)
        if op == 'isin':
            vals = [v.strip() for v in raw_val.split(',')]
            return s.astype(str).isin(vals)
        if op == 'not isin':
            vals = [v.strip() for v in raw_val.split(',')]
            return ~s.astype(str).isin(vals)
        if op == 'contains (str)': return s.astype(str).str.contains(raw_val, na=False)
        if op == 'startswith':     return s.astype(str).str.startswith(raw_val, na=False)
        if op == 'endswith':       return s.astype(str).str.endswith(raw_val, na=False)
        raise ValueError(f"Unknown operator: {op}")

    def _resolve_value(self, df, col_dd_val, const_txt_val):
        if col_dd_val == '(Constant)':
            return self._parse_cond_value(const_txt_val)
        return df[col_dd_val]

    def _apply_condition(self, _):
        df = self._get_df().copy()
        new_name = self.cond_new_col.value or 'new_flag'
        try:
            map_raw = self.cond_map_text.value.strip()
            if map_raw:
                col = self.cond_col.value
                mapping = {}
                for pair in map_raw.replace('\n', ',').split(','):
                    pair = pair.strip()
                    if ':' not in pair: continue
                    k, v = pair.split(':', 1)
                    mapping[self._parse_cond_value(k.strip())] = self._parse_cond_value(v.strip())
                df[new_name] = df[col].map(mapping)
                self.tabular_datasets[self.current_ds] = df
                self.state.log_step("Feature Eng", "Value Map applied", {"col": col, "new_col": new_name})
                self._propagate_to_state()
                self._refresh_columns()
                self._notify(self.cond_out, f"Mapped '{col}' → '{new_name}' ({len(mapping)} rules)")
                self._render_preview()
                return

            mask = self._build_mask(df, self.cond_col.value, self.cond_op.value, self.cond_val.value)
            combine = self.cond_combine.value
            for (r_col, r_op, r_val) in self.cond_extra_rows:
                extra = self._build_mask(df, r_col.value, r_op.value, r_val.value)
                mask = (mask & extra) if combine == 'AND' else (mask | extra)

            then_val = self._resolve_value(df, self.cond_then_col.value, self.cond_then_val.value)
            else_val = self._resolve_value(df, self.cond_else_col.value, self.cond_else_val.value)
            df[new_name] = np.where(mask, then_val, else_val)
            self.tabular_datasets[self.current_ds] = df
            self.state.log_step("Feature Eng", "Condition applied", {"new_col": new_name, "n_true": int(mask.sum())})
            self._propagate_to_state()
            self._refresh_columns()
            self._notify(self.cond_out, f"Created '{new_name}' — {int(mask.sum())} rows matched ({mask.mean()*100:.1f}%)")
            self._render_preview()
        except Exception as e:
            self._notify(self.cond_out, traceback.format_exc(), True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Custom Formula
    # ══════════════════════════════════════════════════════════════════════════
    def _build_formula_tab(self):
        cols = list(self._get_df().columns)
        self._formula_col_html = widgets.HTML(self._formula_col_list_html(cols))

        SNIPPETS = {
            "Binary flag (col != value)":       "new_col = (col != -1).astype(int)",
            "Replace value":                    "col = col.replace(-1, 0)",
            "Clamp between min/max":            "new_col = col.clip(lower=0, upper=100)",
            "Ratio A / B (safe)":               "new_col = col_a / col_b.replace(0, np.nan)",
            "Zscore normalize":                 "new_col = (col - col.mean()) / col.std()",
            "Min-max normalize":                "new_col = (col - col.min()) / (col.max() - col.min())",
            "Log1p transform":                  "new_col = np.log1p(col)",
            "Interaction A x B":                "new_col = col_a * col_b",
            "Conditional value":                "new_col = np.where(col > 0, col * 2, 0)",
            "Map string values":                "new_col = col.map({'yes': 1, 'no': 0})",
            "Extract year from date":           "new_col = pd.to_datetime(col).dt.year",
            "Count occurrences (freq encode)":  "new_col = col.map(col.value_counts())",
            "Rank (dense)":                     "new_col = col.rank(method='dense').astype(int)",
            "Rolling mean (window=3)":          "new_col = col.rolling(window=3, min_periods=1).mean()",
            "Cumulative sum":                   "new_col = col.cumsum()",
            "Difference (lag 1)":               "new_col = col.diff(1)",
            "Bin into N quantiles":             "new_col = pd.qcut(col, q=4, labels=False, duplicates='drop')",
            "Multi-col string concat":          "new_col = col_a.astype(str) + '_' + col_b.astype(str)",
            "Pct change":                       "new_col = col.pct_change().fillna(0)",
        }
        self._snippets = SNIPPETS

        self.formula_snippet = widgets.Dropdown(
            options=['-- pick a snippet --'] + list(SNIPPETS.keys()),
            description='Snippets:', layout=widgets.Layout(width='400px')
        )
        self.formula_insert_btn = widgets.Button(description='Insert', button_style='info', layout=widgets.Layout(width='80px', height='28px'))
        self.formula_editor = widgets.Textarea(
            placeholder=(
                "# Write one or more Python expressions.\n"
                "# Column names are available as pd.Series variables.\n"
                "# np, pd, math are available.\n\n"
                "new_col = (col_a / col_b.replace(0, np.nan)).round(4)"
            ),
            layout=widgets.Layout(width='100%', height='180px', font_family='monospace')
        )
        self.formula_preview_btn = widgets.Button(description='Preview (10 rows)', button_style='', layout=widgets.Layout(width='160px', height='32px'))
        self.formula_apply_btn  = widgets.Button(description='Apply to Dataset',  button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width='160px', height='32px'))
        self.formula_out = widgets.Output()

        def _insert_snippet(_):
            key = self.formula_snippet.value
            if key in SNIPPETS:
                current = self.formula_editor.value
                sep = '\n' if current and not current.endswith('\n') else ''
                self.formula_editor.value = current + sep + SNIPPETS[key] + '\n'

        self.formula_insert_btn.on_click(_insert_snippet)
        self.formula_preview_btn.on_click(lambda _: self._run_formula(preview=True))
        self.formula_apply_btn.on_click(lambda _: self._run_formula(preview=False))

        self.tab_formula = widgets.VBox([
            styles.help_box(
                "<b>Write any Python expression</b> using column names as <code>pd.Series</code> variables.<br>"
                "Available : <code>np</code>, <code>pd</code>, <code>math</code>, all column names, <code>df</code>.<br>"
                "Assign to a new name → creates column. Reassign existing → overwrites.<br>"
                "<b>Preview</b> shows 10 rows without saving. <b>Apply</b> commits to dataset.",
                "#0ea5e9"
            ),
            widgets.HTML("<b style='font-size:0.82em;color:#6b7280;'>Available columns :</b>"),
            self._formula_col_html,
            widgets.HBox([self.formula_snippet, self.formula_insert_btn], layout=widgets.Layout(align_items='center', gap='8px', margin='6px 0')),
            self.formula_editor,
            widgets.HBox([self.formula_preview_btn, self.formula_apply_btn], layout=widgets.Layout(gap='10px', margin='8px 0 0 0')),
            self.formula_out
        ], layout=widgets.Layout(padding='10px'))

    def _formula_col_list_html(self, cols):
        pills = ''.join(
            f"<span style='display:inline-block; background:#ede9fe; color:#5b21b6; "
            f"border-radius:4px; padding:2px 8px; margin:2px 3px; font-size:0.78em; "
            f"font-family:monospace; cursor:pointer;' "
            f"onclick=\"navigator.clipboard.writeText('{c}')\" title='click to copy'>{c}</span>"
            for c in cols
        )
        return f"<div style='line-height:2; margin-bottom:6px;'>{pills}</div>"

    def _refresh_formula_col_list(self):
        cols = list(self._get_df().columns)
        self._formula_col_html.value = self._formula_col_list_html(cols)

    def _run_formula(self, preview=False):
        df = self._get_df().copy()
        code = self.formula_editor.value.strip()
        if not code:
            self._notify(self.formula_out, "Formula editor is empty.", True)
            return

        ns = {col: df[col].copy() for col in df.columns}
        ns.update({'np': np, 'pd': pd, 'math': math, 'df': df})

        with self.formula_out:
            clear_output(wait=True)
            try:
                exec(compile(code, '<formula>', 'exec'), ns)
            except Exception:
                display(HTML(f"<pre style='color:#ef4444; font-size:0.82em;'>{traceback.format_exc()}</pre>"))
                return

        new_or_modified = {}
        for k, v in ns.items():
            if k.startswith('_') or k in ('np', 'pd', 'math', 'df'): continue
            if isinstance(v, pd.Series):
                if k not in df.columns or not df[k].equals(v):
                    new_or_modified[k] = v
            elif isinstance(v, (int, float, str, bool, np.integer, np.floating)):
                if k not in df.columns:
                    new_or_modified[k] = pd.Series([v] * len(df), index=df.index)

        if not new_or_modified:
            self._notify(self.formula_out, "No new or modified columns detected. Assign to a variable name.", True)
            return

        if preview:
            with self.formula_out:
                preview_df = pd.DataFrame({k: v for k, v in new_or_modified.items()}).head(10)
                display(HTML(f"<b style='color:#0ea5e9;'>Preview — {len(new_or_modified)} column(s) :</b>"))
                display(preview_df)
        else:
            for k, v in new_or_modified.items():
                self.tabular_datasets[self.current_ds][k] = v.values if isinstance(v, pd.Series) else v
            self.state.log_step("Feature Eng", "Custom Formula applied", {"columns": list(new_or_modified.keys())})
            self._propagate_to_state()
            self._refresh_columns()
            cols_str = ', '.join(f"'{c}'" for c in new_or_modified)
            self._notify(self.formula_out, f"Applied {len(new_or_modified)} column(s) : {cols_str}")
            self._render_preview()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Text Operations
    # ══════════════════════════════════════════════════════════════════════════
    def _build_text_tab(self):
        df = self._get_df()
        cols = list(df.columns)
        self.text_col = widgets.Dropdown(options=cols, description='Col:', layout=styles.LAYOUT_DD)
        text_ops = self.state.config.get("feature_engineering", {}).get("text_operations", [])
        self.text_op = widgets.Dropdown(
            options=text_ops, value=text_ops[0] if text_ops else 'Lowercase',
            description='Op:', layout=styles.LAYOUT_BTN_LARGE
        )
        self.text_arg1 = widgets.Text(description='Regex/Find/N:', layout=widgets.Layout(width='300px', display='none'))
        self.text_arg2 = widgets.Text(description='Replace With/Sep:', layout=widgets.Layout(width='300px', display='none'))
        self.text_new_col = widgets.Text(description='New Name:', placeholder='text_result', layout=styles.LAYOUT_DD)
        self.text_btn = widgets.Button(description='Apply Text Op', button_style=styles.BTN_PRIMARY)
        self.text_out = widgets.Output()

        def _on_op_change(change):
            op = change.new
            if op in ['Lowercase', 'Uppercase', 'Length']:
                self.text_arg1.layout.display = 'none'
                self.text_arg2.layout.display = 'none'
            elif op == 'Extract Regex':
                self.text_arg1.layout.display = 'block'
                self.text_arg2.layout.display = 'none'
            else:
                self.text_arg1.layout.display = 'block'
                self.text_arg2.layout.display = 'block'

        self.text_op.observe(_on_op_change, names='value')
        self.text_btn.on_click(self._apply_text)

        self.tab_text = widgets.VBox([
            styles.help_box("Process string/text columns.", "#06b6d4"),
            widgets.HBox([self.text_col, self.text_op, self.text_arg1, self.text_arg2]),
            widgets.HBox([self.text_new_col, self.text_btn]),
            self.text_out
        ], layout=widgets.Layout(padding='10px'))

    def _apply_text(self, _):
        df = self._get_df()
        col = self.text_col.value
        op = self.text_op.value
        new_name = self.text_new_col.value or f"{col}_txt"
        try:
            s = df[col].astype(str)
            if op == 'Lowercase':    res = s.str.lower()
            elif op == 'Uppercase':  res = s.str.upper()
            elif op == 'Length':     res = s.str.len()
            elif op == 'Extract Regex':
                res = s.str.extract(f'({self.text_arg1.value})', expand=False)
            elif op == 'Replace':
                res = s.str.replace(self.text_arg1.value, self.text_arg2.value, regex=True)
            elif op == 'Split & Keep N':
                idx = int(self.text_arg1.value)
                res = s.str.split(self.text_arg2.value).str[idx]
            self.tabular_datasets[self.current_ds][new_name] = res
            self.state.log_step("Feature Eng", "Text Operation applied", {"op": op, "new_col": new_name})
            self._propagate_to_state()
            self._refresh_columns()
            self._notify(self.text_out, f"Created column '{new_name}'")
            self._render_preview()
        except Exception as e:
            self._notify(self.text_out, str(e), True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Date / Time
    # ══════════════════════════════════════════════════════════════════════════
    def _build_date_tab(self):
        df = self._get_df()
        cols = list(df.columns)
        self.date_col = widgets.Dropdown(options=cols, description='Date Col:', layout=styles.LAYOUT_DD)
        date_ops = self.state.config.get("feature_engineering", {}).get("date_operations", [])
        self.date_extract = widgets.SelectMultiple(
            options=date_ops,
            value=[date_ops[0], date_ops[1]] if len(date_ops) >= 2 else date_ops,
            description='Extract:', layout=styles.LAYOUT_DD
        )
        self.date_btn = widgets.Button(description='Extract Features', button_style=styles.BTN_PRIMARY)
        self.date_out = widgets.Output()
        self.date_btn.on_click(self._apply_date)

        self.tab_date = widgets.VBox([
            styles.help_box("Extract temporal features from a date/time column.", "#f59e0b"),
            widgets.HBox([self.date_col, self.date_extract]),
            self.date_btn,
            self.date_out
        ], layout=widgets.Layout(padding='10px'))

    def _apply_date(self, _):
        df = self._get_df()
        col = self.date_col.value
        features = self.date_extract.value
        try:
            s = pd.to_datetime(df[col], errors='coerce')
            created = []
            if 'Year'      in features: df[f'{col}_Year']      = s.dt.year;           created.append(f'{col}_Year')
            if 'Month'     in features: df[f'{col}_Month']     = s.dt.month;          created.append(f'{col}_Month')
            if 'Day'       in features: df[f'{col}_Day']       = s.dt.day;            created.append(f'{col}_Day')
            if 'DayOfWeek' in features: df[f'{col}_DayOfWeek'] = s.dt.dayofweek;      created.append(f'{col}_DayOfWeek')
            if 'Hour'      in features: df[f'{col}_Hour']      = s.dt.hour;           created.append(f'{col}_Hour')
            if 'Minute'    in features: df[f'{col}_Minute']    = s.dt.minute;         created.append(f'{col}_Minute')
            if 'IsWeekend' in features:
                df[f'{col}_IsWeekend'] = (s.dt.dayofweek >= 5).astype(int)
                created.append(f'{col}_IsWeekend')
            self.tabular_datasets[self.current_ds] = df
            self.state.log_step("Feature Eng", "Date Features extracted", {"col": col, "features": created})
            self._propagate_to_state()
            self._refresh_columns()
            self._notify(self.date_out, f"Created {len(created)} columns : {', '.join(created)}")
            self._render_preview()
        except Exception as e:
            self._notify(self.date_out, str(e), True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Binning
    # ══════════════════════════════════════════════════════════════════════════
    def _build_binning_tab(self):
        df = self._get_df()
        cols = list(df.columns)
        self.bin_col = widgets.Dropdown(options=cols, description='Num Col:', layout=styles.LAYOUT_DD)
        bin_ops = self.state.config.get("feature_engineering", {}).get("binning_strategies", [])
        self.bin_method = widgets.Dropdown(
            options=bin_ops, value=bin_ops[0] if bin_ops else 'Equal Width (Cut)',
            description='Method:', layout=styles.LAYOUT_BTN_LARGE
        )
        self.bin_bins   = widgets.Text(value='5', description='Bins/Edges:', layout=styles.LAYOUT_DD)
        self.bin_labels = widgets.Checkbox(value=False, description='Numeric labels', layout=styles.LAYOUT_DD)
        self.bin_new_col = widgets.Text(description='New Name:', placeholder='binned_col', layout=styles.LAYOUT_DD)
        self.bin_btn = widgets.Button(description='Apply Binning', button_style=styles.BTN_PRIMARY)
        self.bin_out = widgets.Output()
        self.bin_btn.on_click(self._apply_binning)

        self.tab_binning = widgets.VBox([
            styles.help_box("Discretize continuous variables into bins.", "#10b981"),
            widgets.HBox([self.bin_col, self.bin_method, self.bin_bins]),
            widgets.HBox([self.bin_labels, self.bin_new_col, self.bin_btn]),
            self.bin_out
        ], layout=widgets.Layout(padding='10px'))

    def _apply_binning(self, _):
        df = self._get_df()
        col = self.bin_col.value
        method = self.bin_method.value
        bins_val = self.bin_bins.value
        new_name = self.bin_new_col.value or f"{col}_bin"
        try:
            s = pd.to_numeric(df[col], errors='coerce')
            lbls = False if self.bin_labels.value else None
            if method == 'Equal Width (Cut)':
                res = pd.cut(s, bins=int(bins_val), labels=lbls)
            elif method == 'Equal Frequency (Qcut)':
                res = pd.qcut(s, q=int(bins_val), labels=lbls, duplicates='drop')
            elif method == 'Custom Edges':
                edges = [float(x.strip()) for x in bins_val.split(',')]
                res = pd.cut(s, bins=edges, labels=lbls)
            self.tabular_datasets[self.current_ds][new_name] = res
            self.state.log_step("Feature Eng", "Binning applied", {"method": method, "new_col": new_name})
            self._propagate_to_state()
            self._refresh_columns()
            self._notify(self.bin_out, f"Created column '{new_name}'")
            self._render_preview()
        except Exception as e:
            self._notify(self.bin_out, str(e), True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Visualization
    # ══════════════════════════════════════════════════════════════════════════
    def _build_viz_tab(self):
        df = self._get_df()
        cols = list(df.columns)
        self.viz_x    = widgets.Dropdown(options=cols, description='X:', layout=widgets.Layout(width='250px'))
        self.viz_y    = widgets.Dropdown(options=cols, description='Y:', layout=widgets.Layout(width='250px'))
        self.viz_hue  = widgets.Dropdown(options=['(None)'] + cols, description='Hue:', layout=widgets.Layout(width='250px'))
        viz_ops = self.state.config.get("feature_engineering", {}).get("viz_types", [])
        self.viz_kind = widgets.Dropdown(options=viz_ops, value='auto', description='Type:', layout=styles.LAYOUT_BTN_LARGE)
        self.viz_btn  = widgets.Button(description='Plot', button_style='info')
        self.viz_out  = widgets.Output()
        self.viz_btn.on_click(self._apply_viz)

        self.tab_viz = widgets.VBox([
            styles.help_box("Quickly visualize newly created variables.", "#6366f1"),
            widgets.HBox([self.viz_x, self.viz_y, self.viz_hue, self.viz_kind, self.viz_btn]),
            self.viz_out
        ], layout=widgets.Layout(padding='10px'))

    def _apply_viz(self, _):
        df = self._get_df()
        x, y, hue = self.viz_x.value, self.viz_y.value, self.viz_hue.value
        hue_arg = None if hue == '(None)' else hue
        kind = self.viz_kind.value
        with self.viz_out:
            clear_output(wait=True)
            plt.figure(figsize=(9, 5))
            try:
                if kind == 'auto':
                    if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                        kind = 'scatter'
                    elif pd.api.types.is_numeric_dtype(df[y]):
                        kind = 'box'
                    else:
                        kind = 'bar'
                if kind == 'scatter':  sns.scatterplot(data=df, x=x, y=y, hue=hue_arg, alpha=0.7)
                elif kind == 'line':   sns.lineplot(data=df, x=x, y=y, hue=hue_arg)
                elif kind == 'bar':    sns.barplot(data=df, x=x, y=y, hue=hue_arg)
                elif kind == 'box':    sns.boxplot(data=df, x=x, y=y, hue=hue_arg)
                elif kind == 'violin': sns.violinplot(data=df, x=x, y=y, hue=hue_arg)
                elif kind == 'hist':   sns.histplot(data=df, x=x, hue=hue_arg, kde=True)
                elif kind == 'kde':    sns.kdeplot(data=df, x=x, hue=hue_arg, fill=True)
                plt.title(f"{kind.capitalize()} : {y} vs {x}" if x != y else f"{kind.capitalize()} of {x}")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"[ERROR] Plotting failed : {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Target Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    def _build_dashboard_tab(self):
        df = self._get_df()
        cols = list(df.columns)
        target_val = '(None)'
        if hasattr(self.state, 'business_context') and self.state.business_context.get('target') in cols:
            target_val = self.state.business_context['target']
        self.dash_target   = widgets.Dropdown(options=['(None)'] + cols, value=target_val, description='Target:', layout=styles.LAYOUT_DD)
        self.dash_features = widgets.SelectMultiple(options=cols, description='Features:', layout=widgets.Layout(width='300px', height='150px'))
        self.dash_btn = widgets.Button(description='Plot Dashboard', button_style=styles.BTN_PRIMARY)
        self.dash_out = widgets.Output()
        self.dash_btn.on_click(self._apply_dashboard)

        self.tab_dashboard = widgets.VBox([
            styles.help_box("Multi-feature analysis vs Target. Select 1 to 4 features.", "#be185d"),
            widgets.HBox([self.dash_target, self.dash_features, self.dash_btn]),
            self.dash_out
        ], layout=widgets.Layout(padding='10px'))

    def _apply_dashboard(self, _):
        df = self._get_df()
        target = self.dash_target.value
        features = list(self.dash_features.value)
        if target == '(None)' or not features:
            self._notify(self.dash_out, "Select a valid target and at least ONE feature.", True)
            return
        features = features[:4]
        with self.dash_out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, len(features), figsize=(len(features) * 6, 5))
            if len(features) == 1: axes = [axes]
            target_vals = df[target].dropna().unique()
            is_target_binary = len(target_vals) == 2
            rate_series = None
            overall_rate = None
            if is_target_binary:
                top_val = sorted(target_vals)[-1]
                rate_series = (df[target] == top_val).astype(int)
                overall_rate = rate_series.mean()
            else:
                if pd.api.types.is_numeric_dtype(df[target]):
                    rate_series = pd.to_numeric(df[target], errors='coerce')
                    overall_rate = rate_series.mean()
                else:
                    print("Please use a binary class or numeric target.")
                    return
            sns.set_style("whitegrid")
            for ax, feat in zip(axes, features):
                is_num = pd.api.types.is_numeric_dtype(df[feat])
                n_unique = df[feat].nunique()
                if is_num and n_unique < 20: is_num = False
                if is_num:
                    if is_target_binary:
                        sns.histplot(data=df, x=feat, hue=target, multiple="layer", alpha=0.5, ax=ax)
                        ax.set_title(f"Distribution by {target}")
                    else:
                        sns.scatterplot(data=df, x=feat, y=target, ax=ax, alpha=0.5)
                        ax.set_title(f"{feat} vs {target}")
                else:
                    df_temp = pd.DataFrame({'f': df[feat].astype(str), 'r': rate_series}).dropna()
                    if n_unique > 30:
                        top = df_temp['f'].value_counts().nlargest(15).index
                        df_temp = df_temp[df_temp['f'].isin(top)]
                    rate_by_cat = df_temp.groupby('f')['r'].mean().sort_values(ascending=True)
                    if len(rate_by_cat) > 0:
                        use_h = n_unique > 12 or any(len(str(x)) > 8 for x in rate_by_cat.index)
                        if not use_h:
                            x_pos = np.arange(len(rate_by_cat))
                            ax.bar(x_pos, rate_by_cat.values, color='#a78bfa')
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(rate_by_cat.index, rotation=45)
                            ax.set_ylabel("Rate")
                            if overall_rate is not None:
                                ax.axhline(overall_rate, color='#f28859', linestyle='--', label="Overall mean")
                                ax.legend()
                        else:
                            y_pos = np.arange(len(rate_by_cat))
                            ax.barh(y_pos, rate_by_cat.values, color='#5abcb6')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(rate_by_cat.index)
                            ax.set_xlabel("Rate")
                            if overall_rate is not None:
                                ax.axvline(overall_rate, color='#f28859', linestyle='--', label="Overall mean")
                                ax.legend()
                        ax.set_title(f"Target Rate by {feat}")
            fig.suptitle(f"Key Feature Analysis vs Target '{target}'", fontsize=14, y=1.05)
            plt.tight_layout()
            plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB : Manage Columns
    # ══════════════════════════════════════════════════════════════════════════
    def _build_manage_tab(self):
        df = self._get_df()
        cols = list(df.columns)
        tabular_types = []
        if hasattr(self.state, 'config'):
            enc_cfg = self.state.config.get('encoding', {})
            tabular_cfg = enc_cfg.get('tabular', enc_cfg)
            tabular_types = list(tabular_cfg.keys())
        if not tabular_types:
            tabular_types = ['numeric', 'categorical', 'binary', 'datetime', 'text', 'id_like']

        self.manage_col = widgets.Dropdown(options=cols, description='Column:', layout=styles.LAYOUT_DD)
        manage_ops = self.state.config.get("feature_engineering", {}).get("manage_actions", [])
        self.manage_action = widgets.Dropdown(
            options=manage_ops, value=manage_ops[0] if manage_ops else 'Set Type (Meta)',
            description='Action:', layout=widgets.Layout(width='250px')
        )
        self.manage_type = widgets.Dropdown(
            options=tabular_types, value=tabular_types[0],
            description='Type:', layout=styles.LAYOUT_BTN_LARGE
        )
        self.manage_new_name = widgets.Text(
            description='New Name:', placeholder='for duplication',
            layout=widgets.Layout(width='200px', display='none')
        )
        self.manage_btn = widgets.Button(description='Apply Action', button_style='warning')
        self.manage_out = widgets.Output()

        def _on_action_change(change):
            if change.new == 'Duplicate':
                self.manage_new_name.layout.display = 'block'
                self.manage_type.layout.display = 'none'
            elif change.new == 'Set Type (Meta)':
                self.manage_new_name.layout.display = 'none'
                self.manage_type.layout.display = 'block'
            else:
                self.manage_new_name.layout.display = 'none'
                self.manage_type.layout.display = 'none'

        self.manage_action.observe(_on_action_change, names='value')
        self.manage_btn.on_click(self._apply_manage)

        self.tab_manage = widgets.VBox([
            styles.help_box("Manage columns : override type, duplicate, or delete.", "#eab308"),
            widgets.HBox([self.manage_col, self.manage_action, self.manage_type, self.manage_new_name]),
            widgets.HBox([self.manage_btn]),
            self.manage_out
        ], layout=widgets.Layout(padding='10px'))

    def _apply_manage(self, _):
        df = self._get_df()
        col = self.manage_col.value
        action = self.manage_action.value
        try:
            if action == 'Delete':
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                    self.tabular_datasets[self.current_ds] = df
                    self.state.log_step("Feature Eng", "Column deleted", {"col": col})
                    self._propagate_to_state()
                    self._refresh_columns()
                    self._notify(self.manage_out, f"Deleted column '{col}'")
                    self._render_preview()
                else:
                    self._notify(self.manage_out, f"Column '{col}' not found.", True)

            elif action == 'Duplicate':
                new_name = self.manage_new_name.value or f"{col}_copy"
                if new_name in df.columns:
                    self._notify(self.manage_out, f"Column '{new_name}' already exists.", True)
                    return
                df[new_name] = df[col].copy()
                self.tabular_datasets[self.current_ds] = df
                self.state.log_step("Feature Eng", "Column duplicated", {"col": col, "new_col": new_name})
                self._propagate_to_state()
                self._refresh_columns()
                self._notify(self.manage_out, f"Duplicated '{col}' to '{new_name}'")
                self._render_preview()

            elif action == 'Set Type (Meta)':
                new_type = self.manage_type.value
                ds = self.current_ds
                if not hasattr(self.state, 'meta'): self.state.meta = {}
                if ds not in self.state.meta: self.state.meta[ds] = {}
                if col not in self.state.meta[ds]: self.state.meta[ds][col] = {}
                self.state.meta[ds][col]['kind'] = new_type
                self.state.log_step("Feature Eng", "Type overridden", {"col": col, "new_type": new_type})
                self._notify(self.manage_out, f"Set type of '{col}' to '{new_type}'")
        except Exception as e:
            self._notify(self.manage_out, str(e), True)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def runner(state):
    fe = FeatureEngUI(state)
    if hasattr(fe, 'ui'):
        display(fe.ui)
    return fe


try:
    runner(state)
except NameError:
    pass