"""Étape 3 — EDA (UltimateEDA + PlotDashboard)."""
from __future__ import annotations
import io, base64, warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles

_SEP = widgets.HTML("<div style='height:1px;background:#e5e7eb;margin:10px 0;'></div>")


def infer_types(df: pd.DataFrame, available_types: list | None = None) -> dict:
    if not available_types:
        available_types = ["numeric", "categorical", "datetime", "binary", "id_like", "text"]
    col_meta = {}
    for col in df.columns:
        s, n_unique, n_total = df[col], df[col].nunique(), max(len(df[col].dropna()), 1)
        kind = "unknown"
        if pd.api.types.is_datetime64_any_dtype(s) and "datetime" in available_types:
            kind = "datetime"
        elif (pd.api.types.is_bool_dtype(s) or n_unique == 2) and "binary" in available_types:
            kind = "binary"
        elif pd.api.types.is_numeric_dtype(s):
            if "id_like" in available_types and n_unique / n_total > 0.95 and n_unique > 50:
                kind = "id_like"
            elif "numeric" in available_types:
                kind = "numeric"
        else:
            avg_len = s.dropna().astype(str).str.len().mean() if not s.dropna().empty else 0
            if "text" in available_types and n_unique / n_total > 0.90 and avg_len > 20:
                kind = "text"
            elif "id_like" in available_types and n_unique / n_total > 0.95 and n_unique > 50:
                kind = "id_like"
            elif "categorical" in available_types:
                kind = "categorical"
        if kind == "unknown":
            kind = available_types[0] if available_types else "categorical"
        col_meta[col] = {
            "kind": kind, "n_unique": n_unique,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "pct_miss": float(s.isna().mean() * 100),
        }
    return col_meta


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=96, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class PlotDashboard:
    """Dashboard de plots persistant entre les reruns."""

    def __init__(self, state=None):
        self._state   = state
        self._entries = list(state.eda_dashboard) if (state and hasattr(state, "eda_dashboard")) else []
        self._visible = False
        self._build_widget()

    def _build_widget(self) -> None:
        self.toggle_btn = widgets.Button(
            description=f"Plot Dashboard ({len(self._entries)})",
            button_style="info", layout=widgets.Layout(width="220px", height="32px"))
        self.clear_btn  = widgets.Button(description="Clear all", button_style="warning",
                                          layout=widgets.Layout(width="100px", height="32px"))
        self.export_btn = widgets.Button(description="Export HTML", button_style="primary",
                                          layout=widgets.Layout(width="120px", height="32px"))
        self.toggle_btn.on_click(self._toggle)
        self.clear_btn.on_click(self._clear)
        self.export_btn.on_click(self._export_html)
        self._header = widgets.HBox(
            [self.toggle_btn, self.clear_btn, self.export_btn],
            layout=widgets.Layout(align_items="center", gap="8px", padding="8px 12px",
                                   border="1px solid #e2e8f0", border_radius="8px", margin="12px 0 0 0"))
        self._grid_out = widgets.Output()
        self._body = widgets.VBox([self._grid_out], layout=widgets.Layout(
            display="none", border="1px solid #e2e8f0", border_radius="8px",
            padding="12px", margin="4px 0 0 0", background_color="#ffffff"))
        self.widget = widgets.VBox([self._header, self._body])

    def _toggle(self, b=None) -> None:
        self._visible = not self._visible
        self._body.layout.display = "flex" if self._visible else "none"
        arrow = "v" if self._visible else ">"
        self.toggle_btn.description = f"{arrow} Plot Dashboard ({len(self._entries)})"
        if self._visible:
            self._render_grid()

    def _clear(self, b=None) -> None:
        self._entries.clear(); self._persist(); self._update_label()
        with self._grid_out: clear_output(wait=True)

    def _update_label(self) -> None:
        arrow = "v" if self._visible else ">"
        self.toggle_btn.description = f"{arrow} Plot Dashboard ({len(self._entries)})"

    def _persist(self) -> None:
        if self._state is not None:
            self._state.eda_dashboard = list(self._entries)

    def add(self, fig, title: str = "") -> None:
        b64 = _fig_to_b64(fig)
        self._entries.append({"b64": b64, "title": title})
        self._persist(); self._update_label()
        if self._visible:
            self._render_grid()

    def _render_grid(self) -> None:
        with self._grid_out:
            clear_output(wait=True)
            if not self._entries:
                display(HTML("<div style='color:#94a3b8;font-size:0.85em;padding:8px;'>No plots saved yet.</div>"))
                return
            cards = ""
            for i, entry in enumerate(self._entries):
                label = entry["title"] or f"Plot {i+1}"
                cards += (
                    f"<div style='border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;"
                    f"background:#fff;flex:1 1 420px;min-width:320px;max-width:580px;'>"
                    f"<div style='background:#f8fafc;padding:6px 12px;border-bottom:1px solid #e2e8f0;"
                    f"font-size:0.78em;font-weight:600;color:#475569;display:flex;"
                    f"justify-content:space-between;align-items:center;'>"
                    f"<span>{label}</span><span style='color:#94a3b8;font-weight:400;'>#{i+1}</span></div>"
                    f"<div style='padding:6px;'>"
                    f"<img src='data:image/png;base64,{entry['b64']}' style='width:100%;height:auto;display:block;'/>"
                    f"</div></div>"
                )
            display(HTML(f"<div style='display:flex;flex-wrap:wrap;gap:12px;padding:4px 0;'>{cards}</div>"))

    def _export_html(self, b=None) -> None:
        if not self._entries:
            return
        cards = ""
        for i, entry in enumerate(self._entries):
            label = entry["title"] or f"Plot {i+1}"
            cards += (
                f"<div class='card'>"
                f"<div class='card-header'><span>{label}</span><span class='idx'>#{i+1}</span></div>"
                f"<img src='data:image/png;base64,{entry['b64']}'/></div>"
            )
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>EDA Dashboard</title>"
            "<style>body{font-family:sans-serif;background:#f1f5f9;margin:0;padding:20px}"
            "h1{color:#1e293b;font-size:1.1em;font-weight:600;margin-bottom:16px}"
            ".grid{display:flex;flex-wrap:wrap;gap:16px}"
            ".card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;flex:1 1 420px}"
            ".card-header{background:#f8fafc;padding:6px 12px;border-bottom:1px solid #e2e8f0;"
            "font-size:0.78em;font-weight:600;color:#475569;display:flex;justify-content:space-between}"
            ".idx{color:#94a3b8;font-weight:400}img{width:100%;height:auto;display:block}"
            "</style></head><body>"
            f"<h1>EDA Plot Dashboard — {len(self._entries)} plots</h1>"
            f"<div class='grid'>{cards}</div></body></html>"
        )
        path = "eda_dashboard.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        display(HTML(
            f"<div style='color:#065f46;background:#d1fae5;border-left:4px solid #10b981;"
            f"padding:8px 12px;font-size:0.85em;border-radius:4px;margin-top:6px;'>"
            f"Dashboard exporté → <b>{path}</b> ({len(self._entries)} plots)</div>"
        ))


class EDAVisualizerUtils:
    """Utilitaires de visualisation univariée et bivariée."""

    @staticmethod
    def plot_univariate(df, col, kind=None, plot_type=None, bins=30, kde=True,
                        log_scale=False, color="#3b82f6", hue=None, palette="Set2"):
        is_num = pd.api.types.is_numeric_dtype(df[col])
        if kind is None:
            kind = "numeric" if is_num else "categorical"
        plot_as_numeric = kind in ("numeric", "timeseries") or (
            kind not in ("categorical", "binary", "id_like", "text") and is_num)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
        hue_data = df[hue] if hue and hue in df.columns else None
        if plot_as_numeric:
            plot_type = plot_type or "hist"
            data = df[col].dropna()
            if hue_data is not None:
                groups = df[[col, hue]].dropna()
                unique_vals = groups[hue].unique()
                pal = sns.color_palette(palette, len(unique_vals))
                for i, val in enumerate(unique_vals):
                    subset = groups[groups[hue] == val][col]
                    if plot_type == "hist":
                        sns.histplot(subset, kde=kde, bins=bins, color=pal[i],
                                     label=str(val), alpha=0.6, ax=ax, log_scale=log_scale)
                    elif plot_type == "kde":
                        sns.kdeplot(subset, fill=True, color=pal[i],
                                    label=str(val), alpha=0.5, ax=ax, log_scale=log_scale)
                    elif plot_type in ("box", "violin"):
                        plot_df = df[[col, hue]].dropna().copy()
                        plot_df[hue] = plot_df[hue].astype(str)
                        if plot_type == "box":
                            sns.boxplot(data=plot_df, x=hue, y=col, palette=palette, ax=ax)
                        else:
                            sns.violinplot(data=plot_df, x=hue, y=col, palette=palette, ax=ax)
                        if log_scale:
                            ax.set_yscale("log")
                        break
                ax.legend(title=hue, fontsize=9)
            else:
                if plot_type == "hist":
                    sns.histplot(data, kde=kde, bins=bins, color=color, log_scale=log_scale, ax=ax)
                elif plot_type == "kde":
                    sns.kdeplot(data, fill=True, color=color, log_scale=log_scale, ax=ax)
                elif plot_type == "box":
                    sns.boxplot(y=data, color=color, ax=ax)
                    if log_scale: ax.set_yscale("log")
                elif plot_type == "violin":
                    sns.violinplot(y=data, color=color, ax=ax)
                    if log_scale: ax.set_yscale("log")
            ax.set_title(f"Distribution de {col}" + (f" par {hue}" if hue_data is not None else ""))
        else:
            plot_type = plot_type or "bar"
            if hue_data is not None:
                plot_df = df[[col, hue]].dropna().copy()
                plot_df[col] = plot_df[col].astype(str)
                plot_df[hue] = plot_df[hue].astype(str)
                top_cats = plot_df[col].value_counts().nlargest(bins).index
                plot_df = plot_df[plot_df[col].isin(top_cats)]
                if plot_type == "bar":
                    tbl = pd.crosstab(plot_df[col], plot_df[hue])
                    tbl.plot(kind="bar", ax=ax, colormap=palette, edgecolor="white")
                    ax.legend(title=hue, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
                    ax.set_title(f"Top {len(top_cats)} catégories pour {col} par {hue}")
                elif plot_type == "pie":
                    tbl = pd.crosstab(plot_df[col], plot_df[hue], normalize="index")
                    tbl.plot(kind="bar", stacked=True, ax=ax, colormap=palette)
                    ax.legend(title=hue, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
                    ax.set_title(f"Proportion de {hue} dans {col}")
            else:
                val_counts = df[col].value_counts().head(bins)
                if plot_type == "bar":
                    sns.barplot(y=val_counts.index.astype(str), x=val_counts.values,
                                palette="viridis", ax=ax)
                    ax.set_title(f"Top {len(val_counts)} catégories pour {col}")
                elif plot_type == "pie":
                    ax.axis("off")
                    fig.clear()
                    ax2 = fig.add_subplot(111)
                    ax2.pie(val_counts.values, labels=val_counts.index,
                            autopct="%1.1f%%", colors=sns.color_palette("viridis", len(val_counts)))
                    ax2.set_title(f"Top {len(val_counts)} catégories pour {col}")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_bivariate(df, x_col, y_col, x_kind=None, y_kind=None,
                       plot_type=None, hue=None, alpha=0.5, palette="Set2"):
        num_dtypes = (pd.api.types.is_numeric_dtype(df[x_col]),
                      pd.api.types.is_numeric_dtype(df[y_col]))
        x_kind = x_kind or ("numeric" if num_dtypes[0] else "categorical")
        y_kind = y_kind or ("numeric" if num_dtypes[1] else "categorical")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")

        def _is_num_kind(k, is_n):
            return k in ("numeric", "timeseries") or (
                k not in ("categorical", "binary", "id_like", "text") and is_n)

        x_is_n = _is_num_kind(x_kind, num_dtypes[0])
        y_is_n = _is_num_kind(y_kind, num_dtypes[1])
        hue_data = df[hue] if hue and hue in df.columns else None

        if x_is_n and y_is_n:
            plot_type = plot_type or "scatter"
            if plot_type == "scatter":
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_data,
                                alpha=alpha, palette=palette if hue else None, ax=ax)
            elif plot_type == "hexbin":
                ax.hexbin(x=df[x_col], y=df[y_col], gridsize=30, cmap="Blues", mincnt=1)
                fig.colorbar(ax.collections[0], ax=ax, label="count")
            elif plot_type == "hist2d":
                sns.histplot(data=df, x=x_col, y=y_col, bins=30,
                             pthresh=.1, cmap="mako", cbar=True, ax=ax)
            elif plot_type == "kde":
                sns.kdeplot(data=df, x=x_col, y=y_col, hue=hue_data,
                            fill=True, alpha=alpha, palette=palette, ax=ax)
        elif (x_is_n and not y_is_n) or (not x_is_n and y_is_n):
            plot_type = plot_type or "box"
            c_col = x_col if not x_is_n else y_col
            top_cats = df[c_col].value_counts().nlargest(15).index
            plot_df  = df[df[c_col].isin(top_cats)].copy()
            plot_df[c_col] = plot_df[c_col].astype(str)
            if hue and hue in plot_df.columns:
                plot_df[hue] = plot_df[hue].astype(str)
            if plot_type == "box":
                sns.boxplot(data=plot_df, x=x_col, y=y_col, hue=hue, palette=palette, ax=ax)
            elif plot_type == "violin":
                sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue, palette=palette, ax=ax)
            elif plot_type == "strip":
                sns.stripplot(data=plot_df, x=x_col, y=y_col, hue=hue,
                              alpha=alpha, palette=palette, dodge=bool(hue), ax=ax)
            elif plot_type == "swarm":
                if len(plot_df) > 1000:
                    plot_df = plot_df.sample(1000, random_state=42)
                sns.swarmplot(data=plot_df, x=x_col, y=y_col,
                              hue=hue, palette=palette, dodge=bool(hue), ax=ax)
        else:
            plot_type = plot_type or "heatmap"
            top_x = df[x_col].value_counts().nlargest(15).index
            top_y = df[y_col].value_counts().nlargest(15).index
            plot_df = df[df[x_col].isin(top_x) & df[y_col].isin(top_y)]
            if plot_type == "heatmap":
                tbl = pd.crosstab(plot_df[y_col], plot_df[x_col])
                sns.heatmap(tbl, annot=True, fmt="d", cmap="Blues", ax=ax)
            elif plot_type == "stacked_bar":
                tbl = pd.crosstab(plot_df[x_col], plot_df[y_col], normalize="index")
                tbl.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
                ax.legend(title=y_col, bbox_to_anchor=(1.05, 1), loc="upper left")
            elif plot_type == "pie":
                plt.close(fig)
                tbl = pd.crosstab(plot_df[y_col], plot_df[x_col])
                n_pies = len(tbl.index)
                cols_n = min(n_pies, 3)
                rows_n = (n_pies - 1) // cols_n + 1
                fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 4, rows_n * 4))
                fig.patch.set_facecolor("#f8fafc")
                axes = np.array(axes).reshape(-1)
                for i, (idx, row) in enumerate(tbl.iterrows()):
                    if i < len(axes):
                        axes[i].pie(row.values, labels=row.index, autopct="%1.1f%%",
                                    colors=sns.color_palette("viridis", len(row)))
                        axes[i].set_title(f"{y_col} = {idx}")
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

        if plot_type != "pie":
            ax.set_title(f"{y_col} vs {x_col}")
        plt.tight_layout()
        return fig


class UltimateEDA:
    """Interface EDA complète — onglets Quality, Target, Univarié, Bivarié, Multivarié, Compare."""

    def __init__(self, state):
        self.state = state
        if not hasattr(self.state, "eda_dashboard"):
            self.state.eda_dashboard = []
        self.dashboard = PlotDashboard(state=self.state)
        self.all_datasets: dict = {}
        for k, v in self.state.data_raw.items():
            self.all_datasets[f"[RAW] {k}"] = v
        for k, v in getattr(self.state, "data_cleaned", {}).items():
            self.all_datasets[f"[CLN] {k}"] = v
        for k, v in getattr(self.state, "data_encoded", {}).items():
            self.all_datasets[f"[ENC] {k}"] = v
        self.meta: dict = {}
        if not hasattr(self.state, "meta"):
            self.state.meta = {}
        for k, v in self.all_datasets.items():
            if isinstance(v, pd.DataFrame):
                orig_key = k.split(" ", 1)[1] if " " in k else k
                if orig_key not in self.state.meta:
                    self.state.meta[orig_key] = {}
                enc_cfg = self.state.config.get("encoding", {})
                tabular_cfg = enc_cfg.get("tabular", enc_cfg)
                tabular_types = list(tabular_cfg.keys()) if hasattr(self.state, "config") else None
                inferred = infer_types(v, available_types=tabular_types)
                for col_name, col_info in inferred.items():
                    if col_name not in self.state.meta[orig_key]:
                        self.state.meta[orig_key][col_name] = col_info
                    else:
                        user_kind = self.state.meta[orig_key][col_name].get("kind", col_info["kind"])
                        self.state.meta[orig_key][col_name].update(col_info)
                        self.state.meta[orig_key][col_name]["kind"] = user_kind
                self.meta[k] = self.state.meta[orig_key]
        self.state.visualizers = EDAVisualizerUtils()
        self.current_ds: str | None = None
        self._build_ui()

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset_state(self) -> None:
        for attr in ("data_raw", "data_cleaned", "data_encoded", "data_splits",
                     "data_types", "meta", "business_context", "models", "history",
                     "eda_dashboard", "config"):
            setattr(self.state, attr, {} if attr != "history" and attr != "eda_dashboard" else [])
        self.dashboard._entries.clear()
        self.dashboard._update_label()
        with self.dashboard._grid_out:
            clear_output(wait=True)
        self._reset_msg.value = (
            "<div style='color:#b45309;background:#fef3c7;border-left:4px solid #f59e0b;"
            "padding:8px 12px;font-size:0.85em;border-radius:4px;margin-top:6px;'>"
            "<b>All state has been reset.</b> Reload your data and config to continue.</div>"
        )

    def _build_reset_bar(self) -> widgets.HBox:
        reset_btn = widgets.Button(
            description="⟳ Reset All State — WILL DELETE ALL LOADED DATASETS",
            button_style="danger", layout=widgets.Layout(width="auto", height="32px"))
        self._reset_msg = widgets.HTML("")
        reset_btn.on_click(lambda b: self.reset_state())
        return widgets.HBox(
            [reset_btn, self._reset_msg],
            layout=widgets.Layout(align_items="center", gap="10px", padding="6px 12px",
                                   border="1px solid #fecaca", border_radius="8px",
                                   background_color="#fff5f5", margin="0 0 12px 0"))

    # ── build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        if not self.state.config:
            self.ui = styles.error_msg("[ERROR] Configuration non chargée. Exécutez d'abord l'étape Config.")
            return
        if not self.all_datasets:
            self.ui = widgets.HTML(
                "<div style='padding:12px;color:#b91c1c;'>"
                "[WARNING] Aucun dataset disponible. Chargez des données d'abord.</div>")
            return
        self.ds_selector = widgets.Dropdown(
            options=list(self.all_datasets.keys()),
            description="Dataset:", layout=widgets.Layout(width="360px"))
        self.ds_selector.observe(self.on_ds_change, names="value")
        self.current_ds = self.ds_selector.value
        header = widgets.HTML(styles.card_html("EDA", "Exploratory Data Analysis", ""))
        top_bar = widgets.HBox(
            [header, widgets.HTML("<div style='flex:1'></div>"), self.ds_selector],
            layout=widgets.Layout(align_items="center", justify_content="space-between",
                                   margin="0 0 12px 0", padding="0 0 10px 0",
                                   border_bottom="2px solid #ede9fe"))
        reset_bar = self._build_reset_bar()
        self.dynamic_ui = widgets.VBox([])
        self.ui = widgets.VBox(
            [top_bar, reset_bar, self.dynamic_ui, self.dashboard.widget],
            layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb",
                                   padding="18px", border_radius="10px", background_color="#ffffff"))
        self.on_ds_change(None)

    def on_ds_change(self, change) -> None:
        if change:
            self.current_ds = change["new"]
        data = self.all_datasets[self.current_ds]
        orig_key = self.current_ds.split(" ", 1)[1] if " " in self.current_ds else self.current_ds
        ds_type = self.state.data_types.get(orig_key, "tabular")
        if ds_type in ("tabular", "sklearn", "clipboard", "excel") or isinstance(data, pd.DataFrame):
            self._build_tabular_ui(data)
        elif ds_type == "image":
            self._build_image_ui(data)
        elif ds_type == "text":
            self._build_text_ui(data)
        elif ds_type == "graph":
            self._build_graph_ui(data)
        elif ds_type == "ontology":
            self._build_ontology_ui(data)
        elif ds_type == "timeseries":
            self._build_timeseries_ui(data)
        else:
            self.dynamic_ui.children = [widgets.HTML(
                f"<div style='padding:12px;'>[INFO] Dataset non-tabulaire ({ds_type}). "
                f"Visualisation limitée.</div>")]

    # ── tabular UI ────────────────────────────────────────────────────────────
    def _build_tabular_ui(self, df: pd.DataFrame) -> None:
        self.tabs = widgets.Tab()
        cols = list(df.columns)

        # ── Tab 0 : Quality & Stats ───────────────────────────────────────────
        self.out_recap = widgets.Output()
        enc_cfg = self.state.config.get("encoding", {})
        tabular_cfg = enc_cfg.get("tabular", enc_cfg)
        tabular_types = list(tabular_cfg.keys()) or ["numeric", "categorical", "datetime", "binary", "id_like", "text"]
        self.type_col_dd  = widgets.Dropdown(options=cols, description="Column:", layout=styles.LAYOUT_DD)
        self.type_kind_dd = widgets.Dropdown(options=tabular_types, description="-> type:", layout=styles.LAYOUT_AUTO)
        self.type_btn     = widgets.Button(description="Update", button_style=styles.BTN_WARNING, layout=styles.LAYOUT_BTN_STD)
        self.type_msg     = widgets.HTML("")

        def _update_kind_dd(change):
            if self.type_col_dd.value:
                curr = self.meta[self.current_ds][self.type_col_dd.value].get("kind", "")
                self.type_kind_dd.value = curr if curr in self.type_kind_dd.options else (self.type_kind_dd.options[0] if self.type_kind_dd.options else None)

        self.type_col_dd.observe(_update_kind_dd, names="value")
        if cols:
            _update_kind_dd(None)

        def _on_type_update(b):
            col, new_kind = self.type_col_dd.value, self.type_kind_dd.value
            orig_key = self.current_ds.split(" ", 1)[1] if " " in self.current_ds else self.current_ds
            self.state.meta[orig_key][col]["kind"] = new_kind
            self.meta[self.current_ds][col]["kind"] = new_kind
            self.state.log_step("EDA", "Type Override", {"dataset": orig_key, "column": col, "new_kind": new_kind})
            self.type_msg.value = f"<span style='color:#059669;font-size:0.85em;'>[OK] '{col}' → {new_kind}</span>"
            self._fill_recap_tab(df)

        self.type_btn.on_click(_on_type_update)
        self.btn_missing = widgets.Button(description="Plot Missing", button_style=styles.BTN_INFO, layout=styles.LAYOUT_BTN_STD)
        self.btn_missing.on_click(lambda b: self._plot_missing(df))
        help_recap = styles.help_box(
            "<b>Column Metadata:</b> types, kinds inférés, taux de manquants.<br>"
            "<b>Missing Values:</b> cliquez 'Plot Missing' pour visualiser.<br>"
            "<b>Override Type:</b> corrigez le type inféré via les dropdowns.", "#3b82f6")
        type_row   = widgets.HBox([self.type_col_dd, self.type_kind_dd, self.type_btn, self.type_msg],
                                   layout=widgets.Layout(align_items="center", gap="8px", margin="0 0 10px 0"))
        action_row = widgets.HBox([self.btn_missing], layout=widgets.Layout(margin="8px 0 0 0"))
        tab_recap  = widgets.VBox([help_recap, type_row, _SEP, self.out_recap, action_row])

        # ── Tab 1 : Target Analysis ───────────────────────────────────────────
        target_val = "(None)"
        if hasattr(self.state, "business_context") and self.state.business_context.get("target") in cols:
            target_val = self.state.business_context["target"]
        self.target_dd         = widgets.Dropdown(options=["(None)"] + cols, value=target_val, description="Target:", layout=styles.LAYOUT_DD)
        self.target_feature_dd = widgets.Dropdown(options=cols, description="Feature:", layout=styles.LAYOUT_DD)
        self.target_btn        = widgets.Button(description="Analyze Feature vs Target", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="200px", height="32px"))
        self.target_out        = widgets.Output()

        def _on_target_change(change):
            if change["new"] and change["new"] != "(None)":
                if not hasattr(self.state, "business_context"):
                    self.state.business_context = {}
                self.state.business_context["target"] = change["new"]
                self.state.log_step("EDA", "Target Variable Selected", {"target": change["new"]})

        self.target_dd.observe(_on_target_change, names="value")
        self.target_btn.on_click(lambda b: self._plot_target_analysis(df))
        help_target = styles.help_box(
            "<b>Target Selection:</b> met à jour l'état global pour l'apprentissage supervisé.<br>"
            "Analyse la relation entre n'importe quelle feature et la cible sélectionnée.", "#ec4899")
        tab_target = widgets.VBox([
            help_target,
            widgets.HBox([self.target_dd, self.target_feature_dd, self.target_btn],
                          layout=widgets.Layout(align_items="center", gap="10px")),
            self.target_out])

        # ── Tab 2 : Univariate ────────────────────────────────────────────────
        eda_cfg = self.state.config.get("eda", {})
        self.uni_col  = widgets.Dropdown(options=cols, description="Variable:", layout=styles.LAYOUT_DD)
        self.uni_hue  = widgets.Dropdown(options=["None"] + cols, value="None", description="Hue:", layout=styles.LAYOUT_AUTO)
        self.uni_type = widgets.Dropdown(options=eda_cfg.get("univariate_plots", ["auto", "hist", "kde", "box", "violin", "bar", "pie"]),
                                          value="auto", description="Plot:", layout=styles.LAYOUT_AUTO)
        self.uni_bins = widgets.IntSlider(value=30, min=5, max=100, description="Bins/Top N:",
                                          layout=widgets.Layout(width="220px"),
                                          style={"description_width": "initial"})
        self.uni_kde  = widgets.Checkbox(value=True, description="KDE", layout=styles.LAYOUT_AUTO)
        self.uni_log  = widgets.Checkbox(value=False, description="Log scale", layout=widgets.Layout(width="105px"))
        self.uni_pal  = widgets.Dropdown(options=eda_cfg.get("palettes", ["Set2", "Set1", "viridis", "plasma", "coolwarm"]),
                                          value="Set2", description="Palette:", layout=styles.LAYOUT_AUTO)
        self.uni_btn      = widgets.Button(description="Plot", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
        self.uni_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
        self.uni_out      = widgets.Output()
        self._last_uni_fig = None

        self.uni_btn.on_click(lambda b: self._plot_uni(df))
        self.uni_save_btn.on_click(lambda b: self.dashboard.add(self._last_uni_fig,
            f"Univariate: {self.uni_col.value}" + (f" by {self.uni_hue.value}" if self.uni_hue.value != 'None' else ""))
            if self._last_uni_fig else None)
        help_uni = styles.help_box(
            "<b>Numérique:</b> Histogramme / KDE / Box / Violin.<br>"
            "<b>Catégoriel:</b> Bar ou Pie — fréquences.<br>"
            "<b>Hue:</b> découpe par variable catégorielle.<br>"
            "<b>Save to Dashboard:</b> ajoute le plot au panneau.", "#8b5cf6")
        tab_uni = widgets.VBox([
            help_uni,
            widgets.HBox([self.uni_col, self.uni_hue, self.uni_type],
                          layout=widgets.Layout(align_items="flex-end", gap="10px")),
            widgets.HBox([self.uni_bins, self.uni_kde, self.uni_log, self.uni_pal],
                          layout=widgets.Layout(align_items="center", gap="10px", margin="6px 0 0 0")),
            widgets.HBox([self.uni_btn, self.uni_save_btn],
                          layout=widgets.Layout(align_items="center", gap="8px", margin="8px 0 0 0")),
            self.uni_out])

        # ── Tab 3 : Bivariate ─────────────────────────────────────────────────
        self.bi_x     = widgets.Dropdown(options=cols, description="X:", layout=styles.LAYOUT_DD)
        self.bi_y     = widgets.Dropdown(options=cols, description="Y:", layout=styles.LAYOUT_DD)
        self.bi_type  = widgets.Dropdown(options=eda_cfg.get("bivariate_plots", ["auto", "scatter", "hexbin", "hist2d", "kde", "box", "violin", "strip", "swarm", "heatmap", "stacked_bar", "pie"]),
                                          value="auto", description="Plot:", layout=styles.LAYOUT_AUTO)
        self.bi_hue   = widgets.Dropdown(options=["None"] + cols, value="None", description="Hue:", layout=styles.LAYOUT_AUTO)
        self.bi_alpha = widgets.FloatSlider(value=0.6, min=0.1, max=1.0, step=0.1, description="Alpha:",
                                             layout=widgets.Layout(width="220px"),
                                             style={"description_width": "initial"},
                                             readout_format=".1f")
        self.bi_pal   = widgets.Dropdown(options=eda_cfg.get("palettes", ["Set2", "Set1", "viridis"]),
                                          value="Set2", description="Palette:", layout=styles.LAYOUT_AUTO)
        self.bi_btn      = widgets.Button(description="Plot", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
        self.bi_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
        self.bi_out      = widgets.Output()
        self._last_bi_fig = None

        self.bi_btn.on_click(lambda b: self._plot_bi(df))
        self.bi_save_btn.on_click(lambda b: self.dashboard.add(self._last_bi_fig,
            f"Bivariate: {self.bi_y.value} vs {self.bi_x.value}" + (f" by {self.bi_hue.value}" if self.bi_hue.value != 'None' else ""))
            if self._last_bi_fig else None)
        help_bi = styles.help_box(
            "<b>Num × Num:</b> Scatter, Hexbin, Hist2D, KDE.<br>"
            "<b>Num × Cat:</b> Box, Violin, Strip, Swarm.<br>"
            "<b>Cat × Cat:</b> Heatmap ou Stacked Bar.<br>"
            "<b>Hue:</b> ajoute une 3e dimension catégorielle.", "#10b981")
        tab_bi = widgets.VBox([
            help_bi,
            widgets.HBox([self.bi_x, self.bi_y, self.bi_type],
                          layout=widgets.Layout(align_items="flex-end", gap="10px")),
            widgets.HBox([self.bi_hue, self.bi_pal, self.bi_alpha],
                          layout=widgets.Layout(align_items="flex-end", gap="10px", margin="6px 0 0 0")),
            widgets.HBox([self.bi_btn, self.bi_save_btn],
                          layout=widgets.Layout(align_items="center", gap="8px", margin="8px 0 0 0")),
            self.bi_out])

        # ── Tab 4 : Multivariate ──────────────────────────────────────────────
        self.multi_type = widgets.Dropdown(
            options=eda_cfg.get("multivariate", ["Correlation Matrix", "Pairplot"]),
            value="Correlation Matrix", description="Analysis:", layout=styles.LAYOUT_DD)
        self.multi_hue  = widgets.Dropdown(options=["None"] + cols, value="None", description="Hue (Pair):", layout=styles.LAYOUT_AUTO)
        self.multi_corr = widgets.Dropdown(
            options=eda_cfg.get("correlation_methods", ["pearson", "spearman", "kendall"]),
            value="pearson", description="Method:", layout=styles.LAYOUT_AUTO)
        meta = self.meta[self.current_ds]
        available_kinds: set = set()
        self.multi_col_boxes: dict = {}
        for c in cols:
            k = meta[c]["kind"]
            available_kinds.add(k)
            self.multi_col_boxes[c] = widgets.Checkbox(
                value=bool(pd.api.types.is_numeric_dtype(df[c])),
                description=f"{c} [{k}]",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="auto", margin="1px 4px"))
        btn_layout = widgets.Layout(width="auto", height="26px", margin="2px")
        select_btns = []
        for k in sorted(available_kinds):
            btn = widgets.Button(description=f"All {k}", button_style="info", layout=btn_layout)
            def _make_selector(kind):
                def _select(*a):
                    for cn, cb in self.multi_col_boxes.items():
                        cb.value = (meta[cn]["kind"] == kind)
                return _select
            btn.on_click(_make_selector(k))
            select_btns.append(btn)
        btn_none = widgets.Button(description="Clear all", button_style="warning", layout=btn_layout)
        btn_none.on_click(lambda *a: [setattr(cb, "value", False) for cb in self.multi_col_boxes.values()])
        select_btns.append(btn_none)
        col_boxes_grid = widgets.HBox(list(self.multi_col_boxes.values()),
                                       layout=widgets.Layout(flex_wrap="wrap", gap="4px"))
        multi_cols_container = widgets.VBox([
            widgets.HTML("<div style='font-size:0.82em;font-weight:600;color:#6b7280;margin-bottom:4px;'>COLONNES</div>"),
            widgets.HBox(select_btns, layout=widgets.Layout(flex_wrap="wrap", margin="4px 0 8px 0", gap="4px")),
            col_boxes_grid],
            layout=widgets.Layout(border="1px solid #e5e7eb", border_radius="6px", padding="10px", margin="8px 0"))
        self.multi_btn      = widgets.Button(description="Generate Plot", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="160px", height="32px", margin="4px 0"))
        self.multi_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
        self.multi_out      = widgets.Output()
        self._last_multi_fig = None

        self.multi_btn.on_click(lambda b: self._plot_multivariate(df))
        self.multi_save_btn.on_click(lambda b: self.dashboard.add(self._last_multi_fig,
            f"Multivariate: {self.multi_type.value}") if self._last_multi_fig else None)
        help_multi = styles.help_box(
            "<b>Correlation Matrix:</b> corrélations pairées. Pearson = linéaire; Spearman/Kendall = rang.<br>"
            "<b>Pairplot:</b> scatter + histogrammes. Lent avec >10 colonnes.", "#f59e0b")
        tab_multi = widgets.VBox([
            help_multi,
            widgets.HBox([self.multi_type, self.multi_hue, self.multi_corr],
                          layout=widgets.Layout(align_items="flex-end", gap="10px", margin="0 0 4px 0", flex_wrap="wrap")),
            multi_cols_container,
            widgets.HBox([self.multi_btn, self.multi_save_btn],
                          layout=widgets.Layout(align_items="center", gap="8px", margin="4px 0 0 0")),
            self.multi_out])

        # ── Tab 5 : Compare Sets (optionnel) ──────────────────────────────────
        tab_compare = widgets.VBox([])
        tabular_keys = [k for k, v in self.all_datasets.items() if isinstance(v, pd.DataFrame)]
        if len(tabular_keys) > 1:
            self.comp_ds1 = widgets.Dropdown(options=tabular_keys, value=self.current_ds, description="DS 1:", layout=styles.LAYOUT_DD)
            self.comp_ds2 = widgets.Dropdown(
                options=tabular_keys,
                value=tabular_keys[1] if tabular_keys[0] == self.current_ds else tabular_keys[0],
                description="DS 2:", layout=styles.LAYOUT_DD)
            self.comp_col = widgets.Dropdown(options=[], description="Col:", layout=styles.LAYOUT_DD)

            def _update_comp_cols(*args):
                df1 = self.all_datasets[self.comp_ds1.value]
                df2 = self.all_datasets[self.comp_ds2.value]
                shared = [c for c in df1.columns if c in df2.columns]
                self.comp_col.options = shared
                if shared and self.comp_col.value not in shared:
                    self.comp_col.value = shared[0]

            self.comp_ds1.observe(_update_comp_cols, names="value")
            self.comp_ds2.observe(_update_comp_cols, names="value")
            _update_comp_cols()
            self.comp_btn      = widgets.Button(description="Compare Drift", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
            self.comp_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
            self.comp_out      = widgets.Output()
            self._last_comp_fig = None

            self.comp_btn.on_click(lambda b: self._plot_comparison())
            self.comp_save_btn.on_click(lambda b: self.dashboard.add(self._last_comp_fig,
                f"Drift: {self.comp_col.value} — {self.comp_ds1.value} vs {self.comp_ds2.value}")
                if self._last_comp_fig else None)
            help_comp = styles.help_box(
                "Compare les distributions entre deux datasets (ex. Train vs Test). Détecte le <b>Data Drift</b>.", "#ef4444")
            tab_compare.children = [
                help_comp,
                widgets.HBox([self.comp_ds1, self.comp_ds2, self.comp_col, self.comp_btn, self.comp_save_btn],
                              layout=widgets.Layout(gap="10px", flex_wrap="wrap", align_items="center")),
                self.comp_out]

        tabs_list = [tab_recap, tab_target, tab_uni, tab_bi, tab_multi]
        titles    = ["Quality & Stats", "Target Analysis", "Univariate", "Bivariate", "Multivariate"]
        if len(tabular_keys) > 1:
            tabs_list.append(tab_compare)
            titles.append("Compare Sets")
        self.tabs.children = tabs_list
        for i, title in enumerate(titles):
            self.tabs.set_title(i, title)
        self.dynamic_ui.children = [self.tabs]
        self._fill_recap_tab(df)

    # ── rendering helpers ─────────────────────────────────────────────────────
    def _fill_recap_tab(self, df: pd.DataFrame) -> None:
        with self.out_recap:
            clear_output()
            meta = self.meta[self.current_ds]
            rows = [{"Variable": c, "Type": m["kind"],
                     "Missing": f"{m['missing']} ({m['pct_miss']:.1f}%)",
                     "Unique": m["n_unique"], "Dtype": m["dtype"]}
                    for c, m in meta.items()]
            display(HTML("<b style='color:#374151;font-size:0.9em;'>Column Metadata</b>"))
            display(pd.DataFrame(rows).set_index("Variable"))
            num_cols = df.select_dtypes(include=np.number)
            cat_cols = df.select_dtypes(include=["object", "category"])
            if not num_cols.empty:
                display(HTML("<br><b style='color:#374151;font-size:0.9em;'>Numerical Statistics</b>"))
                display(num_cols.describe().T)
            if not cat_cols.empty:
                display(HTML("<br><b style='color:#374151;font-size:0.9em;'>Categorical Statistics</b>"))
                display(cat_cols.describe().T)

    def _plot_missing(self, df: pd.DataFrame) -> None:
        with self.out_recap:
            missing = df.isna().sum()
            missing = missing[missing > 0]
            if missing.empty:
                print("[OK] No missing values found.")
                return
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
            missing.sort_values(ascending=False).plot(kind="bar", color="#ef4444", ax=ax)
            ax.set_title("Missing Values per Feature"); ax.set_ylabel("Count")
            plt.tight_layout()
            display(fig); plt.close(fig)
            self.dashboard.add(fig, "Missing Values")

    def _plot_uni(self, df: pd.DataFrame) -> None:
        with self.uni_out:
            clear_output(wait=True)
            col    = self.uni_col.value
            kind   = self.meta[self.current_ds][col]["kind"]
            p_type = None if self.uni_type.value == "auto" else self.uni_type.value
            hue    = None if self.uni_hue.value == "None" else self.uni_hue.value
            fig = EDAVisualizerUtils.plot_univariate(
                df, col, kind, plot_type=p_type, bins=self.uni_bins.value,
                kde=self.uni_kde.value, log_scale=self.uni_log.value,
                hue=hue, palette=self.uni_pal.value)
            self._last_uni_fig = fig
            display(fig); plt.close(fig)

    def _plot_bi(self, df: pd.DataFrame) -> None:
        with self.bi_out:
            clear_output(wait=True)
            x_col, y_col = self.bi_x.value, self.bi_y.value
            if x_col == y_col:
                print("[INFO] Sélectionnez deux variables différentes.")
                return
            x_kind = self.meta[self.current_ds][x_col]["kind"]
            y_kind = self.meta[self.current_ds][y_col]["kind"]
            p_type = None if self.bi_type.value == "auto" else self.bi_type.value
            h      = None if self.bi_hue.value == "None" else self.bi_hue.value
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = EDAVisualizerUtils.plot_bivariate(
                    df, x_col, y_col, x_kind, y_kind,
                    plot_type=p_type, hue=h,
                    alpha=self.bi_alpha.value, palette=self.bi_pal.value)
            self._last_bi_fig = fig
            display(fig); plt.close(fig)

    def _plot_multivariate(self, df: pd.DataFrame) -> None:
        with self.multi_out:
            clear_output(wait=True)
            m_type = self.multi_type.value
            cols   = [c for c, cb in self.multi_col_boxes.items() if cb.value]
            if not cols:
                print("[INFO] Sélectionnez au moins une colonne.")
                return
            fig = None
            if m_type == "Correlation Matrix":
                num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
                if len(num_cols) < 2:
                    print(f"[INFO] Besoin d'au moins 2 colonnes numériques (obtenu {len(num_cols)}).")
                    return
                corr = df[num_cols].corr(method=self.multi_corr.value)
                fig, ax = plt.subplots(figsize=(min(max(len(num_cols) * 0.8, 8), 16),
                                                min(max(len(num_cols) * 0.6, 6), 12)))
                fig.patch.set_facecolor("#f8fafc")
                sns.heatmap(corr, annot=len(num_cols) <= 12, cmap="coolwarm",
                            fmt=".2f", center=0, square=True, ax=ax)
                ax.set_title(f"{self.multi_corr.value.capitalize()} Correlation Matrix")
                plt.tight_layout()
                display(fig); plt.close(fig)
            elif m_type == "Pairplot":
                if len(cols) > 10:
                    print("[WARNING] Plus de 10 colonnes — utilisation des 10 premières.")
                    cols = cols[:10]
                h = None if self.multi_hue.value == "None" else self.multi_hue.value
                cols_to_plot = cols + [h] if (h and h not in cols) else cols
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = sns.pairplot(df[cols_to_plot].dropna(), hue=h, palette="Set2")
                    fig = g.fig
                    display(fig); plt.close(fig)
            self._last_multi_fig = fig

    def _plot_target_analysis(self, df: pd.DataFrame) -> None:
        with self.target_out:
            clear_output(wait=True)
            target  = self.target_dd.value
            feature = self.target_feature_dd.value
            if target == "(None)" or not feature:
                print("[INFO] Sélectionnez une Target et une Feature.")
                return
            if target == feature:
                print("[INFO] Target et Feature identiques — trivial.")
                return
            t_kind = self.meta[self.current_ds].get(target, {}).get("kind", "categorical")
            f_kind = self.meta[self.current_ds].get(feature, {}).get("kind", "categorical")
            if t_kind in ("categorical", "binary") and f_kind in ("numeric", "timeseries"):
                p_type = "box"
            elif t_kind in ("numeric", "timeseries") and f_kind in ("categorical", "binary"):
                p_type = "box"
            elif t_kind in ("numeric", "timeseries") and f_kind in ("numeric", "timeseries"):
                p_type = "scatter"
            else:
                p_type = "stacked_bar"
            print(f"[{p_type.capitalize()}] Relation : {feature} vs {target}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = EDAVisualizerUtils.plot_bivariate(
                    df, feature, target, f_kind, t_kind, plot_type=p_type,
                    hue=target if t_kind in ("categorical", "binary") else None,
                    alpha=0.7, palette="Set2")
            display(fig); plt.close(fig)
            self.dashboard.add(fig, f"Target: {feature} vs {target}")

    def _plot_comparison(self) -> None:
        with self.comp_out:
            clear_output(wait=True)
            col = self.comp_col.value
            if not col:
                return
            df1_name, df2_name = self.comp_ds1.value, self.comp_ds2.value
            df1 = self.all_datasets[df1_name][col].dropna()
            df2 = self.all_datasets[df2_name][col].dropna()
            is_num = pd.api.types.is_numeric_dtype(df1) and pd.api.types.is_numeric_dtype(df2)
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
            if is_num:
                sns.kdeplot(df1, fill=True, label=df1_name, color="#3b82f6", alpha=0.5, ax=ax)
                sns.kdeplot(df2, fill=True, label=df2_name, color="#ef4444", alpha=0.5, ax=ax)
                ax.set_title(f"Density Drift for '{col}': {df1_name} vs {df2_name}")
                ax.set_ylabel("Density")
            else:
                v1 = df1.value_counts(normalize=True).rename(df1_name)
                v2 = df2.value_counts(normalize=True).rename(df2_name)
                comp = pd.concat([v1, v2], axis=1).fillna(0).head(15)
                comp.plot(kind="bar", ax=ax, color=["#3b82f6", "#ef4444"])
                ax.set_title(f"Category Distribution Drift for '{col}'")
                ax.set_ylabel("Proportion")
            ax.set_xlabel(col); ax.legend()
            plt.tight_layout()
            display(fig); plt.close(fig)
            self._last_comp_fig = fig

    # ── non-tabular UIs ───────────────────────────────────────────────────────
    def _build_image_ui(self, img) -> None:
        out = widgets.Output()
        self.dynamic_ui.children = [out]
        with out:
            print(f"Format: {getattr(img, 'format', 'Unknown')}  |  "
                  f"Size: {getattr(img, 'size', 'Unknown')}  |  "
                  f"Mode: {getattr(img, 'mode', 'Unknown')}")
            if hasattr(img, "size"):
                scale = min(400 / max(img.size[0], 1), 1)
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                buf = io.BytesIO()
                preview = img if img.mode == "RGB" else img.convert("RGB")
                preview.resize(new_size).save(buf, format="JPEG")
                display(HTML(f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'/>"))
                if img.mode == "RGB":
                    arr = np.array(img)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor("#f8fafc")
                    for i, color in enumerate(("r", "g", "b")):
                        hist, _ = np.histogram(arr[:, :, i].ravel(), bins=256, range=[0, 256])
                        ax.plot(hist, color=color, alpha=0.8)
                        ax.fill_between(range(256), hist, color=color, alpha=0.3)
                    ax.set_title("Color Histogram"); ax.set_xlim([0, 256])
                    plt.tight_layout()
                    display(fig); plt.close(fig)

    def _build_text_ui(self, text: str) -> None:
        out = widgets.Output()
        self.dynamic_ui.children = [out]
        with out:
            print(f"Length: {len(text)} characters\n")
            import re
            from collections import Counter
            words = re.findall(r"\b\w+\b", text.lower())
            if words:
                top = dict(Counter(words).most_common(15))
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor("#f8fafc")
                sns.barplot(x=list(top.values()), y=list(top.keys()), palette="viridis", ax=ax)
                ax.set_title("Top 15 Most Common Words")
                plt.tight_layout()
                display(fig); plt.close(fig)
            print("--- Preview ---")
            print(text[:2000] + ("\n... [TRUNCATED]" if len(text) > 2000 else ""))

    def _build_graph_ui(self, G) -> None:
        out = widgets.Output()
        self.dynamic_ui.children = [out]
        with out:
            import networkx as nx
            print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}  |  Directed: {G.is_directed()}")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor("#f8fafc")
            degrees = [d for n, d in G.degree()]
            sns.histplot(degrees, bins=30, kde=True, color="#10b981", ax=axes[0])
            axes[0].set_title("Node Degree Distribution"); axes[0].set_xlabel("Degree")
            sub_nodes = list(G.nodes())[:100]
            H = G.subgraph(sub_nodes)
            nx.draw(H, node_size=20, alpha=0.6, edge_color="#9ca3af",
                    node_color="#3b82f6", ax=axes[1])
            axes[1].set_title("Graph Sample (100 nodes)")
            plt.tight_layout()
            display(fig); plt.close(fig)

    def _build_ontology_ui(self, g) -> None:
        """EDA complète pour une ontologie RDF/OWL — onglets Stats, Graphe, Triplets, Namespaces, Hiérarchie."""
        from IPython.display import display as _disp
        eda_cfg = getattr(self.state, "config", {}).get("eda", {})
        onto_cfg = eda_cfg.get("ontology", {})

        # ── helpers ───────────────────────────────────────────────────────────
        def _short(uri):
            s = str(uri)
            return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]

        def _ns(uri):
            s = str(uri)
            if "#" in s:
                return s.rsplit("#", 1)[0] + "#"
            parts = s.rsplit("/", 1)
            return parts[0] + "/" if len(parts) > 1 else s

        try:
            from rdflib.namespace import OWL, RDF, RDFS
        except ImportError:
            self.dynamic_ui.children = [widgets.HTML(
                "<div style='padding:12px;color:#b91c1c;'>rdflib non disponible.</div>")]
            return

        # ── collecte des stats ────────────────────────────────────────────────
        n_triples  = len(g)
        classes    = list(g.subjects(RDF.type, OWL.Class))
        props_obj  = list(g.subjects(RDF.type, OWL.ObjectProperty))
        props_dt   = list(g.subjects(RDF.type, OWL.DatatypeProperty))
        props_ann  = list(g.subjects(RDF.type, OWL.AnnotationProperty))
        inds       = list(g.subjects(RDF.type, OWL.NamedIndividual))
        # Namespaces utilisés
        from collections import Counter
        ns_counter: Counter = Counter()
        for s, p, o in g:
            ns_counter[_ns(s)] += 1
            ns_counter[_ns(p)] += 1
        top_ns = ns_counter.most_common(20)

        # ── Tab 0 : Stats récapitulatif ───────────────────────────────────────
        stats_out = widgets.Output()
        with stats_out:
            # Tableau de synthèse
            summary_rows = [
                {"Élément": "Triples totaux",        "Nombre": n_triples},
                {"Élément": "Classes OWL",            "Nombre": len(classes)},
                {"Élément": "Object Properties",      "Nombre": len(props_obj)},
                {"Élément": "Datatype Properties",    "Nombre": len(props_dt)},
                {"Élément": "Annotation Properties",  "Nombre": len(props_ann)},
                {"Élément": "Named Individuals",      "Nombre": len(inds)},
                {"Élément": "Namespaces distincts",   "Nombre": len(ns_counter)},
            ]
            _disp(HTML("<b style='color:#374151;font-size:0.9em;'>Résumé de l'ontologie</b>"))
            _disp(pd.DataFrame(summary_rows).set_index("Élément"))

            # Tableau des classes
            if classes:
                _disp(HTML("<br><b style='color:#374151;font-size:0.9em;'>Classes (extrait)</b>"))
                cls_rows = []
                for c in classes[:50]:
                    label_vals = list(g.objects(c, RDFS.label))
                    lbl = str(label_vals[0]) if label_vals else ""
                    n_sub = len(list(g.subjects(RDFS.subClassOf, c)))
                    cls_rows.append({"IRI (court)": _short(c), "Label": lbl, "Sous-classes": n_sub})
                _disp(pd.DataFrame(cls_rows))

            # Tableau des propriétés
            all_props = [(p, "Object") for p in props_obj] + [(p, "Datatype") for p in props_dt]
            if all_props:
                _disp(HTML("<br><b style='color:#374151;font-size:0.9em;'>Propriétés (extrait)</b>"))
                prop_rows = []
                for p, ptype in all_props[:40]:
                    dom = list(g.objects(p, RDFS.domain))
                    rng = list(g.objects(p, RDFS.range))
                    prop_rows.append({
                        "Propriété": _short(p), "Type": ptype,
                        "Domain": _short(dom[0]) if dom else "",
                        "Range":  _short(rng[0]) if rng else "",
                    })
                _disp(pd.DataFrame(prop_rows))

        # ── Tab 1 : Graphe interactif avec densité ────────────────────────────
        graph_out = widgets.Output()
        default_density = int(onto_cfg.get("default_density", 40))
        max_density     = int(onto_cfg.get("max_density", 200))
        density_slider  = widgets.IntSlider(
            value=default_density, min=5, max=max_density, step=5,
            description="Densité (arêtes):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="380px"))
        layout_dd = widgets.Dropdown(
            options=onto_cfg.get("graph_layouts", ["spring", "kamada_kawai", "circular", "shell"]),
            value="spring", description="Layout:",
            layout=widgets.Layout(width="200px"))
        rel_filter = widgets.SelectMultiple(
            options=onto_cfg.get("relation_types",
                                  ["subClassOf", "domain", "range", "type",
                                   "subPropertyOf", "equivalentClass", "inverseOf"]),
            value=onto_cfg.get("relation_types",
                                ["subClassOf", "domain", "range", "type",
                                 "subPropertyOf", "equivalentClass", "inverseOf"]),
            description="Relations:",
            rows=4, layout=widgets.Layout(width="260px"),
            style={"description_width": "initial"})
        plot_btn  = widgets.Button(description="Générer graphe", button_style=styles.BTN_PRIMARY,
                                    layout=styles.LAYOUT_BTN_STD)
        save_btn  = widgets.Button(description="Save Dashboard", button_style="info",
                                    layout=styles.LAYOUT_BTN_STD)
        self._last_onto_fig = None

        def _plot_onto_graph(_=None):
            with graph_out:
                clear_output(wait=True)
                try:
                    import networkx as nx
                    G = nx.DiGraph()
                    allowed = set(rel_filter.value)
                    max_edges = density_slider.value
                    edges_added = 0
                    for s, p, o in g:
                        if edges_added >= max_edges:
                            break
                        ps = _short(p)
                        if ps in allowed:
                            ss, os_ = _short(s), _short(o)
                            if ss and os_ and ss != os_:
                                G.add_edge(ss, os_, label=ps)
                                edges_added += 1
                    if G.number_of_nodes() == 0:
                        print("Aucune relation trouvée avec les filtres sélectionnés.")
                        return
                    fig, ax = plt.subplots(figsize=(12, 7))
                    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
                    lay = layout_dd.value
                    try:
                        if lay == "kamada_kawai":
                            pos = nx.kamada_kawai_layout(G)
                        elif lay == "circular":
                            pos = nx.circular_layout(G)
                        elif lay == "shell":
                            pos = nx.shell_layout(G)
                        else:
                            pos = nx.spring_layout(G, k=2.0, seed=42)
                    except Exception:
                        pos = nx.spring_layout(G, k=2.0, seed=42)
                    # Couleurs par degré entrant/sortant
                    node_colors = []
                    for node in G.nodes():
                        if G.in_degree(node) == 0:
                            node_colors.append("#3b82f6")
                        elif G.out_degree(node) == 0:
                            node_colors.append("#10b981")
                        else:
                            node_colors.append("#8b5cf6")
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                           node_size=500, alpha=0.88, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white",
                                            font_weight="bold", ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8",
                                           arrows=True, arrowsize=14,
                                           connectionstyle="arc3,rad=0.1", ax=ax)
                    edge_labels = nx.get_edge_attributes(G, "label")
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                                  font_size=6, font_color="#475569", ax=ax)
                    ax.set_title(
                        f"Graphe ontologique — {G.number_of_nodes()} nœuds · {G.number_of_edges()} arêtes "
                        f"(/{n_triples} triples) · layout={lay}",
                        fontsize=9, color="#374151")
                    ax.axis("off")
                    # Légende
                    from matplotlib.patches import Patch
                    legend = [Patch(color="#3b82f6", label="Racines"),
                              Patch(color="#8b5cf6", label="Intermédiaires"),
                              Patch(color="#10b981", label="Feuilles")]
                    ax.legend(handles=legend, loc="lower right", fontsize=7,
                              framealpha=0.8, edgecolor="#e5e7eb")
                    plt.tight_layout()
                    _disp(fig)
                    self._last_onto_fig = fig
                    plt.close(fig)
                except ImportError:
                    print("networkx requis pour la visualisation du graphe.")
                except Exception as e:
                    print(f"Erreur graphe: {e}")

        plot_btn.on_click(_plot_onto_graph)
        save_btn.on_click(lambda b: self.dashboard.add(
            self._last_onto_fig, f"Ontologie: graphe ({density_slider.value} arêtes)")
            if self._last_onto_fig else None)

        help_graph = styles.help_box(
            "<b>Densité:</b> nombre max d'arêtes affichées. Réduire pour moins de bruit.<br>"
            "<b>Relations:</b> filtrer les types de relations à visualiser.<br>"
            "<b>Layout:</b> spring = force-directed, kamada_kawai = plus lisible pour petits graphes.", "#8b5cf6")
        graph_tab = widgets.VBox([
            help_graph,
            widgets.HBox([density_slider, layout_dd],
                          layout=widgets.Layout(gap="12px", align_items="center", margin="6px 0")),
            widgets.HBox([rel_filter],
                          layout=widgets.Layout(margin="4px 0 8px 0")),
            widgets.HBox([plot_btn, save_btn],
                          layout=widgets.Layout(gap="8px", align_items="center")),
            graph_out])

        # ── Tab 2 : Triplets filtrables ───────────────────────────────────────
        max_rows_cfg = int(onto_cfg.get("max_triplets_display", 500))
        pred_filter  = widgets.Dropdown(
            options=["(tous)"] + sorted({_short(p) for _, p, _ in g}),
            value="(tous)", description="Prédicat:",
            layout=widgets.Layout(width="260px"),
            style={"description_width": "initial"})
        subj_filter  = widgets.Text(placeholder="Filtrer sujet (contient)...",
                                     description="Sujet:",
                                     layout=styles.LAYOUT_TEXT,
                                     style={"description_width": "initial"})
        filter_btn   = widgets.Button(description="Filtrer", button_style=styles.BTN_INFO,
                                       layout=styles.LAYOUT_BTN_STD)
        triplets_out = widgets.Output()

        def _show_triplets(_=None):
            with triplets_out:
                clear_output(wait=True)
                pred_sel = pred_filter.value
                subj_sel = subj_filter.value.strip().lower()
                rows = []
                for s, p, o in g:
                    ps = _short(p)
                    if pred_sel != "(tous)" and ps != pred_sel:
                        continue
                    ss = _short(s)
                    if subj_sel and subj_sel not in ss.lower():
                        continue
                    rows.append({"Subject": ss, "Predicate": ps,
                                  "Object": _short(o)})
                    if len(rows) >= max_rows_cfg:
                        break
                _disp(HTML(
                    f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:4px;'>"
                    f"{len(rows)} triplets affichés (/{n_triples} total)</div>"))
                _disp(pd.DataFrame(rows) if rows else HTML("<i>Aucun résultat.</i>"))

        filter_btn.on_click(_show_triplets)
        pred_filter.observe(lambda c: _show_triplets(), names="value")
        _show_triplets()

        triplets_tab = widgets.VBox([
            widgets.HBox([pred_filter, subj_filter, filter_btn],
                          layout=widgets.Layout(gap="10px", align_items="center", margin="6px 0")),
            triplets_out])

        # ── Tab 3 : Namespaces ────────────────────────────────────────────────
        ns_out = widgets.Output()
        with ns_out:
            ns_rows = [{"Namespace": ns, "Occurrences": cnt} for ns, cnt in top_ns]
            _disp(HTML("<b style='color:#374151;font-size:0.9em;'>Namespaces les plus utilisés</b>"))
            _disp(pd.DataFrame(ns_rows))
            # Barplot
            if top_ns:
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
                labels = [ns.split("/")[-2] if ns.endswith("/") else ns.split("#")[0].rsplit("/", 1)[-1]
                          for ns, _ in top_ns[:15]]
                values = [cnt for _, cnt in top_ns[:15]]
                ax.barh(labels[::-1], values[::-1], color="#3b82f6", alpha=0.8)
                ax.set_title("Top namespaces par occurrences", fontsize=9)
                ax.set_xlabel("Occurrences")
                plt.tight_layout()
                _disp(fig); plt.close(fig)

        # ── Tab 4 : Hiérarchie de classes ─────────────────────────────────────
        hier_out = widgets.Output()
        max_hier = int(onto_cfg.get("max_hierarchy_nodes", 60))

        def _build_hierarchy(_=None):
            with hier_out:
                clear_output(wait=True)
                try:
                    import networkx as nx
                    H = nx.DiGraph()
                    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
                        ss, os_ = _short(s), _short(o)
                        if ss and os_ and ss != os_:
                            H.add_edge(os_, ss)  # parent → enfant
                        if H.number_of_nodes() >= max_hier:
                            break
                    if H.number_of_nodes() == 0:
                        print("Aucune relation subClassOf trouvée.")
                        return
                    # Essayer un layout hiérarchique
                    try:
                        from networkx.drawing.nx_agraph import graphviz_layout
                        pos = graphviz_layout(H, prog="dot")
                    except Exception:
                        pos = nx.spring_layout(H, k=2.5, seed=42)
                    fig, ax = plt.subplots(figsize=(12, 7))
                    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
                    roots = [n for n in H.nodes() if H.in_degree(n) == 0]
                    leaves = [n for n in H.nodes() if H.out_degree(n) == 0]
                    node_colors = [
                        "#3b82f6" if n in roots else
                        "#10b981" if n in leaves else "#8b5cf6"
                        for n in H.nodes()
                    ]
                    nx.draw_networkx_nodes(H, pos, node_color=node_colors,
                                           node_size=450, alpha=0.88, ax=ax)
                    nx.draw_networkx_labels(H, pos, font_size=7, font_color="white",
                                            font_weight="bold", ax=ax)
                    nx.draw_networkx_edges(H, pos, edge_color="#94a3b8",
                                           arrows=True, arrowsize=12, ax=ax)
                    ax.set_title(
                        f"Hiérarchie de classes (subClassOf) — {H.number_of_nodes()} classes",
                        fontsize=9, color="#374151")
                    ax.axis("off")
                    from matplotlib.patches import Patch
                    legend = [Patch(color="#3b82f6", label="Racines (Thing)"),
                              Patch(color="#8b5cf6", label="Intermédiaires"),
                              Patch(color="#10b981", label="Feuilles")]
                    ax.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.8)
                    plt.tight_layout()
                    _disp(fig); plt.close(fig)
                except ImportError:
                    print("networkx requis.")
                except Exception as e:
                    print(f"Erreur hiérarchie: {e}")

        hier_btn = widgets.Button(description="Générer hiérarchie", button_style=styles.BTN_PRIMARY,
                                   layout=styles.LAYOUT_BTN_STD)
        hier_btn.on_click(_build_hierarchy)
        hier_tab = widgets.VBox([
            styles.help_box(
                f"Visualise la hiérarchie <b>subClassOf</b> (max {max_hier} nœuds). "
                "Configurable via <code>eda.ontology.max_hierarchy_nodes</code>.", "#10b981"),
            widgets.HBox([hier_btn], layout=widgets.Layout(margin="6px 0")),
            hier_out])

        # ── Assemblage des onglets ────────────────────────────────────────────
        onto_tabs = widgets.Tab(children=[stats_out, graph_tab, triplets_tab, ns_out, hier_tab])
        for i, title in enumerate(["Stats", "Graphe", "Triplets", "Namespaces", "Hiérarchie"]):
            onto_tabs.set_title(i, title)

        self.dynamic_ui.children = [onto_tabs]

    def _build_timeseries_ui(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            self.dynamic_ui.children = [widgets.HTML("<div style='padding:12px;'>Time series non-DataFrame.</div>")]
            return
        num_cols = list(df.select_dtypes(include=np.number).columns)
        all_cols = list(df.columns)
        self.ts_col      = widgets.Dropdown(options=num_cols, description="Measurement:", layout=styles.LAYOUT_DD)
        self.ts_time_col = widgets.Dropdown(options=["Index"] + all_cols, value="Index", description="Time Col:", layout=styles.LAYOUT_DD)
        eda_cfg = getattr(self.state, "config", {}).get("eda", {})
        self.ts_type     = widgets.Dropdown(options=eda_cfg.get("ts_plots", ["Line", "Scatter", "Area", "Box (par mois/année)", "Autocorrélation"]),
                                             value="Line", description="Plot Type:", layout=styles.LAYOUT_DD)
        self.ts_window   = widgets.IntSlider(value=1, min=1, max=365, description="Rolling Window:",
                                              layout=widgets.Layout(width="250px"),
                                              style={"description_width": "initial"})
        self.ts_btn      = widgets.Button(description="Plot", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
        self.ts_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
        self.ts_out      = widgets.Output()
        self._last_ts_fig = None

        self.ts_btn.on_click(lambda b: self._plot_timeseries(df))
        self.ts_save_btn.on_click(lambda b: self.dashboard.add(self._last_ts_fig,
            f"TimeSeries: {self.ts_col.value} ({self.ts_type.value})") if self._last_ts_fig else None)
        help_ts = styles.help_box(
            "<b>Line/Area:</b> évolution temporelle. Rolling Window > 1 lisse le bruit.<br>"
            "<b>Box (par mois/année):</b> nécessite un index datetime.<br>"
            "<b>Autocorrélation:</b> relation entre une valeur et son lag.", "#06b6d4")
        self.dynamic_ui.children = [widgets.VBox([
            help_ts,
            widgets.HBox([self.ts_time_col, self.ts_col],
                          layout=widgets.Layout(align_items="flex-end", gap="10px")),
            widgets.HBox([self.ts_type, self.ts_window, self.ts_btn, self.ts_save_btn],
                          layout=widgets.Layout(align_items="flex-end", gap="10px", margin="6px 0 0 0")),
            self.ts_out])]

    def _plot_timeseries(self, df: pd.DataFrame) -> None:
        with self.ts_out:
            clear_output(wait=True)
            col, time_col = self.ts_col.value, self.ts_time_col.value
            p_type, window = self.ts_type.value, self.ts_window.value
            if not col:
                print("Sélectionnez une colonne de mesure.")
                return
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
            s = df[col] if time_col == "Index" else df.set_index(time_col)[col]
            s = s.dropna()
            x, y = s.index, s.values
            if window > 1:
                y = s.rolling(window).mean().values
            if p_type == "Line":
                ax.plot(x, y, color="#3b82f6", label=f"{col} (window={window})")
                ax.legend()
            elif p_type == "Scatter":
                ax.scatter(x, y, color="#8b5cf6", alpha=0.5, s=10)
            elif p_type == "Area":
                ax.fill_between(x, y, color="#10b981", alpha=0.3)
                ax.plot(x, y, color="#10b981")
            elif p_type == "Box (par mois/année)":
                if not pd.api.types.is_datetime64_any_dtype(s.index):
                    print("L'index doit être datetime pour le box plot saisonnier.")
                    return
                sns.boxplot(x=s.index.month, y=y, palette="Set2", ax=ax)
                ax.set_xlabel("Month")
            elif p_type == "Autocorrélation":
                lag = window
                ax.scatter(y[:-lag], y[lag:], alpha=0.5)
                ax.set_xlabel("y(t)"); ax.set_ylabel(f"y(t+{lag})")
                ax.set_title(f"Autocorrelation (lag={lag})")
            if p_type != "Autocorrélation":
                ax.set_title(f"Time Series: {col}")
                ax.set_xlabel("Time"); ax.set_ylabel(col)
            plt.tight_layout()
            display(fig); plt.close(fig)
            self._last_ts_fig = fig


def runner(state) -> UltimateEDA:
    if not hasattr(state, "eda_dashboard"):
        state.eda_dashboard = []
    eda = UltimateEDA(state)
    display(eda.ui)
    return eda
