"""PipelineStyles — styles et templates HTML partagés."""
from __future__ import annotations
import ipywidgets as widgets
from IPython.display import display, HTML


class PipelineStyles:
    """Styles UI unifiés pour tout le pipeline ML."""

    # ── Layouts ───────────────────────────────────────────────────────────────
    LAYOUT_DD        = widgets.Layout(width="300px")
    LAYOUT_DD_LONG   = widgets.Layout(width="400px")
    LAYOUT_TEXT      = widgets.Layout(width="300px")
    LAYOUT_BTN_STD   = widgets.Layout(width="150px")
    LAYOUT_BTN_LARGE = widgets.Layout(width="200px")
    LAYOUT_BOX       = widgets.Layout(padding="10px", border="1px solid #e2e8f0",
                                      margin="10px 0")
    LAYOUT_ROW       = widgets.Layout(align_items="center", gap="10px", margin="5px 0")
    LAYOUT_SECTION   = widgets.Layout(border="2px solid #cbd5e1",
                                      padding="0 0 5px 0", margin="0 0 10px 0")
    LAYOUT_W95       = widgets.Layout(width="95%")
    LAYOUT_AUTO      = widgets.Layout(width="auto")

    # ── Button styles ─────────────────────────────────────────────────────────
    BTN_PRIMARY = "primary"
    BTN_INFO    = "info"
    BTN_SUCCESS = "success"
    BTN_WARNING = "warning"
    BTN_DANGER  = "danger"

    # ── CSS global ────────────────────────────────────────────────────────────
    CSS_GLOBALS = """
    <style>
        .pipeline-card { border:1px solid #e2e8f0; border-radius:6px; background:#fff;
            padding:16px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
        .pipeline-title { font-size:1.1em; font-weight:600; color:#0f172a;
            margin-bottom:8px; display:flex; align-items:center; gap:8px; }
        .pipeline-badge { background:#e0e7ff; color:#3730a3; padding:2px 8px;
            border-radius:12px; font-size:0.75em; font-weight:bold; }
        .pipeline-kv-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
            gap:12px; margin-bottom:12px; }
        .pipeline-kv-cell { background:#f8fafc; padding:8px 12px; border-radius:4px;
            border:1px solid #f1f5f9; }
        .kv-key { font-size:0.75em; text-transform:uppercase; color:#64748b;
            font-weight:600; letter-spacing:0.5px; margin-bottom:2px; }
        .kv-val { font-size:1.1em; font-weight:600; color:#0f172a;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .pipeline-section-title { font-size:0.85em; font-weight:700; color:#475569;
            text-transform:uppercase; margin:16px 0 8px 0;
            border-bottom:1px solid #e2e8f0; padding-bottom:4px; }
        .pipeline-dtype-grid { display:flex; flex-wrap:wrap; gap:6px; }
        .pipeline-dtype-pill { background:#f1f5f9; border:1px solid #e2e8f0;
            padding:2px 8px; border-radius:4px; font-size:0.85em; color:#334155; }
        .success-box { padding:12px; background:#f0fdf4; border-left:4px solid #16a34a;
            margin-bottom:8px; border-radius:4px; color:#15803d; }
        .warning-box { padding:12px; background:#fffbeb; border-left:4px solid #f59e0b;
            margin-bottom:8px; border-radius:4px; color:#b45309; }
        .error-box   { padding:12px; background:#fef2f2; border-left:4px solid #ef4444;
            margin-bottom:8px; border-radius:4px; color:#b91c1c; }
        .info-box    { padding:12px; background:#eff6ff; border-left:4px solid #3b82f6;
            margin-bottom:8px; border-radius:4px; color:#1d4ed8; }
    </style>
    """

    @classmethod
    def apply_globals(cls) -> None:
        display(HTML(cls.CSS_GLOBALS))

    @classmethod
    def card_html(cls, title: str, subtitle: str, content: str) -> str:
        return (
            f"<div class='pipeline-card'>"
            f"<div class='pipeline-title'>{title} "
            f"<span class='pipeline-badge'>{subtitle}</span></div>"
            f"{content}</div>"
        )

    @classmethod
    def success_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='success-box'>{msg}</div>")

    @classmethod
    def error_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='error-box'>{msg}</div>")

    @classmethod
    def warning_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='warning-box'>{msg}</div>")

    @classmethod
    def info_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='info-box'>{msg}</div>")

    @classmethod
    def help_box(cls, content: str, color: str) -> widgets.Accordion:
        """Boîte d'aide repliable (fermée par défaut)."""
        inner = widgets.HTML(
            f"<div style='padding:10px 12px; background:#f8fafc; "
            f"border-left:4px solid {color}; font-size:0.85em; "
            f"color:#475569; line-height:1.6;'>{content}</div>"
        )
        acc = widgets.Accordion(children=[inner])
        acc.set_title(0, "Guide")
        acc.selected_index = None
        acc.layout = widgets.Layout(margin="0 0 12px 0")
        return acc


# Instance globale partagée
styles = PipelineStyles()
