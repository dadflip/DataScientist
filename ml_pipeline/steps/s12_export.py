"""Étape 12 — Export & Rapport (ReportGenerator)."""
from __future__ import annotations
import json
import os
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles


class ReportGenerator:
    """Génère le script Python, exporte les modèles et produit un rapport HTML."""

    def __init__(self, state):
        self.state = state
        self._build_ui()

    def _build_ui(self) -> None:
        header  = widgets.HTML(styles.card_html("Export", "Pipeline Artifact Exporter", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.btn_export = widgets.Button(description="Generate & Export",
                                          button_style="success",
                                          layout=widgets.Layout(width="200px"))
        self.btn_export.on_click(self._on_export)
        self.output = widgets.Output()
        self.ui = widgets.VBox([
            top_bar,
            styles.help_box(
                "<b>Export Artifacts :</b> génère un script Python depuis l'historique du pipeline, "
                "exporte les modèles entraînés dans <code>trained_models.pkl</code> et produit un rapport HTML.",
                "#10b981"),
            self.btn_export,
            self.output,
        ], layout=widgets.Layout(width="100%", max_width="1000px",
                                  border="1px solid #e5e7eb", padding="18px",
                                  border_radius="10px", background_color="#ffffff"))

    def _on_export(self, b) -> None:
        with self.output:
            clear_output()
            self.generate_all()

    def generate_all(self) -> None:
        self.export_python_script(os.path.join(os.getcwd(), "exported_pipeline.py"))
        self.export_models(os.path.join(os.getcwd(), "trained_models.pkl"))
        self.generate_report(os.path.join(os.getcwd(), "execution_report.html"))
        display(styles.success_msg(
            "[SUCCESS] Export complet : <b>exported_pipeline.py</b>, "
            "<b>trained_models.pkl</b>, <b>execution_report.html</b>."))

    def export_python_script(self, output_path: str = "exported_pipeline.py") -> None:
        script = [
            "# Auto-generated ML Pipeline Script",
            "import pandas as pd",
            "import numpy as np",
            "import joblib",
            "from sklearn.model_selection import train_test_split",
            "",
        ]
        for item in self.state.history:
            script.append(f"# --- Step: {item['step']} | Action: {item['action']} ---")
            script.append(f"# Parameters used: {json.dumps(item['details'], default=str)}")
            script.append("")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(script))
        print(f"Script exporté → {output_path}")

    def export_models(self, path: str = "trained_models.pkl") -> None:
        if self.state.models:
            import joblib
            joblib.dump(self.state.models, path)
            print(f"Modèles exportés → {path}")
        else:
            print("Aucun modèle à exporter.")

    def generate_report(self, path: str = "execution_report.html") -> None:
        html = ["<html><head><meta charset='utf-8'><title>ML Report</title>",
                "<style>body{font-family:sans-serif;max-width:900px;margin:40px auto;color:#1e293b;}",
                "h1{color:#6d28d9;}h2{color:#374151;border-bottom:1px solid #e2e8f0;padding-bottom:6px;}",
                "li{margin-bottom:4px;}pre{background:#f8fafc;padding:10px;border-radius:6px;font-size:0.85em;}</style>",
                "</head><body>",
                "<h1>Machine Learning Execution Report</h1>"]
        if self.state.business_context:
            html.append("<h2>Business Context</h2><ul>")
            for k, v in self.state.business_context.items():
                html.append(f"<li><b>{k}</b>: {v}</li>")
            html.append("</ul>")
        if self.state.models:
            html.append("<h2>Trained Models</h2><ul>")
            for name in self.state.models:
                html.append(f"<li>{name}</li>")
            html.append("</ul>")
        html.append("<h2>Execution History</h2><ul>")
        for step in self.state.history:
            html.append(f"<li><b>Step: {step['step']}</b> — {step['action']}<br>"
                        f"<pre>{json.dumps(step['details'], default=str, indent=2)}</pre></li>")
        html.append("</ul></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"Rapport exporté → {path}")


def runner(state) -> ReportGenerator:
    exporter = ReportGenerator(state)
    display(exporter.ui)
    return exporter
