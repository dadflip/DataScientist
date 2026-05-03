import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import joblib
import json
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os

class ReportGenerator:
    def __init__(self, state):
        self.state = state
        self._build_ui()

    def _build_ui(self):
        header = widgets.HTML("""
            <div style='display:flex; flex-direction:column; gap:4px; margin-bottom:4px;'>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='font-size:1.3em; font-weight:700; color:#6d28d9;'>[Export]</span>
                    <span style='color:#9ca3af; font-size:0.9em;'>Pipeline Artifact Exporter</span>
                </div>
            </div>
        """)
        top_bar = widgets.HBox(
            [header],
            layout=widgets.Layout(
                align_items='center',
                margin='0 0 12px 0',
                padding='0 0 10px 0',
                border_bottom='2px solid #ede9fe'
            )
        )
        self.btn_export = widgets.Button(description="Generate & Export", button_style="success", layout=widgets.Layout(width="200px"))
        self.btn_export.on_click(self._on_export)
        self.output = widgets.Output()

        self.ui = widgets.VBox([
            top_bar,
            widgets.HTML("<div style='background-color:#10b98110; border-left:4px solid #10b981; padding:10px; margin:10px 0; font-size:0.9em; color:#334155;'><b>Export Artifacts:</b> Creates a Python script from pipeline history, packages your trained models into <code>trained_models.pkl</code>, and generates an HTML execution report.</div>"),
            self.btn_export,
            self.output
        ], layout=widgets.Layout(
                width='100%', max_width='1000px',
                border='1px solid #e5e7eb',
                padding='18px',
                border_radius='10px',
                background_color='#ffffff'
        ))

    def _on_export(self, b):
        with self.output:
            from IPython.display import clear_output; clear_output()
            self.generate_all()

    def generate_all(self):
        self.export_python_script(os.path.join(os.getcwd(), "exported_pipeline.py"))
        self.export_models(os.path.join(os.getcwd(), "trained_models.pkl"))
        self.generate_report(os.path.join(os.getcwd(), "execution_report.html"))
        print("[SUCCESS] Export complete: 'exported_pipeline.py', 'trained_models.pkl', 'execution_report.html'.")

    def export_python_script(self, output_path="exported_pipeline.py"):
        script = [
            "# Auto-generated ML Pipeline Script",
            "import pandas as pd",
            "import numpy as np",
            "import joblib",
            "from sklearn.model_selection import train_test_split",
            ""
        ]
        for item in self.state.history:
            script.append(f"# --- Step: {item['step']} | Action: {item['action']} ---")
            script.append(f"# Parameters used: {json.dumps(item['details'], default=str)}")
            script.append("")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(script))

    def export_models(self, path="trained_models.pkl"):
        if self.state.models:
            joblib.dump(self.state.models, path)
            print(f"Models exported to {path}")
        else:
            print("No models to export.")

    def generate_report(self, path="execution_report.html"):
        html = ["<html><head><meta charset='utf-8'><title>ML Report</title></head><body>"]
        html.append("<h1>Machine Learning Execution Report</h1>")

        if self.state.business_context:
            html.append("<h2>Business Context</h2><ul>")
            for k, v in self.state.business_context.items():
                html.append(f"<li><b>{k}</b>: {v}</li>")
            html.append("</ul>")

        html.append("<h2>Execution History (Pipeline)</h2><ul>")
        for step in self.state.history:
            html.append(f"<li><b>Step: {step['step']}</b> - {step['action']}<br/><small>{json.dumps(step['details'], default=str)}</small></li>")
        html.append("</ul></body></html>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

def runner(state):
    exporter = ReportGenerator(state)
    display(exporter.ui)
    return exporter

try:
    runner(state)
except NameError:
    pass