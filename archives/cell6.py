import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
class BusinessEditorUI:
    """
    User interface for defining the business context and objective.
    Fully adapts to all ML domains and inputs to provide a highly assisted setup.
    """
    def __init__(self, state):
        self.state = state
        if not self.state.config:
            self.ui = widgets.HTML("<div style='color:red;'>[ERROR] Configuration not loaded. Please run Cell 1a (Config) first.</div>")
        else:
            self._build_ui()

    def _build_ui(self):
        header = widgets.HTML(styles.card_html("Business Context", "Context & Problem Definition", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(align_items='center', margin='0 0 12px 0', padding='0 0 10px 0', border_bottom='2px solid #ede9fe'))
        
        # Build Data Summary
        data_summary = []
        for name, dtype in self.state.data_types.items():
            if dtype == 'tabular':
                df = self.state.data_raw[name]
                cols = len(df.columns) if hasattr(df, 'columns') else 0
                rows = len(df) if hasattr(df, '__len__') else 0
                col_names = ", ".join(list(df.columns)[:5]) + ("..." if cols > 5 else "")
                data_summary.append(f"<li style='margin-bottom:4px;'><b>{name}</b>: Tabular (Rows: {rows:,}, Cols: {cols}) <br/><span style='font-size:0.85em;color:#6b7280;'>Cols: {col_names}</span></li>")
            else:
                data_summary.append(f"<li><b>{name}</b>: {dtype.capitalize()}</li>")
        
        data_hint = f"<div style='margin-top:8px;'><b style='color:#374151;'>Loaded Datasets:</b><ul style='margin-top:4px; padding-left:20px;'>{''.join(data_summary) if data_summary else '<li><i>No data loaded yet.</i></li>'}</ul></div>"
        
        description_box = widgets.HTML(f"""
        <div style='padding:16px; margin-bottom:16px; border:1px solid #e2e8f0; border-radius:8px; background-color:#f8fafc; font-family:sans-serif; color:#334155;'>
            <p style='margin-top:0; font-weight:600;'>Configure your machine learning project</p>
            <p style='font-size:0.9em; margin-bottom:6px;'>Define the business objectives, constraints and select the primary ML domain to reveal task-specific settings.</p>
            {data_hint}
        </div>
        """)
        
        # Business Overviews
        self.project_name = widgets.Text(description="Project Title:", placeholder="e.g. Predictive Maintenance", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
        self.problem_ta = widgets.Textarea(description="Problem:", placeholder="Describe the business problem we want to solve...", layout=widgets.Layout(width='95%', height='60px'), style={'description_width': '150px'})
        self.impact_ta = widgets.Textarea(description="Expected Impact:", placeholder="What is the expected ROI or usage of this model in production?", layout=widgets.Layout(width='95%', height='60px'), style={'description_width': '150px'})
        
        self.latency_req = widgets.Dropdown(description="Latency Req:", options=["Real-time (<10ms)", "Online (<100ms)", "Batch (seconds+)", "No rigid constraint"], value="No rigid constraint", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
        self.interpretability_req = widgets.Checkbox(description="Interpretability Required (e.g. White-box models only)", value=False, layout=styles.LAYOUT_W95, style={'description_width': 'initial'}, indent=False)

        domain_cfg = self.state.config.get("domains", [])
        domain_options = [(d["label"], d["value"]) for d in domain_cfg]
        self.domain_label_map = {d["value"]: d["label"] for d in domain_cfg}
        self.domain_dd = widgets.Dropdown(options=domain_options, description="ML Domain:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
        
        self.dynamic_settings = widgets.VBox([])
        
        self.btn_save = widgets.Button(description="Validate Context", button_style=styles.BTN_PRIMARY, icon='check', layout=widgets.Layout(width='max-content', padding='4px 20px', margin='10px 0 0 0'))
        self.out_msg = widgets.Output()
        
        self.domain_dd.observe(self._on_domain_change, names='value')
        self.btn_save.on_click(self._on_save)
        
        self._on_domain_change({'new': self.domain_dd.value})
        
        form_box = widgets.VBox([
            widgets.HTML("<b style='color:#334155;'>1. Business Definition</b>"),
            self.project_name, 
            self.problem_ta,
            self.impact_ta,
            self.latency_req,
            self.interpretability_req,
            widgets.HTML("<hr style='border:1px solid #e2e8f0; margin:15px 0;'>"),
            widgets.HTML("<b style='color:#334155;'>2. Machine Learning Task</b>"),
            self.domain_dd, 
            self.dynamic_settings,
            widgets.HTML("<hr style='border:1px solid #e2e8f0; margin:15px 0;'>"),
            self.btn_save
        ], layout=widgets.Layout(padding='20px', border='1px solid #f1f5f9', background_color='#ffffff', border_radius='8px', gap='5px'))
        
        main_content = widgets.VBox([description_box, form_box, self.out_msg])
        self.ui = widgets.VBox([top_bar, main_content], layout=widgets.Layout(width='100%', max_width='1000px', border='1px solid #e5e7eb', padding='18px', border_radius='10px', background_color='#ffffff'))

    def _get_tabular_columns(self):
        cols = ['(None)']
        for v in self.state.data_raw.values():
            if isinstance(v, pd.DataFrame):
                cols.extend(list(v.columns))
        return list(dict.fromkeys(cols))

    def _on_domain_change(self, change):
        domain = change['new']
        children = []
        self.dyn_widgets = {}
        
        cols = self._get_tabular_columns()
        
        if domain in ["classification_binary"]:
            self.dyn_widgets['target'] = widgets.Combobox(options=cols, description="Target Var:", placeholder="Variable to predict", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['pos_label'] = widgets.Text(description="Pos Class Label:", placeholder="e.g. 1 or 'Yes'", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['metric'] = widgets.Dropdown(options=["F1-Score", "ROC-AUC", "Accuracy", "Precision", "Recall"], value="F1-Score", description="Primary Metric:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['features_exclude'] = widgets.Text(description="Exclude Cols:", placeholder="Comma separated columns to ignore", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['target'], self.dyn_widgets['pos_label'], self.dyn_widgets['metric'], self.dyn_widgets['features_exclude']])
            
        elif domain in ["classification_multiclass"]:
            self.dyn_widgets['target'] = widgets.Combobox(options=cols, description="Target Var:", placeholder="Variable to predict", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['metric'] = widgets.Dropdown(options=["Macro F1", "Micro F1", "Weighted F1", "Accuracy"], value="Macro F1", description="Primary Metric:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['features_exclude'] = widgets.Text(description="Exclude Cols:", placeholder="Comma separated columns to ignore", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['target'], self.dyn_widgets['metric'], self.dyn_widgets['features_exclude']])
            
        elif domain in ["regression_continuous"]:
            self.dyn_widgets['target'] = widgets.Combobox(options=cols, description="Target Var:", placeholder="Variable to predict", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['metric'] = widgets.Dropdown(options=["RMSE", "MAE", "R2", "MAPE"], value="RMSE", description="Primary Metric:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['features_exclude'] = widgets.Text(description="Exclude Cols:", placeholder="Comma separated columns to ignore", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['target'], self.dyn_widgets['metric'], self.dyn_widgets['features_exclude']])
            
        elif domain in ["clustering"]:
            self.dyn_widgets['expected_clusters'] = widgets.IntText(description="Expected Clusters:", value=3, layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['metric'] = widgets.Dropdown(options=["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"], value="Silhouette Score", description="Eval Metric:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['features_exclude'] = widgets.Text(description="Exclude Cols:", placeholder="Columns to ignore (e.g. IDs)", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['expected_clusters'], self.dyn_widgets['metric'], self.dyn_widgets['features_exclude']])
            
        elif domain in ["ontology"]:
            onto_cfg = self.state.config.get("domain_tasks", {}).get("ontology", {})
            self.dyn_widgets['onto_task'] = widgets.Dropdown(options=onto_cfg.get("tasks", ["Link Prediction", "Node Classification", "Graph embeddings"]), description="Task:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['inference_level'] = widgets.Dropdown(options=onto_cfg.get("inference", ["RDFS", "OWL-RL", "None"]), description="Inference:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['onto_task'], self.dyn_widgets['inference_level']])
            
        elif domain in ["nlp"]:
            nlp_cfg = self.state.config.get("domain_tasks", {}).get("nlp", {})
            self.dyn_widgets['nlp_task'] = widgets.Dropdown(options=nlp_cfg.get("tasks", ["Text Classification", "Sentiment Analysis", "NER", "Topic Modeling"]), description="Text Task:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['text_col'] = widgets.Combobox(options=cols, description="Text Col:", placeholder="Main text variable", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['target'] = widgets.Combobox(options=cols, description="Target Var:", placeholder="(Optional) Target for supervised NLP", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['nlp_task'], self.dyn_widgets['text_col'], self.dyn_widgets['target']])
            
        elif domain in ["computer_vision"]:
            cv_cfg = self.state.config.get("domain_tasks", {}).get("computer_vision", {})
            self.dyn_widgets['cv_task'] = widgets.Dropdown(options=cv_cfg.get("tasks", ["Image Classification", "Object Detection", "Image Segmentation", "Image Generation"]), description="Task:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['img_shape'] = widgets.Text(description="Target Size:", placeholder="e.g. 224,224", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['cv_task'], self.dyn_widgets['img_shape']])
            
        elif domain in ["timeseries"]:
            self.dyn_widgets['ts_target'] = widgets.Combobox(options=cols, description="Target:", placeholder="Variable to forecast", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['ts_time_col'] = widgets.Combobox(options=cols, description="Time Col:", placeholder="Datetime variable", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['ts_horizon'] = widgets.IntText(value=7, description="Forecast Horizon:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            self.dyn_widgets['metric'] = widgets.Dropdown(options=["RMSE", "MAE", "MAPE", "sMAPE"], value="RMSE", description="Metric:", layout=styles.LAYOUT_W95, style={'description_width': '150px'})
            children.extend([self.dyn_widgets['ts_target'], self.dyn_widgets['ts_time_col'], self.dyn_widgets['ts_horizon'], self.dyn_widgets['metric']])
            
        self.dynamic_settings.children = children

    def _on_save(self, btn):
        with self.out_msg:
            from IPython.display import clear_output; clear_output()
            dyn_params = {}
            for k, w in self.dyn_widgets.items():
                dyn_params[k] = w.value
            
            self.state.business_context = {
                "project_name": self.project_name.value,
                "domain": self.domain_dd.value,
                "problem": self.problem_ta.value,
                "impact": self.impact_ta.value,
                "latency_req": self.latency_req.value,
                "interpretability": self.interpretability_req.value,
                "domain_parameters": dyn_params
            }
            
            if 'target' in dyn_params:
                self.state.business_context['target'] = dyn_params['target']
            elif 'ts_target' in dyn_params:
                self.state.business_context['target'] = dyn_params['ts_target']
            else:
                self.state.business_context['target'] = None
                
            self.state.log_step("Business Context", "Context Defined", self.state.business_context)
            
            domain_lbl = self.domain_label_map.get(self.domain_dd.value, self.domain_dd.value)
            dyn_html = "".join([f"<p style='margin:2px 0;'><strong>{k}:</strong> <span style='color:#0f172a;'>{v}</span></p>" for k, v in dyn_params.items()])
            
            display(HTML(f"""
            <div style='margin-top:20px; padding:16px; border:1px solid #10b981; border-radius:8px; background-color:#ecfdf5; font-family:sans-serif;'>
                <b style='color:#047857; font-size:1.1em;'>[OK] Business Context Validated & Saved</b>
                <div style='display:flex; flex-direction:column; gap:8px; margin-top:12px; color:#065f46; font-size:0.95em;'>
                    <p style='margin:0;'><strong>Project:</strong> <span style='color:#0f172a;'>{self.project_name.value if self.project_name.value else '<i>Not specified</i>'}</span></p>
                    <p style='margin:0;'><strong>ML Domain:</strong> <span style='color:#0f172a;'>{domain_lbl}</span></p>
                    <p style='margin:0;'><strong>Problem:</strong> <span style='color:#0f172a;'>{self.problem_ta.value if self.problem_ta.value else '<i>Not specified</i>'}</span></p>
                    <p style='margin:0;'><strong>Impact:</strong> <span style='color:#0f172a;'>{self.impact_ta.value if self.impact_ta.value else '<i>Not specified</i>'}</span></p>
                    <p style='margin:0;'><strong>Latency constraint:</strong> <span style='color:#0f172a;'>{self.latency_req.value}</span></p>
                    <p style='margin:0;'><strong>Interpretability required:</strong> <span style='color:#0f172a;'>{"Yes" if self.interpretability_req.value else "No"}</span></p>
                    <hr style='border:1px dashed #6ee7b7; margin:5px 0;'/>
                    {dyn_html}
                </div>
            </div>
            """))
def runner(state):
    editor = BusinessEditorUI(state)
    display(editor.ui)
    return editor
try:
    runner(state)
except NameError:
    pass