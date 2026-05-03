"""Étape 2 — Contexte métier et domaine ML (BusinessEditorUI)."""
from __future__ import annotations
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles


class BusinessEditorUI:
    """Définition du contexte métier et de la tâche ML."""

    def __init__(self, state):
        self.state = state
        self._build_ui()

    def _build_ui(self) -> None:
        header  = widgets.HTML(styles.card_html("Business Context", "Contexte & Définition du problème", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))

        # Résumé des données chargées
        data_summary = []
        for name, dtype in self.state.data_types.items():
            if dtype == "tabular":
                df   = self.state.data_raw[name]
                cols = len(df.columns) if hasattr(df, "columns") else 0
                rows = len(df) if hasattr(df, "__len__") else 0
                col_names = ", ".join(list(df.columns)[:5]) + ("..." if cols > 5 else "")
                data_summary.append(
                    f"<li><b>{name}</b>: Tabular ({rows:,} rows, {cols} cols) "
                    f"<br><span style='font-size:0.85em;color:#6b7280;'>Cols: {col_names}</span></li>"
                )
            else:
                data_summary.append(f"<li><b>{name}</b>: {dtype.capitalize()}</li>")

        data_hint = (
            f"<div style='margin-top:8px;'><b>Datasets chargés :</b>"
            f"<ul style='margin-top:4px;padding-left:20px;'>"
            f"{''.join(data_summary) if data_summary else '<li><i>Aucune donnée chargée.</i></li>'}"
            f"</ul></div>"
        )
        description_box = widgets.HTML(
            f"<div style='padding:16px;margin-bottom:16px;border:1px solid #e2e8f0;"
            f"border-radius:8px;background:#f8fafc;font-family:sans-serif;color:#334155;'>"
            f"<p style='margin-top:0;font-weight:600;'>Configurez votre projet ML</p>"
            f"{data_hint}</div>"
        )

        # Champs métier
        self.project_name = widgets.Text(description="Titre projet :", placeholder="ex. Predictive Maintenance",
                                          layout=styles.LAYOUT_W95, style={"description_width": "150px"})
        self.problem_ta   = widgets.Textarea(description="Problème :", placeholder="Décrivez le problème métier...",
                                              layout=widgets.Layout(width="95%", height="60px"),
                                              style={"description_width": "150px"})
        self.impact_ta    = widgets.Textarea(description="Impact attendu :", placeholder="ROI, usage en production...",
                                              layout=widgets.Layout(width="95%", height="60px"),
                                              style={"description_width": "150px"})
        self.latency_req  = widgets.Dropdown(
            description="Latence :",
            options=["Real-time (<10ms)", "Online (<100ms)", "Batch (seconds+)", "No rigid constraint"],
            value="No rigid constraint", layout=styles.LAYOUT_W95, style={"description_width": "150px"})
        self.interpretability_req = widgets.Checkbox(
            description="Interprétabilité requise (modèles white-box uniquement)",
            value=False, layout=styles.LAYOUT_W95, style={"description_width": "initial"}, indent=False)

        # Domaine ML
        domain_cfg = self.state.config.get("domains", {}).get("supported", [])
        domain_options = [(d["label"], d["value"]) for d in domain_cfg]
        self.domain_label_map = {d["value"]: d["label"] for d in domain_cfg}
        self.domain_dd = widgets.Dropdown(options=domain_options, description="Domaine ML :",
                                           layout=styles.LAYOUT_W95, style={"description_width": "150px"})
        self.dynamic_settings = widgets.VBox([])
        self.dyn_widgets: dict = {}

        self.btn_save = widgets.Button(description="Valider le contexte", button_style=styles.BTN_PRIMARY,
                                        icon="check", layout=widgets.Layout(width="max-content",
                                                                              padding="4px 20px", margin="10px 0 0 0"))
        self.out_msg = widgets.Output()

        self.domain_dd.observe(self._on_domain_change, names="value")
        self.btn_save.on_click(self._on_save)
        self._on_domain_change({"new": self.domain_dd.value})

        form_box = widgets.VBox([
            widgets.HTML("<b style='color:#334155;'>1. Définition métier</b>"),
            self.project_name, self.problem_ta, self.impact_ta,
            self.latency_req, self.interpretability_req,
            widgets.HTML("<hr style='border:1px solid #e2e8f0;margin:15px 0;'>"),
            widgets.HTML("<b style='color:#334155;'>2. Tâche Machine Learning</b>"),
            self.domain_dd, self.dynamic_settings,
            widgets.HTML("<hr style='border:1px solid #e2e8f0;margin:15px 0;'>"),
            self.btn_save,
        ], layout=widgets.Layout(padding="20px", border="1px solid #f1f5f9",
                                  background_color="#ffffff", border_radius="8px", gap="5px"))

        self.ui = widgets.VBox(
            [top_bar, description_box, form_box, self.out_msg],
            layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb",
                                   padding="18px", border_radius="10px", background_color="#ffffff")
        )

    def _get_tabular_columns(self) -> list[str]:
        cols = ["(None)"]
        for v in self.state.data_raw.values():
            if isinstance(v, pd.DataFrame):
                cols.extend(list(v.columns))
        return list(dict.fromkeys(cols))

    def _on_domain_change(self, change) -> None:
        domain   = change["new"]
        children = []
        self.dyn_widgets = {}
        cols = self._get_tabular_columns()

        def _dd(key, desc, opts, val=None):
            w = widgets.Dropdown(options=opts, value=val or (opts[0] if opts else None),
                                  description=desc, layout=styles.LAYOUT_W95,
                                  style={"description_width": "150px"})
            self.dyn_widgets[key] = w
            return w

        def _combo(key, desc, placeholder=""):
            w = widgets.Combobox(options=cols, description=desc, placeholder=placeholder,
                                  layout=styles.LAYOUT_W95, style={"description_width": "150px"})
            self.dyn_widgets[key] = w
            return w

        def _text(key, desc, placeholder=""):
            w = widgets.Text(description=desc, placeholder=placeholder,
                              layout=styles.LAYOUT_W95, style={"description_width": "150px"})
            self.dyn_widgets[key] = w
            return w

        def _int(key, desc, val=3):
            w = widgets.IntText(value=val, description=desc,
                                 layout=styles.LAYOUT_W95, style={"description_width": "150px"})
            self.dyn_widgets[key] = w
            return w

        if domain == "classification_binary":
            children = [_combo("target", "Variable cible :", "Variable à prédire"),
                        _text("pos_label", "Classe positive :", "ex. 1 ou 'Yes'"),
                        _dd("metric", "Métrique :", ["F1-Score", "ROC-AUC", "Accuracy", "Precision", "Recall"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes séparées par virgule")]
        elif domain == "classification_multiclass":
            children = [_combo("target", "Variable cible :", "Variable à prédire"),
                        _dd("metric", "Métrique :", ["Macro F1", "Micro F1", "Weighted F1", "Accuracy"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes séparées par virgule")]
        elif domain == "regression_continuous":
            children = [_combo("target", "Variable cible :", "Variable à prédire"),
                        _dd("metric", "Métrique :", ["RMSE", "MAE", "R2", "MAPE"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes séparées par virgule")]
        elif domain == "clustering":
            children = [_int("expected_clusters", "Clusters attendus :", 3),
                        _dd("metric", "Métrique :", ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes à ignorer (ex. IDs)")]
        elif domain == "ontology":
            onto_cfg = self.state.config.get("domains", {}).get("tasks_by_domain", {}).get("ontology", {})
            children = [_dd("onto_task", "Tâche :", onto_cfg.get("tasks", ["Consistency Checking"])),
                        _dd("inference_level", "Inférence :", onto_cfg.get("inference", ["RDFS", "OWL-DL"]))]
        elif domain == "nlp":
            nlp_cfg = self.state.config.get("domains", {}).get("tasks_by_domain", {}).get("nlp", {})
            children = [_dd("nlp_task", "Tâche NLP :", nlp_cfg.get("tasks", ["Text Classification"])),
                        _combo("text_col", "Colonne texte :", "Variable texte principale"),
                        _combo("target", "Variable cible :", "(Optionnel) pour NLP supervisé")]
        elif domain == "computer_vision":
            cv_cfg = self.state.config.get("domains", {}).get("tasks_by_domain", {}).get("computer_vision", {})
            children = [_dd("cv_task", "Tâche CV :", cv_cfg.get("tasks", ["Image Classification"])),
                        _text("img_shape", "Taille cible :", "ex. 224,224")]
        elif domain == "timeseries":
            children = [_combo("ts_target", "Cible :", "Variable à prévoir"),
                        _combo("ts_time_col", "Colonne temps :", "Variable datetime"),
                        _int("ts_horizon", "Horizon prévision :", 7),
                        _dd("metric", "Métrique :", ["RMSE", "MAE", "MAPE", "sMAPE"])]

        self.dynamic_settings.children = children

    def _on_save(self, btn) -> None:
        with self.out_msg:
            clear_output()
            dyn_params = {k: w.value for k, w in self.dyn_widgets.items()}
            self.state.business_context = {
                "project_name":    self.project_name.value,
                "domain":          self.domain_dd.value,
                "problem":         self.problem_ta.value,
                "impact":          self.impact_ta.value,
                "latency_req":     self.latency_req.value,
                "interpretability": self.interpretability_req.value,
                "domain_parameters": dyn_params,
            }
            target = dyn_params.get("target") or dyn_params.get("ts_target")
            self.state.business_context["target"] = target
            self.state.log_step("Business Context", "Context Defined", self.state.business_context)

            domain_lbl = self.domain_label_map.get(self.domain_dd.value, self.domain_dd.value)
            dyn_html   = "".join(
                f"<p style='margin:2px 0;'><strong>{k}:</strong> {v}</p>"
                for k, v in dyn_params.items()
            )
            display(HTML(
                f"<div style='margin-top:20px;padding:16px;border:1px solid #10b981;"
                f"border-radius:8px;background:#ecfdf5;font-family:sans-serif;'>"
                f"<b style='color:#047857;font-size:1.1em;'>[OK] Contexte validé</b>"
                f"<div style='margin-top:12px;color:#065f46;font-size:0.95em;'>"
                f"<p><strong>Projet :</strong> {self.project_name.value or '<i>Non spécifié</i>'}</p>"
                f"<p><strong>Domaine ML :</strong> {domain_lbl}</p>"
                f"<p><strong>Problème :</strong> {self.problem_ta.value or '<i>Non spécifié</i>'}</p>"
                f"<hr style='border:1px dashed #6ee7b7;margin:5px 0;'/>"
                f"{dyn_html}</div></div>"
            ))


def runner(state) -> BusinessEditorUI:
    editor = BusinessEditorUI(state)
    display(editor.ui)
    return editor
