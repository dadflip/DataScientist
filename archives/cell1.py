import sys
import os
from pathlib import Path
from IPython.display import display, clear_output
import ipywidgets as widgets
import importlib
import pandas as pd

# Ajout du répertoire racine au chemin de recherche des modules
_repo_root = str(Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# --- Définition des groupes de packages sous forme de DataFrame ---
data = [
    {
        "id": "core",
        "label": "Core (always installed)",
        "description": "pandas, numpy, matplotlib, seaborn, scikit-learn, scipy",
        "packages": ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "scipy"],
        "check": ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "scipy"],
        "default": True,
        "required": True,
    },
    {
        "id": "boosting",
        "label": "[Gradient Boosting] (XGBoost, LightGBM, CatBoost)",
        "description": "xgboost, lightgbm, catboost — required for boosting models",
        "packages": ["xgboost", "lightgbm", "catboost"],
        "check": ["xgboost", "lightgbm", "catboost"],
        "default": True,
        "required": False,
    },
    {
        "id": "cat_encoders",
        "label": "[Category Encoders] (BinaryEncoder, LOO, BaseN…)",
        "description": "category_encoders — provides Binary, BaseN, LeaveOneOut, etc.",
        "packages": ["category_encoders"],
        "check": ["category_encoders"],
        "default": True,
        "required": False,
    },
    {
        "id": "torch",
        "label": "[PyTorch] (torch, torchvision)",
        "description": "torch + torchvision — needed for GPU inference, ResNet features, etc.",
        "packages": ["torch", "torchvision"],
        "check": ["torch", "torchvision"],
        "default": False,
        "required": False,
    },
    {
        "id": "transformers",
        "label": "[Transformers] (Transformers & Sentence-Transformers)",
        "description": "HuggingFace transformers + sentence-transformers for embeddings",
        "packages": ["transformers", "sentence-transformers", "tokenizers"],
        "check": ["transformers", "sentence_transformers"],
        "default": False,
        "required": False,
    },
    {
        "id": "nlp_classic",
        "label": "[NLP Classic] (NLTK, spaCy)",
        "description": "NLTK (stemming, stopwords) + spaCy (lemmatization, NER)",
        "packages": ["nltk", "spacy"],
        "check": ["nltk", "spacy"],
        "default": False,
        "required": False,
    },
    {
        "id": "vision",
        "label": "[Computer Vision] (OpenCV, Pillow)",
        "description": "opencv-python-headless + Pillow — image loading, transforms, cv2 ops",
        "packages": ["opencv-python-headless", "Pillow"],
        "check": ["cv2", "PIL"],
        "default": True,
        "required": False,
    },
    {
        "id": "graph",
        "label": "[Graph] (NetworkX)",
        "description": "networkx — graph loading, centrality measures, adjacency matrix",
        "packages": ["networkx"],
        "check": ["networkx"],
        "default": True,
        "required": False,
    },
    {
        "id": "ontology",
        "label": "[Ontology] (rdflib, owlready2)",
        "description": "rdflib, owlready2 — parsing and querying ontologies",
        "packages": ["rdflib", "owlready2"],
        "check": ["rdflib", "owlready2"],
        "default": True,
        "required": False,
    },
    {
        "id": "timeseries",
        "label": "[Time Series] (statsmodels, prophet)",
        "description": "statsmodels (STL, ARIMA, OLS) + prophet (optional forecasting)",
        "packages": ["statsmodels", "prophet"],
        "check": ["statsmodels"],
        "default": False,
        "required": False,
    },
    {
        "id": "dimreduce",
        "label": "[Dimensionality Reduction] (UMAP, TSNE)",
        "description": "umap-learn — UMAP projection used in feature selection config",
        "packages": ["umap-learn"],
        "check": ["umap"],
        "default": False,
        "required": False,
    },
    {
        "id": "arrow",
        "label": "[Arrow / Columnar] (PyArrow, Feather, ORC)",
        "description": "pyarrow — needed to read .feather and .orc files in the loader",
        "packages": ["pyarrow"],
        "check": ["pyarrow"],
        "default": True,
        "required": False,
    },
    {
        "id": "explainability",
        "label": "[Explainability] (SHAP, LIME)",
        "description": "shap + lime — model interpretation, feature importance plots",
        "packages": ["shap", "lime"],
        "check": ["shap", "lime"],
        "default": False,
        "required": False,
    },
    {
        "id": "hpo",
        "label": "[Hyperparameter Tuning] (Optuna, Hyperopt)",
        "description": "optuna + hyperopt — Bayesian / TPE optimisation",
        "packages": ["optuna", "hyperopt"],
        "check": ["optuna", "hyperopt"],
        "default": False,
        "required": False,
    },
    {
        "id": "imbalanced",
        "label": "[Imbalanced Data] (imbalanced-learn)",
        "description": "imbalanced-learn — SMOTE, ADASYN, RandomOverSampler, etc.",
        "packages": ["imbalanced-learn"],
        "check": ["imblearn"],
        "default": False,
        "required": False,
    },
    {
        "id": "plotting",
        "label": "[Plotting Extras] (Plotly, Bokeh)",
        "description": "plotly + bokeh — interactive charts beyond matplotlib/seaborn",
        "packages": ["plotly", "bokeh"],
        "check": ["plotly", "bokeh"],
        "default": False,
        "required": False,
    },
    {
        "id": "tracking",
        "label": "[Experiment Tracking] (MLflow)",
        "description": "mlflow — logging runs, metrics, artifacts, model registry",
        "packages": ["mlflow"],
        "check": ["mlflow"],
        "default": False,
        "required": False,
    },
    {
        "id": "utils",
        "label": "[Utilities] (tqdm, joblib, python-dotenv)",
        "description": "tqdm (progress bars), joblib (parallel), python-dotenv (env vars)",
        "packages": ["tqdm", "joblib", "python-dotenv"],
        "check": ["tqdm", "joblib", "dotenv"],
        "default": True,
        "required": False,
    },
]

# Création du DataFrame
PACKAGE_GROUPS = pd.DataFrame(data).set_index("id")

# --- Fonctions utilitaires ---
def _pip_install(packages: list[str], output_widget) -> tuple[bool, list[str]]:
    """Installe les packages via pip et retourne (succès, liste_des_packages_échoués)."""
    failed = []
    _ipy = get_ipython()
    for pkg in packages:
        with output_widget:
            print(f"   Installing {pkg}...")
        try:
            _ipy.run_line_magic("pip", f"install {pkg} -q")
        except Exception as exc:
            failed.append(pkg)
            with output_widget:
                print(f"   [FAILED] {pkg} — {exc}")
    return len(failed) == 0, failed

def _check_imports(modules: list[str]) -> dict[str, bool]:
    """Vérifie si les modules peuvent être importés. Retourne {module: succès}."""
    return {mod: importlib.import_module(mod) is not None for mod in modules}

# --- Interface utilisateur ---
class InstallerUI:
    def __init__(self):
        self._checkboxes = {}
        self._build_ui()

    def _build_ui(self):
        # En-tête
        header = widgets.HTML("""
            <div style='display:flex; flex-direction:column; gap:4px; margin-bottom:4px;'>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='font-size:1.3em; font-weight:700; color:#6d28d9;'>[Installer]</span>
                    <span style='color:#9ca3af; font-size:0.9em;'>Package Installer</span>
                </div>
                <div style='font-size:0.85em; color:#64748b;'>
                    Select the groups you need, then click Install selected. Required groups are always installed.
                </div>
            </div>
        """)

        # Barre supérieure
        top_bar = widgets.HBox(
            [header],
            layout=widgets.Layout(
                align_items="center",
                justify_content="space-between",
                margin="0 0 12px 0",
                padding="0 0 10px 0",
                border_bottom="2px solid #ede9fe"
            )
        )

        # Tableau Headers
        table_header = widgets.HBox([
            widgets.HTML("<div style='width:30px; font-weight:bold; color:#475569;'>Sel</div>"),
            widgets.HTML("<div style='width:450px; font-weight:bold; color:#475569;'>Package Group / Description</div>"),
            widgets.HTML("<div style='width:90px; font-weight:bold; color:#475569;'>Action</div>"),
            widgets.HTML("<div style='width:160px; font-weight:bold; color:#475569;'>Status</div>")
        ], layout=widgets.Layout(
            border_bottom='2px solid #cbd5e1', 
            padding='0 8px 5px 8px', 
            margin='0 0 5px 0',
            align_items="center"
        ))

        # Liste des groupes de packages
        rows = []
        for i, (grp_id, grp) in enumerate(PACKAGE_GROUPS.iterrows()):
            cb = widgets.Checkbox(
                value=grp["default"] or grp["required"],
                disabled=grp["required"],
                indent=False,
                layout=widgets.Layout(width="30px"),
            )
            self._checkboxes[grp_id] = cb
            lbl = widgets.HTML(
                f'<div style="width:450px;"><span style="font-family:monospace; font-size:0.92em;">'
                f'<b>{grp["label"]}</b><br>'
                f'<span style="color:#666; font-size:0.88em;">{grp["description"]}</span>'
                f'</span></div>'
            )
            check_btn = widgets.Button(
                description="Check",
                button_style="",
                layout=widgets.Layout(width="80px", height="28px", padding="0", margin="0 10px 0 0"),
                tooltip="Verify if these packages are already installed",
            )
            status_html = widgets.HTML(
                "<div style='width:160px;'><span style='color:#aaa; font-size:0.8em;'>not checked</span></div>",
            )
            check_btn.on_click(self._make_checker(grp_id, status_html))
            
            # Application d'un zébra-striping léger pour identifier les lignes du tableau
            bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
            
            row = widgets.HBox(
                [cb, lbl, check_btn, status_html],
                layout=widgets.Layout(
                    align_items="center",
                    padding="6px 8px",
                    border_bottom="1px solid #f1f5f9",
                    background_color=bg_color,
                    margin="0",
                ),
            )
            rows.append(row)

        groups_box = widgets.VBox([table_header] + rows, layout=widgets.Layout(width="800px", border="1px solid #e2e8f0", border_radius="6px"))

        # Boutons de sélection rapide
        sel_all = widgets.Button(description="Select all", button_style="", layout=widgets.Layout(width="max-content", padding="0 10px"))
        sel_def = widgets.Button(description="Defaults", button_style="", layout=widgets.Layout(width="max-content", padding="0 10px"))
        sel_non = widgets.Button(description="Deselect all", button_style="", layout=widgets.Layout(width="max-content", padding="0 10px"))
        chk_all = widgets.Button(description="Check all", button_style="info", layout=widgets.Layout(width="max-content", padding="0 10px"))

        sel_all.on_click(lambda _: self._quick_select("all"))
        sel_def.on_click(lambda _: self._quick_select("default"))
        sel_non.on_click(lambda _: self._quick_select("none"))
        chk_all.on_click(self._check_all)

        quick_row = widgets.HBox(
            [sel_all, sel_def, sel_non, chk_all],
            layout=widgets.Layout(margin="8px 0 4px 0", gap="6px")
        )

        # Bouton d'installation et barre de progression
        self._install_btn = widgets.Button(
            description="Install selected",
            button_style="primary",
            layout=widgets.Layout(width="max-content", height="38px", margin="0 10px 0 0", padding="0 20px"),
        )
        self._install_btn.on_click(self._on_install)

        self._progress = widgets.IntProgress(
            min=0, max=1, value=0,
            bar_style="info",
            layout=widgets.Layout(width="600px", visibility="hidden"),
        )

        # Zone de log
        self._log = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #ddd",
                max_height="280px",
                overflow_y="auto",
                padding="6px",
                width="800px",
                font_family="monospace",
            )
        )

        # Assemblage de l'interface
        self.ui = widgets.VBox([
            top_bar,
            groups_box,
            quick_row,
            widgets.HBox([self._install_btn, self._progress], layout=widgets.Layout(align_items="center", gap="12px")),
            self._log,
        ], layout=widgets.Layout(
            width="100%",
            max_width="1000px",
            border="1px solid #e5e7eb",
            padding="18px",
            border_radius="10px",
            background_color="#ffffff"
        ))

    def _make_checker(self, grp_id, status_html):
        """Génère une fonction pour vérifier les imports d'un groupe."""
        def _check(_):
            grp = PACKAGE_GROUPS.loc[grp_id]
            result = _check_imports(grp["check"])
            ok_all = all(result.values())
            parts = [f"[OK] {mod}" if ok else f"[FAIL] {mod}" for mod, ok in result.items()]
            color = "#2e7d32" if ok_all else "#c62828"
            status_html.value = f'<span style="font-size:0.78em; color:{color};">{" &nbsp; ".join(parts)}</span>'
        return _check

    def _check_all(self, _):
        """Vérifie tous les groupes."""
        with self._log:
            clear_output()
            print("Checking all groups...")
            for grp_id, grp in PACKAGE_GROUPS.iterrows():
                result = _check_imports(grp["check"])
                ok_all = all(result.values())
                icon = "[OK]" if ok_all else "[FAIL]"
                status = " | ".join(f"{'OK' if v else 'FAIL'} {k}" for k, v in result.items())
                print(f"  {icon} {grp['label']:50s}  {status}")

    def _quick_select(self, mode: str):
        """Sélection rapide des groupes."""
        for grp_id, grp in PACKAGE_GROUPS.iterrows():
            cb = self._checkboxes[grp_id]
            if grp["required"]:
                continue
            if mode == "all":
                cb.value = True
            elif mode == "none":
                cb.value = False
            elif mode == "default":
                cb.value = grp["default"]

    def _on_install(self, _):
        """Installe les packages sélectionnés."""
        selected = [
            grp for grp_id, grp in PACKAGE_GROUPS.iterrows()
            if self._checkboxes[grp_id].value or grp["required"]
        ]
        total_pkgs = sum(len(grp["packages"]) for grp in selected)
        self._progress.max = max(total_pkgs, 1)
        self._progress.value = 0
        self._progress.layout.visibility = "visible"
        self._install_btn.disabled = True

        with self._log:
            clear_output()
            print(f"Installing {len(selected)} group(s) — {total_pkgs} package spec(s)...")
            print("=" * 60)
            overall_ok = True
            installed_count = 0
            all_failed = []

            for grp in selected:
                with self._log:
                    print(f"\n[GROUP] {grp['label']}")
                ok, failed = _pip_install(grp["packages"], self._log)
                installed_count += len(grp["packages"]) - len(failed)
                self._progress.value += len(grp["packages"])
                all_failed.extend(failed)
                if not ok:
                    overall_ok = False

            print("\n" + "=" * 60)
            print("Verifying imports...")
            any_fail = False
            for grp in selected:
                result = _check_imports(grp["check"])
                ok_all = all(result.values())
                icon = "[OK]" if ok_all else "[WARN]"
                mods = " | ".join(f"{'OK' if v else 'FAIL'} {k}" for k, v in result.items())
                print(f"  {icon} {grp.name:20s}  {mods}")
                if not ok_all:
                    any_fail = True

            print("\n" + "=" * 60)
            if not any_fail and overall_ok:
                print("[SUCCESS] All packages installed and importable. Ready to run 00_environment.py!")
            else:
                print("[WARNING] Some packages could not be verified. Check the log above.")
                if all_failed:
                    print(f"   Failed specs: {all_failed}")

        self._progress.bar_style = "success" if (overall_ok and not any_fail) else "danger"
        self._install_btn.disabled = False

def runner():
    """Affiche l'interface de l'installateur."""
    ui = InstallerUI()
    display(ui.ui)
    return ui

_installer = runner()