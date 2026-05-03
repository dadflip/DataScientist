import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from ml_pipeline.cell_0d_styles import styles
except ImportError:
    if 'styles' in globals():
        styles = globals()['styles']
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "styles", os.path.join(os.path.dirname(__file__), "cell_0d_styles.py"))
        styles = importlib.util.module_from_spec(spec)
        sys.modules["styles"] = styles
        spec.loader.exec_module(styles)

_PAL  = ['#6366f1','#10b981','#f59e0b','#ef4444','#3b82f6','#8b5cf6','#ec4899']
_GRAY = '#64748b'
_BG   = '#f8fafc'
_GRID = '#e2e8f0'


def _is_valid(arr):
    if arr is None: return False
    if isinstance(arr, pd.DataFrame) and arr.empty: return False
    if isinstance(arr, pd.Series)    and arr.empty: return False
    if isinstance(arr, np.ndarray)   and arr.size == 0: return False
    return True

def _is_inference_mode(splits):
    return not _is_valid(splits.get('y_test')) and _is_valid(splits.get('X_test'))

def _resolve_eval_data(splits, predictions, model_name):
    pred = (predictions or {}).get(model_name, {})
    Xv = pred.get('X_val');  yv = pred.get('y_val')
    if not _is_valid(Xv) or not _is_valid(yv):
        Xv = splits.get('X_test'); yv = splits.get('y_test')
    return Xv, yv

def _section(title, color='#6366f1'):
    return widgets.HTML(
        f"<div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;'>"
        f"<div style='width:4px;height:20px;background:{color};border-radius:2px;'></div>"
        f"<span style='font-size:0.95em;font-weight:700;color:#1e293b;'>{title}</span></div>")

def _warn(msg):
    return widgets.HTML(
        f"<div style='color:#92400e;background:#fffbeb;border-left:4px solid #f59e0b;"
        f"padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")

def _info(msg):
    return widgets.HTML(
        f"<div style='color:#1e40af;background:#eff6ff;border-left:4px solid #3b82f6;"
        f"padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")

def _success(msg):
    return widgets.HTML(
        f"<div style='color:#065f46;background:#ecfdf5;border-left:4px solid #10b981;"
        f"padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")

def _metric_card(name, value, color='#6366f1', subtitle=''):
    return (f"<div style='background:#ffffff;border:1px solid #e2e8f0;"
            f"border-top:3px solid {color};border-radius:6px;padding:12px 16px;"
            f"text-align:center;min-width:130px;'>"
            f"<div style='font-size:0.7em;text-transform:uppercase;"
            f"letter-spacing:0.08em;color:#94a3b8;margin-bottom:4px;'>{name}</div>"
            f"<div style='font-size:1.4em;font-weight:700;color:#1e293b;'>{value}</div>"
            f"<div style='font-size:0.72em;color:#94a3b8;margin-top:2px;'>{subtitle}</div>"
            f"</div>")

def _fig(w=10, h=5, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    ax.grid(color=_GRID, linewidth=0.8, zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(_GRID)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# SÉRIALISATION DE L'ESPACE DE RECHERCHE
# ══════════════════════════════════════════════════════════════════════════════
# Problème original : repr() de scipy.stats.randint/uniform génère une repr
# qui n'est PAS re-évaluable directement (ex: "randint(low=2, high=15)" avec
# des kwargs nommés qui varient selon la version de scipy).
#
# Solution : on sérialise l'espace avec une repr lisible et re-parsable :
#   - randint(a, b)   → "randint(a, b)"     (positional, toujours valide)
#   - uniform(a, b)   → "uniform(a, b)"
#   - listes Python   → repr() standard      (toujours valide)
#
# Le parser évalue dans un namespace enrichi {randint, uniform, np, None}.

def _serialize_search_space(space: dict) -> str:
    """
    Convertit un dict d'espace de recherche en texte Python lisible et re-parsable.
    Gère scipy.stats.randint/uniform et les listes Python.
    """
    from scipy.stats import randint, uniform
    lines = ["{\n"]
    for k, v in space.items():
        cls = type(v).__name__
        # scipy randint : args = (low, high)
        if hasattr(v, 'a') and hasattr(v, 'b') and cls in ('randint_frozen', 'randint'):
            serialized = f"randint({v.a}, {v.b})"
        # scipy uniform : args = (loc, scale)
        elif hasattr(v, 'args') and cls in ('uniform_frozen', 'uniform'):
            loc, scale = v.args if v.args else (v.kwds.get('loc', 0), v.kwds.get('scale', 1))
            serialized = f"uniform({loc}, {scale})"
        # Fallback générique pour d'autres distributions scipy (rv_frozen)
        elif hasattr(v, 'dist') and hasattr(v, 'args'):
            dist_name = v.dist.name
            args_str = ', '.join(repr(a) for a in v.args)
            serialized = f"{dist_name}({args_str})"
        else:
            # listes, None, scalaires : repr() standard fonctionne toujours
            serialized = repr(v)
        lines.append(f"    '{k}': {serialized},\n")
    lines.append("}")
    return ''.join(lines)


def _parse_search_space(code: str) -> dict:
    """
    Évalue l'espace de recherche depuis une chaîne Python.
    Supporte : randint(a,b), uniform(a,b), listes, None, np.*, scalaires.
    Lève ValueError avec un message clair en cas d'erreur de syntaxe.
    """
    from scipy.stats import randint, uniform
    code = code.strip()
    # Ignorer si vide ou uniquement des commentaires
    if not code or all(line.strip().startswith('#') for line in code.splitlines() if line.strip()):
        return {}
    try:
        ns = {
            'randint':  randint,
            'uniform':  uniform,
            'np':       np,
            'None':     None,
            'True':     True,
            'False':    False,
        }
        result = eval(code, {"__builtins__": {}}, ns)
        if not isinstance(result, dict):
            raise ValueError("L'espace de recherche doit être un dict Python { ... }")
        return result
    except SyntaxError as e:
        raise ValueError(
            f"Syntaxe invalide à la ligne {e.lineno} : {e.msg}\n"
            f"Vérifiez que l'espace est un dict Python valide.\n"
            f"Exemple valide :\n"
            f"{{\n"
            f"    'C': uniform(0.01, 10),\n"
            f"    'max_iter': [500, 1000],\n"
            f"}}"
        )
    except Exception as e:
        raise ValueError(f"Erreur d'évaluation : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ESPACES DE RECHERCHE PRÉDÉFINIS
# ══════════════════════════════════════════════════════════════════════════════

def _default_search_space(model_class_name: str) -> dict:
    from scipy.stats import randint, uniform
    spaces = {
        'LogisticRegression': {
            'C':        uniform(0.01, 10),
            'solver':   ['lbfgs', 'saga'],
            'max_iter': [500, 1000, 2000],
        },
        'RandomForestClassifier': {
            'n_estimators':      randint(50, 400),
            'max_depth':         [None, 5, 10, 15, 20],
            'min_samples_split': randint(2, 15),
            'min_samples_leaf':  randint(1, 8),
            'max_features':      ['sqrt', 'log2', 0.5],
        },
        'RandomForestRegressor': {
            'n_estimators':      randint(50, 400),
            'max_depth':         [None, 5, 10, 15, 20],
            'min_samples_split': randint(2, 15),
            'min_samples_leaf':  randint(1, 8),
            'max_features':      ['sqrt', 'log2', 0.5],
        },
        'GradientBoostingClassifier': {
            'n_estimators':  randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth':     randint(2, 8),
            'subsample':     uniform(0.6, 0.4),
        },
        'GradientBoostingRegressor': {
            'n_estimators':  randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth':     randint(2, 8),
            'subsample':     uniform(0.6, 0.4),
        },
        'XGBClassifier': {
            'n_estimators':     randint(50, 400),
            'learning_rate':    uniform(0.01, 0.3),
            'max_depth':        randint(3, 10),
            'subsample':        uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma':            uniform(0, 5),
            'reg_alpha':        uniform(0, 2),
            'reg_lambda':       uniform(0.5, 2),
        },
        'XGBRegressor': {
            'n_estimators':     randint(50, 400),
            'learning_rate':    uniform(0.01, 0.3),
            'max_depth':        randint(3, 10),
            'subsample':        uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
        },
        'CatBoostClassifier': {
            'iterations':    randint(50, 400),
            'learning_rate': uniform(0.01, 0.3),
            'depth':         randint(3, 10),
            'l2_leaf_reg':   uniform(1, 9),
        },
        'CatBoostRegressor': {
            'iterations':    randint(50, 400),
            'learning_rate': uniform(0.01, 0.3),
            'depth':         randint(3, 10),
            'l2_leaf_reg':   uniform(1, 9),
        },
        'LGBMClassifier': {
            'n_estimators':     randint(50, 400),
            'learning_rate':    uniform(0.01, 0.3),
            'num_leaves':       randint(15, 128),
            'max_depth':        [-1, 5, 10, 15],
            'subsample':        uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
        },
        'LGBMRegressor': {
            'n_estimators':  randint(50, 400),
            'learning_rate': uniform(0.01, 0.3),
            'num_leaves':    randint(15, 128),
            'subsample':     uniform(0.5, 0.5),
        },
        'AdaBoostClassifier': {
            'n_estimators':  randint(30, 200),
            'learning_rate': uniform(0.01, 1.5),
            'algorithm':     ['SAMME', 'SAMME.R'],
        },
        'AdaBoostRegressor': {
            'n_estimators':  randint(30, 200),
            'learning_rate': uniform(0.01, 1.5),
            'loss':          ['linear', 'square', 'exponential'],
        },
        'SVC': {
            'C':      uniform(0.01, 100),
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma':  ['scale', 'auto'],
        },
        'SVR': {
            'C':       uniform(0.01, 100),
            'kernel':  ['rbf', 'poly'],
            'gamma':   ['scale', 'auto'],
            'epsilon': uniform(0.01, 1.0),
        },
        'Ridge': {
            'alpha':  uniform(0.001, 100),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
        },
        'Lasso': {
            'alpha':    uniform(0.001, 10),
            'max_iter': [1000, 2000, 5000],
        },
        'ElasticNet': {
            'alpha':    uniform(0.001, 10),
            'l1_ratio': uniform(0.05, 0.9),
            'max_iter': [1000, 2000],
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (200, 100)],
            'alpha':              uniform(0.0001, 0.1),
            'learning_rate':      ['constant', 'adaptive'],
        },
        'MLPRegressor': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'alpha':              uniform(0.0001, 0.1),
            'learning_rate':      ['constant', 'adaptive'],
        },
    }
    return spaces.get(model_class_name, {})


def _scoring_for_task(task, subtask):
    if task == 'classification':
        return 'roc_auc' if subtask == 'binary' else 'f1_macro'
    return 'r2'


# ══════════════════════════════════════════════════════════════════════════════
# UI PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class OptimizationUI:

    def __init__(self, state):
        self.state       = state
        self.splits      = getattr(state, 'data_splits', {})
        self.models      = getattr(state, 'models', {})
        self.predictions = getattr(state, 'predictions', {})
        self.config      = getattr(state, 'config', {})

        if not self.models:
            self.ui = styles.error_msg(
                "Aucun modèle trouvé. Exécutez d'abord la cellule Modeling.")
            return
        self._build_ui()

    def _detect_task(self):
        y = self.splits.get('y_train')
        prob, sub = 'classification', 'binary'
        if _is_valid(y):
            n = y.nunique() if hasattr(y, 'nunique') else len(np.unique(y))
            if n > 20:   prob, sub = 'regression', 'continuous'
            elif n > 2:  prob, sub = 'classification', 'multiclass'
        return prob, sub

    def _get_train_data(self, model_name):
        pred = self.predictions.get(model_name, {})
        X_tr = pred.get('X_train', self.splits.get('X_train'))
        y_tr = pred.get('y_train', self.splits.get('y_train'))
        return X_tr, y_tr

    def _build_ui(self):
        self.task, self.subtask = self._detect_task()
        scoring_default = _scoring_for_task(self.task, self.subtask)

        header  = widgets.HTML(styles.card_html(
            "Optimisation & Sélection",
            "Hyperparameter Tuning — Meilleur Modèle", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items='center', margin='0 0 12px 0',
            padding='0 0 10px 0', border_bottom='2px solid #ede9fe'))

        self.dd_model = widgets.Dropdown(
            options=list(self.models.keys()),
            description='Modèle à optimiser :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='380px'))
        self.dd_model.observe(self._on_model_change, names='value')

        self.dd_method = widgets.Dropdown(
            options=[('RandomizedSearchCV (recommandé)', 'randomized'),
                     ('GridSearchCV (exhaustif)', 'grid')],
            value='randomized',
            description='Méthode :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='380px'))

        self.int_n_iter = widgets.IntSlider(
            value=20, min=5, max=200, step=5,
            description='n_iter (Random) :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='380px'))

        self.int_cv = widgets.IntSlider(
            value=5, min=2, max=10, step=1,
            description='CV folds :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'))

        self.dd_scoring = widgets.Dropdown(
            options=['roc_auc', 'f1', 'f1_macro', 'f1_weighted',
                     'accuracy', 'precision', 'recall',
                     'r2', 'neg_mean_absolute_error',
                     'neg_mean_squared_error', 'neg_root_mean_squared_error'],
            value=scoring_default,
            description='Scoring :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'))

        self.chk_refit = widgets.Checkbox(
            value=True,
            description='Refit sur toutes les données (X_train complet)',
            layout=widgets.Layout(width='380px'))

        # ── Éditeur d'espace de recherche ─────────────────────────────────────
        # Note : l'éditeur affiche du Python lisible avec randint(a,b) /
        # uniform(a,b) — format garanti re-parsable par _parse_search_space().
        self.search_space_editor = widgets.Textarea(
            placeholder=(
                "# Espace de recherche pré-rempli à la sélection du modèle.\n"
                "# Syntaxe supportée :\n"
                "#   'param': randint(a, b)     # entier aléatoire dans [a, b)\n"
                "#   'param': uniform(loc, scale) # flottant dans [loc, loc+scale)\n"
                "#   'param': [val1, val2, ...]   # liste de valeurs discrètes\n"
            ),
            layout=widgets.Layout(width='100%', height='240px',
                                   font_family='monospace'))
        self._fill_search_space(self.dd_model.value)

        self.btn_reset_space = widgets.Button(
            description='Réinitialiser espace',
            button_style='', layout=widgets.Layout(width='180px', height='28px'))
        self.btn_reset_space.on_click(lambda _: self._fill_search_space(self.dd_model.value))

        self.chk_compare_all = widgets.Checkbox(
            value=True,
            description='Comparer tous les modèles après optimisation',
            layout=widgets.Layout(width='400px'))

        self.btn_run = widgets.Button(
            description="Lancer l'optimisation",
            button_style=styles.BTN_PRIMARY,
            icon='cogs',
            layout=styles.LAYOUT_BTN_LARGE)
        self.btn_run.on_click(self._run_optimization)

        self.dd_best_manual = widgets.Dropdown(
            options=list(self.models.keys()),
            description='Définir comme meilleur :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='320px'))
        self.btn_set_best = widgets.Button(
            description='Définir best_model',
            button_style='warning',
            layout=widgets.Layout(width='200px'))
        self.btn_set_best.on_click(self._set_best_manual)

        self.output = widgets.Output()

        def _on_method_change(change):
            self.int_n_iter.layout.display = \
                'flex' if change.new == 'randomized' else 'none'
        self.dd_method.observe(_on_method_change, names='value')

        self.ui = widgets.VBox([
            top_bar,
            styles.help_box(
                "<b>Optimisation des hyperparamètres</b> par RandomizedSearchCV ou GridSearchCV.<br>"
                "L'espace de recherche est pré-rempli selon le modèle sélectionné. "
                "Syntaxe supportée : <code>randint(a, b)</code>, <code>uniform(loc, scale)</code>, listes Python.<br>"
                "Après optimisation, le meilleur modèle est stocké dans "
                "<code>state.best_model</code> et prêt pour la cellule Prédictions.",
                "#6366f1"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Modèle à optimiser</b>"),
            self.dd_model,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Méthode de recherche</b>"),
            widgets.HBox([self.dd_method, self.int_n_iter],
                          layout=widgets.Layout(gap='16px', align_items='center')),
            widgets.HBox([self.int_cv, self.dd_scoring],
                          layout=widgets.Layout(gap='16px', align_items='center', margin='6px 0')),
            self.chk_refit,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HTML(
                "<b style='color:#374151;font-size:0.9em;'>Espace de recherche</b> "
                "<span style='font-size:0.8em;color:#64748b;'>"
                "— <code>randint(a, b)</code> · <code>uniform(loc, scale)</code> · <code>[v1, v2, ...]</code>"
                "</span>"),
            self.search_space_editor,
            widgets.HBox([self.btn_reset_space],
                          layout=widgets.Layout(margin='4px 0 8px 0')),
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            self.chk_compare_all,
            self.btn_run,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:12px 0;'>"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>"
                          "Sélection manuelle du meilleur modèle</b>"),
            widgets.HBox([self.dd_best_manual, self.btn_set_best],
                          layout=widgets.Layout(gap='12px', align_items='center')),
            self.output,
        ], layout=widgets.Layout(
            width='100%', max_width='1000px',
            border='1px solid #e5e7eb',
            padding='18px', border_radius='10px',
            background_color='#ffffff'))

    def _on_model_change(self, change):
        if change.new:
            self._fill_search_space(change.new)

    def _fill_search_space(self, model_name):
        """Auto-rempli l'éditeur avec un texte Python re-parsable."""
        model = self.models.get(model_name)
        if model is None:
            return
        cls_name = model.__class__.__name__
        space = _default_search_space(cls_name)
        if space:
            # On utilise _serialize_search_space pour garantir un texte valide
            self.search_space_editor.value = _serialize_search_space(space)
        else:
            self.search_space_editor.value = (
                f"# Aucun espace prédéfini pour {cls_name}.\n"
                f"# Exemple :\n"
                f"{{\n"
                f"    'param_name': [val1, val2, val3],\n"
                f"    'learning_rate': uniform(0.01, 0.3),\n"
                f"    'n_estimators': randint(50, 300),\n"
                f"}}"
            )

    def _run_optimization(self, b):
        with self.output:
            clear_output(wait=True)
            model_name = self.dd_model.value
            model      = self.models.get(model_name)
            if model is None:
                display(_warn(f"Modèle '{model_name}' introuvable.")); return

            X_tr, y_tr = self._get_train_data(model_name)
            if not _is_valid(X_tr) or not _is_valid(y_tr):
                display(_warn("X_train / y_train introuvables.")); return

            # ── parse espace de recherche ──────────────────────────────────────
            try:
                param_dist = _parse_search_space(self.search_space_editor.value)
            except ValueError as e:
                display(_warn(str(e))); return

            if not param_dist:
                display(_warn(
                    "L'espace de recherche est vide. "
                    "Remplissez l'éditeur avant de lancer."
                )); return

            method  = self.dd_method.value
            n_iter  = self.int_n_iter.value
            cv_fold = self.int_cv.value
            scoring = self.dd_scoring.value

            display(_info(
                f"Lancement <b>{method}</b> sur <b>{model_name}</b> "
                f"| scoring={scoring} | CV={cv_fold} folds"
                + (f" | n_iter={n_iter}" if method == 'randomized' else " | exhaustif")))

            from sklearn.model_selection import (StratifiedKFold, KFold,
                                                  RandomizedSearchCV, GridSearchCV)
            from sklearn.base import clone

            cv_splitter = (StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=42)
                           if self.task == 'classification'
                           else KFold(n_splits=cv_fold, shuffle=True, random_state=42))

            base_model = clone(model)

            try:
                if method == 'randomized':
                    searcher = RandomizedSearchCV(
                        base_model, param_dist,
                        n_iter=n_iter, scoring=scoring,
                        cv=cv_splitter, n_jobs=-1, verbose=0,
                        random_state=42, refit=True)
                else:
                    searcher = GridSearchCV(
                        base_model, param_dist,
                        scoring=scoring, cv=cv_splitter,
                        n_jobs=-1, verbose=0, refit=True)

                searcher.fit(X_tr, y_tr)
            except Exception as e:
                display(_warn(f"Erreur durant la recherche : {e}")); return

            best_score  = searcher.best_score_
            best_params = searcher.best_params_
            best_model  = searcher.best_estimator_

            display(_section(f"Résultats — {model_name}", '#6366f1'))
            orig_score = self._cv_score_current(model, X_tr, y_tr, scoring, cv_splitter)
            gain = best_score - orig_score

            cards = (
                _metric_card("Score CV original", f"{orig_score:.4f}", _PAL[2], f"{scoring}") +
                _metric_card("Score CV optimisé", f"{best_score:.4f}", _PAL[1], f"{scoring}") +
                _metric_card("Gain", f"{'+' if gain >= 0 else ''}{gain:.4f}",
                              '#10b981' if gain >= 0 else '#ef4444', "optimisé − original")
            )
            display(HTML(
                f"<div style='display:flex;flex-wrap:wrap;gap:10px;margin:10px 0;'>{cards}</div>"))

            display(HTML("<b style='font-size:0.85em;color:#374151;'>Meilleurs paramètres :</b>"))
            params_html = ''.join(
                f"<div style='margin:2px 0;font-size:0.83em;'>"
                f"<span style='color:#6d28d9;font-weight:600;'>{k}</span> = {v}</div>"
                for k, v in best_params.items())
            display(HTML(f"<div style='background:#f8fafc;padding:10px;border-radius:6px;"
                          f"border:1px solid #e2e8f0;'>{params_html}</div>"))

            if self.chk_refit.value:
                display(_info("Refit sur X_train complet…"))
                try:
                    best_model.fit(X_tr, y_tr)
                    display(_success("Refit terminé."))
                except Exception as e:
                    display(_warn(f"Erreur refit : {e}"))

            optimized_name = f"{model_name} (optimisé)"
            self.state.models[optimized_name] = best_model
            self.state.best_model = best_model
            self.state.best_model_name = optimized_name

            self.dd_best_manual.options = list(self.state.models.keys())
            self.dd_best_manual.value   = optimized_name

            Xv, yv = _resolve_eval_data(self.splits, self.predictions, model_name)
            if _is_valid(Xv) and _is_valid(yv):
                try:
                    y_pred_val = best_model.predict(Xv)
                    pred_entry = self.predictions.get(model_name, {}).copy()
                    pred_entry.update({
                        'model':           optimized_name,
                        'y_pred_val':      y_pred_val,
                        'feature_columns': list(X_tr.columns),
                    })
                    self.state.predictions[optimized_name] = pred_entry
                except Exception as e:
                    display(_warn(f"Prédictions val du modèle optimisé : {e}"))

            import joblib
            models_dir = pathlib.Path("models")
            models_dir.mkdir(exist_ok=True)
            safe = optimized_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            path = models_dir / f"{safe}.pkl"
            joblib.dump(best_model, path)
            display(_success(
                f"Modèle optimisé sauvegardé → <b>{path}</b><br>"
                f"Stocké dans <code>state.models['{optimized_name}']</code> "
                f"et <code>state.best_model</code>."))

            self.state.log_step("Optimization", "Hyperparameter Tuning", {
                "model": model_name, "method": method,
                "best_score": best_score, "best_params": str(best_params)})

            self._plot_search_results(searcher, model_name, scoring)

            if self.chk_compare_all.value:
                self._compare_all_models(X_tr, y_tr, scoring, cv_splitter)

    def _cv_score_current(self, model, X_tr, y_tr, scoring, cv):
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        try:
            scores = cross_val_score(clone(model), X_tr, y_tr,
                                      cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()
        except Exception:
            return 0.0

    def _plot_search_results(self, searcher, model_name, scoring):
        try:
            results = searcher.cv_results_
            means   = results['mean_test_score']
            stds    = results['std_test_score']
            ranks   = results['rank_test_score']

            sorted_idx = np.argsort(means)[::-1]
            top_n = min(20, len(means))
            idx   = sorted_idx[:top_n]

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
            ax.grid(color=_GRID, linewidth=0.8, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color(_GRID)

            colors = [_PAL[1] if i == 0 else _PAL[0] for i in range(top_n)]
            bars = ax.barh(range(top_n), means[idx][::-1],
                            xerr=stds[idx][::-1],
                            color=colors[::-1], edgecolor='white',
                            height=0.6, capsize=4)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f"Config #{ranks[i]}" for i in idx[::-1]], fontsize=8)
            ax.set_xlabel(scoring, color=_GRAY)
            ax.set_title(f"Top {top_n} configurations — {model_name}",
                          fontsize=11, fontweight='bold', color='#1e293b')
            for bar, mean in zip(bars, means[idx][::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f'{mean:.4f}', va='center', fontsize=8, color='#1e293b')
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    def _compare_all_models(self, X_tr, y_tr, scoring, cv):
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone

        display(_section("Comparaison globale de tous les modèles", '#ec4899'))
        display(_info("Calcul des scores CV en cours…"))

        scores_dict = {}
        for name, model in self.state.models.items():
            try:
                sc = cross_val_score(clone(model), X_tr, y_tr,
                                      cv=cv, scoring=scoring, n_jobs=-1)
                scores_dict[name] = sc
            except Exception as e:
                scores_dict[name] = None
                display(_warn(f"{name} : erreur CV — {e}"))

        rows = []
        for name, sc in scores_dict.items():
            if sc is not None:
                rows.append({'Modèle': name,
                              f'{scoring} mean': round(sc.mean(), 4),
                              f'{scoring} std':  round(sc.std(),  4),
                              'Min': round(sc.min(), 4),
                              'Max': round(sc.max(), 4)})
        if rows:
            df = pd.DataFrame(rows).set_index('Modèle').sort_values(
                f'{scoring} mean', ascending=False)
            best_row = df.index[0]
            display(df.style
                    .highlight_max(subset=[f'{scoring} mean'], color='#d1fae5')
                    .set_properties(**{'font-size': '0.85em'})
                    .format(precision=4))

            if best_row in self.state.models:
                self.state.best_model      = self.state.models[best_row]
                self.state.best_model_name = best_row
                self.dd_best_manual.options = list(self.state.models.keys())
                self.dd_best_manual.value   = best_row
                display(_success(
                    f"Meilleur modèle sélectionné automatiquement : "
                    f"<b>{best_row}</b> ({scoring}={df.loc[best_row, f'{scoring} mean']:.4f})"))

        valid = {n: sc for n, sc in scores_dict.items() if sc is not None}
        if len(valid) > 1:
            fig, ax = plt.subplots(figsize=(max(8, len(valid) * 1.8), 4))
            fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
            ax.grid(color=_GRID, linewidth=0.8, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color(_GRID)
            bp = ax.boxplot(list(valid.values()),
                             labels=list(valid.keys()),
                             patch_artist=True, widths=0.5)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(_PAL[i % len(_PAL)])
                patch.set_alpha(0.7)
            ax.set_ylabel(scoring, color=_GRAY)
            ax.set_title(f'Distribution CV — {scoring}',
                          fontsize=11, fontweight='bold', color='#1e293b')
            plt.xticks(rotation=20, ha='right', fontsize=9)
            plt.tight_layout(); plt.show()

    def _set_best_manual(self, b):
        with self.output:
            clear_output(wait=True)
            chosen = self.dd_best_manual.value
            if chosen not in self.state.models:
                display(_warn(f"'{chosen}' introuvable dans state.models.")); return
            self.state.best_model      = self.state.models[chosen]
            self.state.best_model_name = chosen
            display(_success(
                f"<code>state.best_model</code> défini sur <b>{chosen}</b>.<br>"
                f"Ce modèle sera utilisé par la cellule Prédictions &amp; Export."))
            self.state.log_step("Optimization", "Best Model Set Manually", {"model": chosen})


def runner(state):
    opt = OptimizationUI(state)
    if hasattr(opt, 'ui'):
        display(opt.ui)
    return opt

try:
    runner(state)
except NameError:
    pass