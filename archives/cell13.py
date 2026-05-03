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
        spec = importlib.util.spec_from_file_location("styles", os.path.join(os.path.dirname(__file__), "cell_0d_styles.py"))
        styles = importlib.util.module_from_spec(spec)
        sys.modules["styles"] = styles
        spec.loader.exec_module(styles)

_PAL  = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#ec4899']
_GRAY = '#64748b'
_BG   = '#f8fafc'
_GRID = '#e2e8f0'


def _is_valid(arr):
    if arr is None:
        return False
    if isinstance(arr, pd.DataFrame) and arr.empty:
        return False
    if isinstance(arr, pd.Series) and arr.empty:
        return False
    if isinstance(arr, np.ndarray) and arr.size == 0:
        return False
    return True


def _is_inference_mode(splits):
    return not _is_valid(splits.get('y_test')) and _is_valid(splits.get('X_test'))


def _resolve_eval_data(splits, predictions, model_name):
    pred_entry = (predictions or {}).get(model_name, {})
    X_val = pred_entry.get('X_val')
    y_val = pred_entry.get('y_val')
    if not _is_valid(X_val) or not _is_valid(y_val):
        X_val = splits.get('X_test')
        y_val = splits.get('y_test')
    return X_val, y_val


def _make_fast_clone(model):
    from sklearn.base import clone
    fast = clone(model)
    if hasattr(fast, 'n_estimators'):
        fast.set_params(n_estimators=min(30, fast.n_estimators))
    if hasattr(fast, 'n_jobs'):
        fast.set_params(n_jobs=1)
    return fast


def _fig(w=10, h=5, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.grid(color=_GRID, linewidth=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(_GRID)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    return fig, ax


def _multi_fig(rows, cols, w=14, h=4):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h * rows))
    fig.patch.set_facecolor(_BG)
    axes_flat = np.array(axes).flatten() if rows * cols > 1 else [axes]
    for ax in axes_flat:
        ax.set_facecolor(_BG)
        ax.grid(color=_GRID, linewidth=0.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(_GRID)
    return fig, axes_flat


def _metric_card(name, value, color='#6366f1'):
    return (
        f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-top:3px solid {color};"
        f"border-radius:6px; padding:12px 16px; text-align:center; min-width:120px;'>"
        f"<div style='font-size:0.7em; text-transform:uppercase; letter-spacing:0.08em;"
        f"color:#94a3b8; margin-bottom:6px;'>{name}</div>"
        f"<div style='font-size:1.5em; font-weight:700; color:#1e293b;'>{value}</div></div>"
    )


def _section(title, color='#6366f1'):
    return widgets.HTML(
        f"<div style='display:flex; align-items:center; gap:10px; margin:18px 0 10px 0;'>"
        f"<div style='width:4px; height:20px; background:{color}; border-radius:2px;'></div>"
        f"<span style='font-size:0.95em; font-weight:700; color:#1e293b; letter-spacing:0.02em;'>{title}</span>"
        f"</div>"
    )


def _warn(msg):
    return widgets.HTML(
        f"<div style='color:#92400e; background:#fffbeb; border-left:4px solid #f59e0b;"
        f"padding:8px 12px; font-size:0.85em; border-radius:4px;'>{msg}</div>"
    )


def _info(msg):
    return widgets.HTML(
        f"<div style='color:#1e40af; background:#eff6ff; border-left:4px solid #3b82f6;"
        f"padding:8px 12px; font-size:0.85em; border-radius:4px;'>{msg}</div>"
    )


def _compute_metrics(model, X_eval, y_eval, task, subtask, cfg_metrics):
    from sklearn import metrics as skm
    results = {}
    if task == 'classification':
        y_pred = model.predict(X_eval)
        sub_cfg = cfg_metrics.get('classification', {}).get(subtask, cfg_metrics.get('classification', {}).get('binary', []))
        for m in sub_cfg:
            try:
                fn = getattr(skm, m['func'])
                kwargs = m.get('kwargs', {})
                if m['func'] in ('roc_auc_score', 'log_loss'):
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_eval)
                        val = fn(y_eval, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob, **kwargs)
                    else:
                        continue
                else:
                    val = fn(y_eval, y_pred, **kwargs)
                results[m['name']] = val
            except Exception as e:
                results[m['name']] = f'ERR: {e}'
    elif task == 'regression':
        y_pred = model.predict(X_eval)
        for m in cfg_metrics.get('regression', []):
            try:
                fn = getattr(skm, m['func'])
                val = fn(y_eval, y_pred)
                if m.get('post') == 'np.sqrt':
                    val = np.sqrt(val)
                results[m['name']] = val
            except Exception as e:
                results[m['name']] = f'ERR: {e}'
    return results


def _plot_confusion_matrix(model, X_eval, y_eval, model_name, ax=None):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_pred = model.predict(X_eval)
    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    if ax is None:
        fig, ax = _fig(5, 4, f'Confusion Matrix — {model_name}')
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=10, fontweight='bold', color='#1e293b')
    ax.set_facecolor(_BG)
    return ax.figure


def _plot_roc_curves(models_dict, X_eval, y_eval):
    from sklearn.metrics import roc_curve, auc
    fig, ax = _fig(7, 5, 'ROC Curves')
    for i, (name, model) in enumerate(models_dict.items()):
        try:
            if not hasattr(model, 'predict_proba'):
                continue
            y_prob = model.predict_proba(X_eval)
            scores = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.max(axis=1)
            fpr, tpr, _ = roc_curve(y_eval, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=_PAL[i % len(_PAL)], lw=2, label=f'{name} (AUC={roc_auc:.3f})')
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], '--', color=_GRAY, lw=1)
    ax.set_xlabel('False Positive Rate', color=_GRAY)
    ax.set_ylabel('True Positive Rate', color=_GRAY)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    return fig


def _plot_precision_recall(models_dict, X_eval, y_eval):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = _fig(7, 5, 'Precision-Recall Curves')
    for i, (name, model) in enumerate(models_dict.items()):
        try:
            if not hasattr(model, 'predict_proba'):
                continue
            y_prob = model.predict_proba(X_eval)
            scores = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.max(axis=1)
            precision, recall, _ = precision_recall_curve(y_eval, scores)
            ap = average_precision_score(y_eval, scores)
            ax.plot(recall, precision, color=_PAL[i % len(_PAL)], lw=2, label=f'{name} (AP={ap:.3f})')
        except Exception:
            pass
    ax.set_xlabel('Recall', color=_GRAY)
    ax.set_ylabel('Precision', color=_GRAY)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    return fig


def _plot_calibration(models_dict, X_eval, y_eval):
    from sklearn.calibration import calibration_curve
    fig, ax = _fig(7, 5, 'Calibration Curves')
    ax.plot([0, 1], [0, 1], '--', color=_GRAY, lw=1, label='Perfect calibration')
    for i, (name, model) in enumerate(models_dict.items()):
        try:
            if not hasattr(model, 'predict_proba'):
                continue
            y_prob = model.predict_proba(X_eval)[:, 1]
            frac_pos, mean_pred = calibration_curve(y_eval, y_prob, n_bins=10)
            ax.plot(mean_pred, frac_pos, 's-', color=_PAL[i % len(_PAL)], lw=2, label=name)
        except Exception:
            pass
    ax.set_xlabel('Mean predicted probability', color=_GRAY)
    ax.set_ylabel('Fraction of positives', color=_GRAY)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def _plot_residuals(model, X_eval, y_eval, model_name):
    y_pred = model.predict(X_eval)
    y_arr = np.array(y_eval)
    residuals = y_arr - y_pred
    fig, axes = _multi_fig(1, 3, w=14, h=4)
    axes[0].scatter(y_pred, residuals, alpha=0.5, color=_PAL[0], s=20)
    axes[0].axhline(0, color=_PAL[3], lw=1.5, ls='--')
    axes[0].set_xlabel('Predicted', color=_GRAY)
    axes[0].set_ylabel('Residuals', color=_GRAY)
    axes[0].set_title(f'Residuals vs Predicted — {model_name}', fontsize=10, fontweight='bold', color='#1e293b')
    _mn = min(y_arr.min(), y_pred.min())
    _mx = max(y_arr.max(), y_pred.max())
    axes[1].scatter(y_arr, y_pred, alpha=0.5, color=_PAL[1], s=20)
    axes[1].plot([_mn, _mx], [_mn, _mx], '--', color=_PAL[3], lw=1.5)
    axes[1].set_xlabel('Actual', color=_GRAY)
    axes[1].set_ylabel('Predicted', color=_GRAY)
    axes[1].set_title('Actual vs Predicted', fontsize=10, fontweight='bold', color='#1e293b')
    axes[2].hist(residuals, bins=30, color=_PAL[2], edgecolor='white', alpha=0.85)
    axes[2].set_xlabel('Residual', color=_GRAY)
    axes[2].set_ylabel('Count', color=_GRAY)
    axes[2].set_title('Residual Distribution', fontsize=10, fontweight='bold', color='#1e293b')
    plt.tight_layout()
    return fig


def _plot_feature_importance(model, feature_names, model_name, top_n=20):
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if importance is None:
        return None
    idx = np.argsort(importance)[-top_n:]
    fig, ax = _fig(8, max(4, len(idx) * 0.35), f'Feature Importance — {model_name}')
    colors = [_PAL[0] if v > np.median(importance[idx]) else _PAL[2] for v in importance[idx]]
    ax.barh(np.array(feature_names)[idx], importance[idx], color=colors, edgecolor='white', height=0.7)
    ax.set_xlabel('Importance', color=_GRAY)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


# ── LEARNING CURVE CORRIGÉE ──────────────────────────────────────────────────
# Pourquoi c'est lent :
#   1. learning_curve entraîne le modèle cv × len(train_sizes) fois
#      (ex: 3 folds × 8 tailles = 24 fits)
#   2. n_jobs=-1 avec joblib/loky dans Jupyter crée un overhead de
#      sérialisation (pickling) des objets Python qui peut être PLUS lent
#      que la parallélisation ne gagne — surtout pour AdaBoost qui est
#      séquentiel en interne (chaque estimator dépend du précédent).
#   3. Gros datasets : chaque fit copie et sérialise X_train entier.
#
# Corrections appliquées :
#   - Sous-échantillonnage à MAX_LC_SAMPLES lignes (évite les gros fits)
#   - n_jobs=1 (élimine l'overhead joblib — plus rapide sur <10k lignes)
#   - train_sizes réduit à 6 points au lieu de 8
#   - cv=3 maintenu (bon compromis biais/variance)

_MAX_LC_SAMPLES = 5_000  # limite pour garder la LC interactive


def _plot_learning_curve(model, X_train, y_train, model_name, task,
                         max_samples=_MAX_LC_SAMPLES, cv=3, n_points=6):
    from sklearn.model_selection import learning_curve

    scoring = 'accuracy' if task == 'classification' else 'r2'

    # Sous-échantillonnage
    X_lc, y_lc = X_train, y_train
    n = len(X_lc)
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        if hasattr(X_lc, 'iloc'):
            X_lc = X_lc.iloc[idx]
            y_lc = y_lc.iloc[idx] if hasattr(y_lc, 'iloc') else y_lc[idx]
        else:
            X_lc = X_lc[idx]
            y_lc = y_lc[idx]

    try:
        train_sz, train_sc, val_sc = learning_curve(
            model, X_lc, y_lc,
            cv=cv,
            n_jobs=1,
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, n_points)
        )
    except Exception:
        return None

    sample_note = f" (sous-échantillon {len(X_lc):,} lignes)" if n > max_samples else f" ({n:,} lignes)"
    fig, ax = _fig(8, 4.5, f'Learning Curve — {model_name}{sample_note}')

    # Zones d'incertitude (±1 std entre folds)
    ax.fill_between(train_sz,
                    train_sc.mean(1) - train_sc.std(1),
                    train_sc.mean(1) + train_sc.std(1),
                    alpha=0.15, color=_PAL[0], label='_nolegend_')
    ax.fill_between(train_sz,
                    val_sc.mean(1) - val_sc.std(1),
                    val_sc.mean(1) + val_sc.std(1),
                    alpha=0.15, color=_PAL[1], label='_nolegend_')

    ax.plot(train_sz, train_sc.mean(1), 'o-', color=_PAL[0], lw=2,
            label=f'Train (final: {train_sc.mean(1)[-1]:.3f})')
    ax.plot(train_sz, val_sc.mean(1), 's-', color=_PAL[1], lw=2,
            label=f'Validation CV (final: {val_sc.mean(1)[-1]:.3f})')

    # Annotation de l'écart final (gap = diagnostic overfitting)
    gap = train_sc.mean(1)[-1] - val_sc.mean(1)[-1]
    gap_color = '#ef4444' if gap > 0.1 else '#f59e0b' if gap > 0.03 else '#10b981'
    ax.annotate(
        f'Écart final : {gap:+.3f}',
        xy=(train_sz[-1], (train_sc.mean(1)[-1] + val_sc.mean(1)[-1]) / 2),
        xytext=(-80, 0), textcoords='offset points',
        fontsize=8.5, color=gap_color, fontweight='bold',
        arrowprops=dict(arrowstyle='-', color=gap_color, lw=1.2)
    )

    ax.set_xlabel('Nombre d\'exemples d\'entraînement', color=_GRAY)
    ax.set_ylabel(scoring.upper(), color=_GRAY)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    plt.tight_layout()
    return fig


def _plot_shap(model, X_train, X_eval, model_name):
    try:
        import shap
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_eval[:min(100, len(X_eval))])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(_BG)
        plt.sca(axes[0])
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        axes[0].set_title(f'SHAP Beeswarm — {model_name}', fontsize=10, fontweight='bold', color='#1e293b')
        plt.sca(axes[1])
        shap.plots.bar(shap_values, max_display=15, show=False)
        axes[1].set_title(f'SHAP Feature Importance — {model_name}', fontsize=10, fontweight='bold', color='#1e293b')
        plt.tight_layout()
        return fig
    except ImportError:
        return None
    except Exception:
        return None


def _plot_lime_tabular(model, X_train, X_eval, y_eval, feature_names, model_name, task):
    try:
        import lime
        import lime.lime_tabular
        mode = 'classification' if task == 'classification' else 'regression'
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values if hasattr(X_train, 'values') else X_train,
            feature_names=feature_names,
            mode=mode
        )
        predict_fn = model.predict_proba if hasattr(model, 'predict_proba') and task == 'classification' else model.predict
        exp = explainer.explain_instance(
            X_eval.iloc[0].values if hasattr(X_eval, 'iloc') else X_eval[0],
            predict_fn,
            num_features=15
        )
        fig = exp.as_pyplot_figure()
        fig.patch.set_facecolor(_BG)
        fig.suptitle(f'LIME Explanation (sample 0) — {model_name}', fontsize=10, fontweight='bold', color='#1e293b')
        plt.tight_layout()
        return fig
    except ImportError:
        return None
    except Exception:
        return None


def _plot_pred_distribution(model, X_eval, y_eval, model_name, task):
    y_pred = model.predict(X_eval)
    y_arr = np.array(y_eval)

    if task == 'classification':
        classes = np.unique(np.concatenate([y_arr, y_pred]))
        fig, axes = _multi_fig(1, 2, w=12, h=4)
        x = np.arange(len(classes))
        axes[0].bar(x - 0.2, [np.sum(y_arr == c) for c in classes], 0.4,
                    label='y_true', color=_PAL[0], edgecolor='white')
        axes[0].bar(x + 0.2, [np.sum(y_pred == c) for c in classes], 0.4,
                    label='y_pred', color=_PAL[1], edgecolor='white')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes)
        axes[0].set_title('Class distribution: y_true vs y_pred', fontsize=10, fontweight='bold', color='#1e293b')
        axes[0].set_ylabel('Count', color=_GRAY)
        axes[0].legend(fontsize=9)
        err_by_class = {
            c: np.mean(y_pred[y_arr == c] != c)
            for c in classes if np.sum(y_arr == c) > 0
        }
        axes[1].bar(
            [str(k) for k in err_by_class.keys()],
            list(err_by_class.values()),
            color=_PAL[3], edgecolor='white'
        )
        axes[1].set_title('Error rate by class', fontsize=10, fontweight='bold', color='#1e293b')
        axes[1].set_ylabel('Error rate', color=_GRAY)
        axes[1].set_ylim(0, 1)
        for i, (k, v) in enumerate(err_by_class.items()):
            axes[1].text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontsize=9, color='#1e293b')
    else:
        errors = y_arr - y_pred
        fig, axes = _multi_fig(1, 3, w=14, h=4)
        axes[0].hist(y_arr, bins=30, color=_PAL[1], alpha=0.7, label='y_true', edgecolor='white')
        axes[0].hist(y_pred, bins=30, color=_PAL[0], alpha=0.7, label='y_pred', edgecolor='white')
        axes[0].set_title('Distribution y_true vs y_pred', fontsize=10, fontweight='bold', color='#1e293b')
        axes[0].set_ylabel('Count', color=_GRAY)
        axes[0].legend(fontsize=9)
        axes[1].hist(errors, bins=30, color=_PAL[2], edgecolor='white', alpha=0.85)
        axes[1].axvline(0, color=_PAL[3], lw=1.5, ls='--')
        axes[1].set_title('Error distribution (y_true - y_pred)', fontsize=10, fontweight='bold', color='#1e293b')
        axes[1].set_xlabel('Error', color=_GRAY)
        axes[1].set_ylabel('Count', color=_GRAY)
        axes[2].scatter(y_arr, np.abs(errors), alpha=0.4, color=_PAL[4], s=15)
        axes[2].axhline(np.abs(errors).mean(), color=_PAL[3], lw=1.5, ls='--',
                        label=f'Mean abs error: {np.abs(errors).mean():.3f}')
        axes[2].set_xlabel('y_true', color=_GRAY)
        axes[2].set_ylabel('|error|', color=_GRAY)
        axes[2].set_title('Absolute error vs y_true', fontsize=10, fontweight='bold', color='#1e293b')
        axes[2].legend(fontsize=9)

    plt.tight_layout()
    return fig, y_pred


def _plot_metric_comparison(all_metrics, metric_name):
    names = list(all_metrics.keys())
    raw_vals = [all_metrics[n].get(metric_name) for n in names]
    pairs = [(n, v) for n, v in zip(names, raw_vals) if isinstance(v, (int, float))]
    if not pairs:
        return None
    names, vals = zip(*pairs)
    fig, ax = _fig(7, 3.5, f'Model Comparison — {metric_name}')
    bars = ax.bar(names, vals, color=_PAL[:len(vals)], edgecolor='white', width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, color='#1e293b')
    ax.set_ylabel(metric_name, color=_GRAY)
    ax.set_ylim(0, min(1.1, max(vals) * 1.15))
    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.tight_layout()
    return fig


class EvaluationUI:
    def __init__(self, state):
        self.state       = state
        self.config      = getattr(state, 'config', {})
        self.splits      = getattr(state, 'data_splits', {})
        self.models      = getattr(state, 'models', {})
        self.predictions = getattr(state, 'predictions', {})
        self.cfg_metrics = self.config.get('metrics', {})
        self.cfg_exp     = self.config.get('explainability', {})
        self.inference_mode = _is_inference_mode(self.splits)

        if not self.models:
            self.ui = styles.error_msg("No trained model found. Run the Modeling step first.")
            return

        if self.inference_mode:
            any_val = any(
                _is_valid(p.get('X_val')) and _is_valid(p.get('y_val'))
                for p in self.predictions.values()
            )
            if not any_val:
                self.ui = styles.error_msg(
                    "Inference mode: no internal validation data found in state.predictions. "
                    "Check that the Modeling step performed an internal split."
                )
                return
        else:
            if not _is_valid(self.splits.get('X_test')) or not _is_valid(self.splits.get('y_test')):
                self.ui = styles.error_msg(
                    "Evaluation mode: X_test or y_test missing. "
                    "Check the Splitting step."
                )
                return

        self._build_ui()

    def _build_ui(self):
        header = widgets.HTML(styles.card_html("Evaluation", "Evaluation & Interpretability", ""))
        top_bar = widgets.HBox(
            [header],
            layout=widgets.Layout(
                align_items='center', margin='0 0 12px 0',
                padding='0 0 10px 0', border_bottom='2px solid #ede9fe'
            )
        )

        model_names = list(self.models.keys())
        self.dd_model = widgets.Dropdown(
            options=['-- All --'] + model_names,
            description='Model:',
            layout=widgets.Layout(width='280px')
        )

        self.task, self.subtask = self._detect_task()
        task_badge = widgets.HTML(
            f"<span style='background:#ede9fe; color:#6d28d9; border-radius:4px; padding:4px 10px;"
            f"font-size:0.8em; font-weight:700;'>Task: {self.task} / {self.subtask}</span>"
        )

        if self.inference_mode:
            mode_banner = styles.help_box(
                "<b>Inference mode:</b> y_test absent from split. "
                "Evaluation is performed on the internal validation partition. "
                "Metrics reflect performance on this portion of the train dataset.",
                "#f59e0b"
            )
        else:
            mode_banner = styles.help_box(
                "<b>Evaluation mode:</b> Evaluation on the provided test dataset.",
                "#10b981"
            )

        self.tab = widgets.Tab()

        self.out_metrics = widgets.Output()
        tab0 = widgets.VBox([
            styles.help_box("<b>Metrics:</b> All configured metrics computed on the evaluation set.", '#6366f1'),
            widgets.HTML("<div style='height:6px'></div>"),
            self.out_metrics
        ])

        self.out_clf_plots = widgets.Output()
        self.btn_clf = widgets.Button(
            description='Generate plots', button_style='primary',
            layout=widgets.Layout(width='200px', margin='8px 0')
        )
        self.btn_clf.on_click(self._on_clf_plots)
        tab1 = widgets.VBox([
            styles.help_box("<b>Classification:</b> Confusion matrix, ROC curves, Precision-Recall, calibration.", '#10b981'),
            self.btn_clf, self.out_clf_plots
        ])

        self.out_reg_plots = widgets.Output()
        self.btn_reg = widgets.Button(
            description='Generate plots', button_style='primary',
            layout=widgets.Layout(width='200px', margin='8px 0')
        )
        self.btn_reg.on_click(self._on_reg_plots)
        tab2 = widgets.VBox([
            styles.help_box("<b>Regression:</b> Residuals vs Predicted, Actual vs Predicted, error distribution.", '#f59e0b'),
            self.btn_reg, self.out_reg_plots
        ])

        self.out_fi = widgets.Output()
        self.btn_fi = widgets.Button(
            description='Feature Importance', button_style='primary',
            layout=widgets.Layout(width='200px', margin='8px 4px 8px 0')
        )
        self.btn_fi.on_click(self._on_fi_plots)
        self.btn_lc = widgets.Button(
            description='▶ Générer Learning Curve', button_style='warning',
            layout=widgets.Layout(width='230px', margin='0')
        )
        self.btn_lc.on_click(self._on_lc_plots)

        # ── Contrôles Learning Curve ──────────────────────────────────────────
        n_total_lc = len(self.splits.get('X_train')) if _is_valid(self.splits.get('X_train')) else _MAX_LC_SAMPLES
        lc_sample_default = min(n_total_lc, _MAX_LC_SAMPLES)

        self.lc_samples = widgets.BoundedIntText(
            value=lc_sample_default,
            min=200, max=n_total_lc, step=500,
            description='Max lignes :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='220px'),
            tooltip='Nombre max de lignes pour le sous-échantillonnage. Plus = plus lent.'
        )
        self.lc_cv = widgets.BoundedIntText(
            value=3, min=2, max=10, step=1,
            description='CV folds :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='160px'),
            tooltip='Nombre de folds de cross-validation. Plus = plus stable mais plus lent.'
        )
        self.lc_points = widgets.BoundedIntText(
            value=6, min=3, max=12, step=1,
            description='Nb points :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='160px'),
            tooltip='Nombre de tailles d\'entraînement testées (de 10% à 100%).'
        )
        self.lc_n_est = widgets.BoundedIntText(
            value=30, min=5, max=200, step=5,
            description='n_estimators :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='190px'),
            tooltip='Pour les modèles d\'ensemble (RF, XGB…) : réduit le nombre d\'arbres pour accélérer.'
        )

        lc_controls = widgets.VBox([
            widgets.HTML(
                "<div style='font-size:0.82em;font-weight:700;color:#374151;"
                "margin:10px 0 6px 0;letter-spacing:0.03em;'>⚙ Paramètres Learning Curve</div>"
            ),
            widgets.HBox(
                [self.lc_samples, self.lc_cv, self.lc_points, self.lc_n_est],
                layout=widgets.Layout(gap='12px', align_items='center', flex_wrap='wrap')
            ),
            widgets.HTML(
                "<div style='font-size:0.75em;color:#94a3b8;margin-top:2px;'>"
                f"Dataset : {n_total_lc:,} lignes — temps estimé ≈ cv × points fits "
                f"({'sous-échantillonné' if n_total_lc > _MAX_LC_SAMPLES else 'dataset complet'})"
                "</div>"
            ),
        ], layout=widgets.Layout(
            background='#f8fafc', border='1px solid #e2e8f0',
            border_radius='6px', padding='10px 14px', margin='0 0 8px 0'
        ))

        # ── Guide de lecture ──────────────────────────────────────────────────
        lc_guide = widgets.HTML("""
        <details style='margin:6px 0 10px 0; font-size:0.82em; color:#374151;'>
          <summary style='cursor:pointer; font-weight:700; color:#6366f1;
                          padding:6px 10px; background:#ede9fe; border-radius:6px;
                          list-style:none; user-select:none;'>
            📖 Comment lire ce graphe ?
          </summary>
          <div style='padding:12px 14px; border:1px solid #e0e7ff;
                      border-radius:0 0 6px 6px; background:#ffffff; line-height:1.7;'>

            <b style='color:#1e293b;'>Axe X</b> — nombre d'exemples d'entraînement utilisés (de ~10% à 100% du sous-échantillon).<br>
            <b style='color:#1e293b;'>Axe Y</b> — score (accuracy ou R²) : plus c'est haut, mieux c'est.<br><br>

            <div style='display:flex; gap:16px; flex-wrap:wrap; margin-bottom:10px;'>
              <div style='display:flex; align-items:center; gap:6px;'>
                <div style='width:28px; height:3px; background:#6366f1; border-radius:2px;'></div>
                <span><b>Train</b> — évalué sur les données vues à l'entraînement</span>
              </div>
              <div style='display:flex; align-items:center; gap:6px;'>
                <div style='width:28px; height:3px; background:#10b981; border-radius:2px;'></div>
                <span><b>Validation (CV)</b> — évalué sur des données <i>jamais vues</i></span>
              </div>
            </div>

            <table style='border-collapse:collapse; width:100%; font-size:0.92em;'>
              <tr style='background:#f1f5f9;'>
                <th style='padding:6px 10px; text-align:left; border-radius:4px 0 0 0;'>Diagnostic</th>
                <th style='padding:6px 10px; text-align:left;'>Signal visible</th>
                <th style='padding:6px 10px; text-align:left; border-radius:0 4px 0 0;'>Action recommandée</th>
              </tr>
              <tr style='border-bottom:1px solid #e2e8f0;'>
                <td style='padding:6px 10px; color:#10b981; font-weight:600;'>✅ Bon modèle</td>
                <td style='padding:6px 10px;'>Les deux courbes convergent vers un score élevé, écart faible</td>
                <td style='padding:6px 10px;'>Rien — déployer</td>
              </tr>
              <tr style='border-bottom:1px solid #e2e8f0; background:#fefce8;'>
                <td style='padding:6px 10px; color:#f59e0b; font-weight:600;'>⚠ Overfitting</td>
                <td style='padding:6px 10px;'>Train très haut, Validation bas — <b>grand écart persistant</b></td>
                <td style='padding:6px 10px;'>Régularisation ↑ · Moins de features · Plus de données</td>
              </tr>
              <tr style='border-bottom:1px solid #e2e8f0; background:#fff1f2;'>
                <td style='padding:6px 10px; color:#ef4444; font-weight:600;'>❌ Underfitting</td>
                <td style='padding:6px 10px;'>Les <b>deux courbes basses</b> et proches l'une de l'autre</td>
                <td style='padding:6px 10px;'>Modèle plus complexe · Plus de features · Régularisation ↓</td>
              </tr>
              <tr style='background:#f0fdf4;'>
                <td style='padding:6px 10px; color:#6366f1; font-weight:600;'>📈 Manque de données</td>
                <td style='padding:6px 10px;'>Validation <b>monte encore</b> à l'extrême droite du graphe</td>
                <td style='padding:6px 10px;'>Collecter plus de données — le modèle en bénéficiera</td>
              </tr>
            </table>

            <div style='margin-top:10px; padding:8px 10px; background:#eff6ff;
                        border-left:3px solid #3b82f6; border-radius:0 4px 4px 0; font-size:0.9em;'>
              💡 <b>Zone ombrée</b> = ± 1 écart-type entre les folds CV.
              Une zone large indique une <b>variance élevée</b> : le score dépend beaucoup
              du fold tiré, signe d'instabilité ou de dataset trop petit.
            </div>
          </div>
        </details>
        """)

        tab3 = widgets.VBox([
            styles.help_box(
                "<b>Feature Importance :</b> feature_importances_ ou coef_ selon le modèle.<br>"
                "<b>Learning Curve :</b> diagnostique overfitting / underfitting / manque de données. "
                "Réglez les paramètres ci-dessous selon la taille de votre dataset et le temps disponible.",
                '#3b82f6'
            ),
            widgets.HBox([self.btn_fi], layout=widgets.Layout(margin='4px 0 12px 0')),
            widgets.HTML("<hr style='border:1px solid #f1f5f9; margin:0 0 10px 0;'>"),
            lc_controls,
            lc_guide,
            widgets.HBox([self.btn_lc], layout=widgets.Layout(margin='0 0 8px 0')),
            self.out_fi
        ])

        self.out_exp = widgets.Output()
        exp_types = self._available_explainers()
        self.dd_explainer = widgets.Dropdown(
            options=exp_types,
            description='Explainer:',
            layout=widgets.Layout(width='280px')
        )
        self.btn_exp = widgets.Button(
            description="Run explanation", button_style='primary',
            layout=widgets.Layout(width='220px', margin='8px 0')
        )
        self.btn_exp.on_click(self._on_explainability)
        tab4 = widgets.VBox([
            styles.help_box("<b>Interpretability:</b> SHAP (TreeExplainer or KernelExplainer) and LIME tabular.", '#8b5cf6'),
            widgets.HBox([self.dd_explainer, self.btn_exp]),
            self.out_exp
        ])

        self.out_compare = widgets.Output()
        self.btn_compare = widgets.Button(
            description='Compare models', button_style='primary',
            layout=widgets.Layout(width='220px', margin='8px 0')
        )
        self.btn_compare.on_click(self._on_compare)
        tab5 = widgets.VBox([
            styles.help_box("<b>Comparison:</b> One chart per metric to identify the best model.", '#ec4899'),
            self.btn_compare, self.out_compare
        ])

        self.out_pred = widgets.Output()
        self.btn_pred = widgets.Button(
            description='Visualize predictions', button_style='primary',
            layout=widgets.Layout(width='220px', margin='8px 0')
        )
        self.btn_pred.on_click(self._on_pred_plots)
        tab6 = widgets.VBox([
            styles.help_box(
                "<b>Predictions:</b> Distribution of y_pred vs y_true, error analysis by class or value range, and CSV export.",
                '#3b82f6'
            ),
            self.btn_pred, self.out_pred
        ])

        all_tabs   = [tab0, tab1, tab2, tab3, tab4, tab5, tab6]
        all_titles = ['Metrics', 'Clf — Curves', 'Reg — Residuals', 'Feature Importance', 'Interpretability', 'Comparison', 'Predictions']

        if self.task == 'regression':
            all_tabs.pop(1)
            all_titles.pop(1)
        elif self.task == 'classification':
            all_tabs.pop(2)
            all_titles.pop(2)

        self.tab.children = all_tabs
        for i, t in enumerate(all_titles):
            self.tab.set_title(i, t)

        controls = widgets.HBox(
            [self.dd_model, task_badge],
            layout=widgets.Layout(align_items='center', gap='16px', margin='0 0 10px 0')
        )

        self.ui = widgets.VBox(
            [top_bar, mode_banner, controls, self.tab],
            layout=widgets.Layout(
                width='100%', max_width='1100px',
                border='1px solid #e5e7eb',
                padding='18px',
                border_radius='10px',
                background_color='#ffffff'
            )
        )

        self._compute_all_metrics()
        self._render_metrics()

    def _detect_task(self):
        y_train   = self.splits.get('y_train')
        prob_type = 'classification'
        subtask   = 'binary'
        if _is_valid(y_train):
            n_unique = y_train.nunique() if hasattr(y_train, 'nunique') else len(np.unique(y_train))
            if n_unique > 20:
                prob_type, subtask = 'regression', 'continuous'
            elif n_unique > 2:
                prob_type, subtask = 'classification', 'multiclass'
            else:
                prob_type, subtask = 'classification', 'binary'
        if hasattr(self.state, 'business_context') and self.state.business_context.get('domain'):
            dv = self.state.business_context['domain']
            for d in self.config.get('domains', []):
                if d['value'] == dv and d.get('task'):
                    prob_type = d['task']
                    subtask   = d.get('subtask') or subtask
                    break
        return prob_type, subtask

    def _selected_models(self):
        sel = self.dd_model.value
        if sel == '-- All --':
            return self.models
        return {sel: self.models[sel]}

    def _available_explainers(self):
        exp_list = [e['name'] for e in self.cfg_exp.get('tabular', [])]
        if not exp_list:
            exp_list = ['SHAP (Tree/Kernel)', 'LIME (Tabular)']
        return exp_list

    def _compute_all_metrics(self):
        self._all_metrics = {}
        for name, model in self.models.items():
            X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, name)
            if not _is_valid(X_eval) or not _is_valid(y_eval):
                self._all_metrics[name] = {'ERROR': 'evaluation data unavailable'}
                continue
            try:
                self._all_metrics[name] = _compute_metrics(
                    model, X_eval, y_eval, self.task, self.subtask, self.cfg_metrics
                )
            except Exception as e:
                self._all_metrics[name] = {'ERROR': str(e)}

    def _render_metrics(self):
        with self.out_metrics:
            clear_output(wait=True)
            for model_name, metrics in self._all_metrics.items():
                cards_html = ""
                valid = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                for i, (k, v) in enumerate(valid.items()):
                    cards_html += _metric_card(k, f'{v:.4f}', _PAL[i % len(_PAL)])
                display(HTML(
                    f"<div style='margin-bottom:16px;'>"
                    f"<div style='font-size:0.85em; font-weight:700; color:#6d28d9; margin-bottom:8px;"
                    f"letter-spacing:0.05em; text-transform:uppercase;'>{model_name}</div>"
                    f"<div style='display:flex; flex-wrap:wrap; gap:8px;'>{cards_html}</div>"
                    f"</div>"
                ))
            for model_name, metrics in self._all_metrics.items():
                errs = {k: v for k, v in metrics.items() if str(v).startswith('ERR')}
                if errs:
                    display(_warn(f"<b>{model_name}:</b> " + " | ".join(f"{k}: {v}" for k, v in errs.items())))

    def _on_clf_plots(self, b):
        with self.out_clf_plots:
            clear_output(wait=True)
            sel = self._selected_models()
            n = len(sel)
            if n == 1:
                name, model = list(sel.items())[0]
                X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, name)
                if not _is_valid(X_eval) or not _is_valid(y_eval):
                    display(_warn(f"Evaluation data unavailable for {name}."))
                    return
                display(_section('Confusion Matrix', '#6366f1'))
                try:
                    fig = _plot_confusion_matrix(model, X_eval, y_eval, name)
                    plt.show()
                except Exception as e:
                    display(_warn(f"Unavailable for {name}: {e}"))
            else:
                display(_section('Confusion Matrices', '#6366f1'))
                cols = min(3, n)
                rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
                fig.patch.set_facecolor(_BG)
                axes_flat = np.array(axes).flatten() if rows * cols > 1 else [axes]
                for ax, (name, model) in zip(axes_flat, sel.items()):
                    X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, name)
                    if not _is_valid(X_eval) or not _is_valid(y_eval):
                        ax.set_title(f'{name}: data unavailable', fontsize=9, color='#ef4444')
                        continue
                    try:
                        _plot_confusion_matrix(model, X_eval, y_eval, name, ax=ax)
                    except Exception:
                        ax.set_title(f'{name}: ERR', fontsize=9, color='#ef4444')
                for ax in axes_flat[len(sel):]:
                    ax.set_visible(False)
                plt.tight_layout()
                plt.show()

            first_name = list(sel.keys())[0]
            X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, first_name)

            if _is_valid(X_eval) and _is_valid(y_eval):
                display(_section('ROC Curves', '#10b981'))
                try:
                    plt.show(_plot_roc_curves(sel, X_eval, y_eval))
                except Exception as e:
                    display(_warn(f"ROC unavailable: {e}"))

                display(_section('Precision-Recall', '#f59e0b'))
                try:
                    plt.show(_plot_precision_recall(sel, X_eval, y_eval))
                except Exception as e:
                    display(_warn(f"Precision-Recall unavailable: {e}"))

                display(_section('Calibration', '#3b82f6'))
                try:
                    plt.show(_plot_calibration(sel, X_eval, y_eval))
                except Exception as e:
                    display(_warn(f"Calibration unavailable: {e}"))

    def _on_reg_plots(self, b):
        with self.out_reg_plots:
            clear_output(wait=True)
            sel = self._selected_models()
            for name, model in sel.items():
                X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, name)
                if not _is_valid(X_eval) or not _is_valid(y_eval):
                    display(_warn(f"Evaluation data unavailable for {name}."))
                    continue
                display(_section(f'Residuals — {name}', '#f59e0b'))
                try:
                    plt.show(_plot_residuals(model, X_eval, y_eval, name))
                except Exception as e:
                    display(_warn(f"Residuals unavailable for {name}: {e}"))

    def _on_fi_plots(self, b):
        with self.out_fi:
            clear_output(wait=True)
            X_train = self.splits.get('X_train')
            sel = self._selected_models()
            if not _is_valid(X_train):
                display(_warn("X_train missing."))
                return
            feature_names = (
                list(X_train.columns) if hasattr(X_train, 'columns')
                else [f'f{i}' for i in range(X_train.shape[1])]
            )
            for name, model in sel.items():
                display(_section(f'Feature Importance — {name}', '#3b82f6'))
                fig = _plot_feature_importance(model, feature_names, name)
                if fig:
                    plt.show()
                else:
                    display(_warn(f"{name} does not expose feature_importances_ or coef_."))

    def _on_lc_plots(self, b):
        with self.out_fi:
            clear_output(wait=True)
            X_train = self.splits.get('X_train')
            y_train = self.splits.get('y_train')
            sel = self._selected_models()

            if not _is_valid(X_train) or not _is_valid(y_train):
                display(_warn("X_train or y_train missing."))
                return

            # ── Lecture des paramètres depuis les widgets ─────────────────────
            max_samples = self.lc_samples.value
            cv_folds    = self.lc_cv.value
            n_points    = self.lc_points.value
            n_est_fast  = self.lc_n_est.value

            n_total = len(X_train)
            if n_total > max_samples:
                display(_info(
                    f"Dataset large ({n_total:,} lignes) — sous-échantillonnage à "
                    f"{max_samples:,} lignes (aléatoire, seed=42). "
                    f"Augmentez <b>Max lignes</b> pour plus de précision (plus lent)."
                ))

            for name, model in sel.items():
                display(_section(f'Learning Curve — {name}', '#6366f1'))

                # Clone rapide avec les paramètres choisis
                from sklearn.base import clone as sk_clone
                fast_model = sk_clone(model)
                if hasattr(fast_model, 'n_estimators'):
                    fast_model.set_params(n_estimators=min(n_est_fast, fast_model.n_estimators))
                if hasattr(fast_model, 'n_jobs'):
                    fast_model.set_params(n_jobs=1)

                n_est_actual = getattr(fast_model, 'n_estimators', None)
                notes = []
                if n_est_actual is not None:
                    notes.append(f"n_estimators={n_est_actual}")
                notes.append(f"cv={cv_folds}, {n_points} points")
                notes.append(f"max {max_samples:,} lignes")
                display(_info(f"Paramètres : {' · '.join(notes)}"))

                fig = _plot_learning_curve(
                    fast_model, X_train, y_train, name, self.task,
                    max_samples=max_samples, cv=cv_folds, n_points=n_points
                )
                if fig:
                    plt.show()
                else:
                    display(_warn(f"Learning curve unavailable for {name}."))

    def _on_pred_plots(self, b):
        with self.out_pred:
            clear_output(wait=True)
            sel = self._selected_models()
            for name, model in sel.items():
                X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, name)
                if not _is_valid(X_eval) or not _is_valid(y_eval):
                    display(_warn(f"Evaluation data unavailable for {name}."))
                    continue
                display(_section(f'Prediction distribution — {name}', '#3b82f6'))
                try:
                    fig, y_pred = _plot_pred_distribution(model, X_eval, y_eval, name, self.task)
                    plt.show()
                except Exception as e:
                    display(_warn(f"Plot failed for {name}: {e}"))
                    continue
                try:
                    y_arr = np.array(y_eval)
                    df_out = pd.DataFrame({'y_true': y_arr, 'y_pred': y_pred})
                    if hasattr(X_eval, 'index'):
                        df_out.index = X_eval.index
                    safe_name = name.replace(' ', '_').replace('/', '_')
                    csv_path = f"predictions_{safe_name}.csv"
                    df_out.to_csv(csv_path, index=True)
                    display(_info(
                        f"CSV exported: <b>{csv_path}</b> "
                        f"({len(df_out)} rows, columns: y_true, y_pred)"
                    ))
                except Exception as e:
                    display(_warn(f"CSV export failed: {e}"))

    def _on_explainability(self, b):
        with self.out_exp:
            clear_output(wait=True)
            X_train        = self.splits.get('X_train')
            sel            = self._selected_models()
            explainer_name = self.dd_explainer.value
            if not _is_valid(X_train):
                display(_warn("X_train invalid."))
                return
            feature_names = (
                list(X_train.columns) if hasattr(X_train, 'columns')
                else [f'f{i}' for i in range(X_train.shape[1])]
            )
            for model_name, model in sel.items():
                X_eval, y_eval = _resolve_eval_data(self.splits, self.predictions, model_name)
                if not _is_valid(X_eval):
                    display(_warn(f"Evaluation data unavailable for {model_name}."))
                    continue
                display(_section(f'{explainer_name} — {model_name}', '#8b5cf6'))
                if 'SHAP' in explainer_name:
                    display(_info('Computing SHAP values...'))
                    fig = _plot_shap(model, X_train, X_eval, model_name)
                    if fig:
                        clear_output(wait=True)
                        display(_section(f'SHAP — {model_name}', '#8b5cf6'))
                        plt.show()
                    else:
                        display(_warn('SHAP not available — pip install shap'))
                elif 'LIME' in explainer_name:
                    display(_info('Computing LIME explanation...'))
                    fig = _plot_lime_tabular(model, X_train, X_eval, y_eval, feature_names, model_name, self.task)
                    if fig:
                        clear_output(wait=True)
                        display(_section(f'LIME — {model_name}', '#8b5cf6'))
                        plt.show()
                    else:
                        display(_warn('LIME not available — pip install lime'))
                else:
                    for e in self.cfg_exp.get('tabular', []):
                        if e['name'] == explainer_name:
                            display(HTML(
                                f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:12px;'>"
                                f"<pre style='margin:0; font-size:0.82em; color:#1e293b;'>{e['code']}</pre>"
                                f"</div>"
                            ))
                            break

    def _on_compare(self, b):
        with self.out_compare:
            clear_output(wait=True)
            if len(self._all_metrics) < 2:
                display(_warn('Comparison requires at least 2 trained models.'))
            all_metric_names = set()
            for m in self._all_metrics.values():
                all_metric_names.update([k for k, v in m.items() if isinstance(v, (int, float))])
            for metric in sorted(all_metric_names):
                display(_section(f'Comparison — {metric}', '#ec4899'))
                fig = _plot_metric_comparison(self._all_metrics, metric)
                if fig:
                    plt.show()
            display(_section('Summary table', '#1e293b'))
            rows = []
            for name, metrics in self._all_metrics.items():
                row = {'Model': name}
                row.update({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in metrics.items()})
                rows.append(row)
            if rows:
                df_summary = pd.DataFrame(rows).set_index('Model')
                num_cols = [c for c in df_summary.columns if df_summary[c].dtype in [float, np.float64]]
                styled = df_summary.style.set_properties(**{'font-size': '0.85em'})
                if num_cols:
                    styled = styled.background_gradient(cmap='Blues', axis=0, subset=num_cols)
                display(styled)
            self.state.log_step(
                "Evaluation", "Metrics Computed",
                {"models": list(self._all_metrics.keys()), "task": self.task}
            )


def runner(state):
    ev = EvaluationUI(state)
    display(ev.ui)
    return ev

try:
    runner(state)
except NameError:
    pass