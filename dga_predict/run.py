"""
Run experiments and reproduce DGA detection results (ManualRF vs LSTM)
Simplified version: removes Bigram model entirely.
"""

import gc
import os
import pickle
import argparse
import random
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

import dga_classifier.manual_rf as manualrf
import dga_classifier.lstm as lstm

RESULT_FILE = 'results.pkl'
METRICS_CSV = 'metrics.csv'


def set_seed(seed):
    """Ensure reproducibility across runs"""
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)


def run_experiments(is_manualrf=True, islstm=True, nfolds=10, force=False, seed=42):
    """Runs selected experiments and caches results"""
    if not force and os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'rb') as f:
            print(f"Loading existing results from {RESULT_FILE}")
            return pickle.load(f)

    set_seed(seed)
    results = {'manualrf': None, 'lstm': None}

    if is_manualrf:
        print("\n=== Running Manual Statistical Features + RandomForest ===")
        results['manualrf'] = manualrf.run(nfolds=nfolds)

    gc.collect()

    if islstm:
        print("\n=== Running LSTM Deep Model ===")
        results['lstm'] = lstm.run(nfolds=nfolds)

    with open(RESULT_FILE, 'wb') as f:
        pickle.dump(results, f)
        print(f"\n✓ Saved all results to {RESULT_FILE}")

    return results


def summarize_results_arr(results):
    """Aggregate multiple folds into unified ROC metrics"""
    all_y, all_probs = [], []
    for r in results:
        all_y.extend(r['y'])
        all_probs.extend(np.ravel(r['probs']).tolist())

    preds = [1 if p > 0.5 else 0 for p in all_probs]

    acc = accuracy_score(all_y, preds)
    pre = precision_score(all_y, preds)
    rec = recall_score(all_y, preds)
    f1 = f1_score(all_y, preds)
    cm = confusion_matrix(all_y, preds)
    fpr, tpr, _ = roc_curve(all_y, all_probs)
    auc_score = auc(fpr, tpr)

    metrics = {
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': cm.tolist()
    }
    return fpr, tpr, metrics


def save_metrics_csv(metrics_dict, out=METRICS_CSV):
    """Save metrics for all models to CSV"""
    rows = []
    for model_name, metrics in metrics_dict.items():
        if metrics is None:
            continue
        rows.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'confusion_matrix': metrics['confusion_matrix']
        })
    with open(out, 'w', newline='') as csvfile:
        fieldnames = ['model','accuracy','precision','recall','f1','auc','confusion_matrix']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"✓ Metrics saved to {out}")


def create_figs(force=False, nfolds=10, models=('manualrf','lstm'), seed=42):
    """Generate ROC curve comparison for selected models"""
    is_manualrf = 'manualrf' in models
    islstm = 'lstm' in models

    results = run_experiments(is_manualrf=is_manualrf, islstm=islstm, nfolds=nfolds, force=force, seed=seed)

    plt.figure(figsize=(10, 8))
    plt.style.use('bmh')

    metrics_out = {}

    if results.get('manualrf'):
        fpr_m, tpr_m, metrics_m = summarize_results_arr(results['manualrf'])
        metrics_out['manualrf'] = metrics_m
        plt.plot(fpr_m, tpr_m, label=f"ManualRF (AUC={metrics_m['auc']:.4f})", color='darkorange')

    if results.get('lstm'):
        fpr_l, tpr_l, metrics_l = summarize_results_arr(results['lstm'])
        metrics_out['lstm'] = metrics_l
        plt.plot(fpr_l, tpr_l, label=f"LSTM (AUC={metrics_l['auc']:.4f})", color='steelblue')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve - ManualRF vs LSTM', fontsize=18)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.savefig('roc.png', dpi=300)
    print("✓ ROC curve saved to roc.png")

    save_metrics_csv(metrics_out, METRICS_CSV)
    return metrics_out


def main():
    parser = argparse.ArgumentParser(description='Run DGA experiments (ManualRF vs LSTM)')
    parser.add_argument('--nfolds', type=int, default=10, help='Number of folds (paper uses 10)')
    parser.add_argument('--force', action='store_true', help='Force re-generate results (overwrite results.pkl)')
    parser.add_argument('--models', type=str, default='manualrf,lstm', help='Comma-separated models to run (manualrf,lstm)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--fast', action='store_true', help='Use nfolds=1 for quick test (overrides nfolds)')
    args = parser.parse_args()

    nfolds = 1 if args.fast else args.nfolds
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    print(f"Running with models={models}, nfolds={nfolds}, seed={args.seed}, force={args.force}")
    create_figs(force=args.force, nfolds=nfolds, models=models, seed=args.seed)


if __name__ == "__main__":
    main()
