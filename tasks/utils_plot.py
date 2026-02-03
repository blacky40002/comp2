"""Funzioni di plotting condivise per tutti i task."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")


def _save(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Grafico salvato: {path}")


def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 11})


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel("Reale")
    ax.set_xlabel("Predetto")
    ax.set_title(title)
    _save(fig, filename)


def plot_feature_importance(features, values, title, filename, top_n=20):
    """Feature importance per una singola classe."""
    indices = np.argsort(np.abs(values))[-top_n:]
    names = np.array(features)[indices]
    vals = np.array(values)[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Peso (Coefficiente SVM)")
    ax.set_title(title)
    _save(fig, filename)


def plot_feature_importance_multi(features, coef_matrix, class_names, title, filename, top_n=20):
    """Feature importance con ordine condiviso tra tutte le classi.

    Seleziona le top-N features per max |coef| tra tutte le classi,
    poi plotta un subplot per classe con lo stesso ordine.
    """
    features = np.array(features)
    max_abs = np.max(np.abs(coef_matrix), axis=0)
    indices = np.argsort(max_abs)[-top_n:]
    names = features[indices]

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 8), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for ax, cls_name, cls_idx in zip(axes, class_names, range(n_classes)):
        vals = coef_matrix[cls_idx][indices]
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        if cls_idx == 0:
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Peso")
        ax.set_title(cls_name)

    fig.suptitle(title, fontsize=14)
    _save(fig, filename)


def plot_model_comparison(results_list, metric_key, title, filename):
    df = pd.DataFrame(results_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="name", y=metric_key, data=df, palette="viridis",
                hue="name", legend=False, order=df["name"], ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_ylim(0, 1.0)
    _save(fig, filename)


def plot_training_curves(train_losses, val_losses, val_accs, val_f1s, filename):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs, train_losses, "b-o", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_accs, "g-o", linewidth=2, label="Accuracy")
    ax2.plot(epochs, val_f1s, "m-s", linewidth=2, label="F1-Macro")
    for i, (acc, f1) in enumerate(zip(val_accs, val_f1s)):
        ax2.annotate(f"{acc:.2f}", (epochs[i], acc), textcoords="offset points",
                     xytext=(0, 10), ha="center", color="green")
        ax2.annotate(f"{f1:.2f}", (epochs[i], f1), textcoords="offset points",
                     xytext=(0, -15), ha="center", color="purple")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Score (0-1)")
    ax2.set_title("Validation Performance")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right")
    ax2.grid(True)

    _save(fig, filename)


def plot_learning_curve(train_sizes, train_scores, val_scores, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_scores, "b-o", label="Train F1")
    ax.plot(train_sizes, val_scores, "r-o", label="Val F1")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("F1-Macro")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True)
    _save(fig, filename)
