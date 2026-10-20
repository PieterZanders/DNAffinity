"""Plotting helpers, kept separate from the training logic so the package can
run headless and the Model class stays matplotlib-free.

Each function optionally saves the figure to disk; when a save path is given
the figure is not shown interactively.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions(y_test, y_pred, r2=None, y_train=None, y_train_pred=None,
                     r2_train=None, save_path=None):
    """
    Scatter plot of predicted vs ground-truth values with the identity line.
    If training data is provided it is shown underneath the test data, so
    train/test performance can be compared at a glance.

    :param y_test: ground-truth test values
    :param y_pred: predicted test values
    :param r2: optional test r2 to show in the legend
    :param y_train: optional ground-truth training values
    :param y_train_pred: optional predicted training values
    :param r2_train: optional training r2 to show in the legend
    :param save_path: optional path to save the figure as an image
    :return: None
    """
    fig = plt.figure(figsize=(5, 5))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.xlabel('Ground truth')
    plt.ylabel('Predicted')
    if y_train is not None and y_train_pred is not None:
        train_label = f'train, r$^2$ = {r2_train}' if r2_train is not None else 'train'
        plt.scatter(y_train, y_train_pred, label=train_label, s=8, alpha=0.25,
                    color='darkorange')
    test_label = f'test, r$^2$ = {r2}' if r2 is not None else 'test'
    plt.scatter(y_test, y_pred, label=test_label, s=8, alpha=0.5, color='tab:blue')
    plt.plot([0, 1], color='red')
    plt.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_feature_importances(features, importances_sorted, save_dir=None):
    """
    Pie chart of the relative importance of presence / electrostatic / shape
    features, plus a bar chart of the top individual features.

    :param features: list of [feature_type, importance] pairs (Model.features)
    :param importances_sorted: importances sorted in decreasing order (Model.l)
    :param save_dir: optional directory to save 'feature_importances.png' (pie)
                     and 'top_features.png' (bar chart)
    :return: pd.DataFrame with the per-feature importances, or None if the
             model does not provide importances
    """
    if not features or len(features) == 0:
        return None

    df = pd.DataFrame(features, columns=['Feature', 'Importance'])

    index = [k for k in range(len(df)) if df['Feature'].iloc[k] == 'Presence']
    p = sum([df['Importance'].iloc[k] for k in index]) if index else 0

    index = [k for k in range(len(df)) if df['Feature'].iloc[k] == 'Electro']
    e = sum([df['Importance'].iloc[k] for k in index]) if index else 0

    shape = 1 - e - p
    y = np.array([p, e, shape])

    # pie chart: importance grouped by feature class
    fig = plt.figure(figsize=(8, 8))
    labels = ["Presence", "Electrostatic", "Shape"]
    patches, texts = plt.pie(y, startangle=0)
    plt.legend(patches, labels, loc="best")
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'feature_importances.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    # bar chart: top individual features
    if importances_sorted is not None and len(importances_sorted) > 0:
        fig = plt.figure(figsize=(8, 5))
        plt.xlabel('Feature rank')
        plt.ylabel('Relative importance')
        plt.bar(range(len(importances_sorted[0:10])), importances_sorted[0:10],
                color='red', align="center")
        if save_dir is not None:
            fig.savefig(Path(save_dir) / 'top_features.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    return df


def plot_training_curve(history, save_path=None):
    """
    Per-epoch training loss (and validation loss on the test split, if recorded).

    :param history: dict {'train': [...], 'val': [...] or None} (Model.history)
    :param save_path: optional path to save the figure as an image
    :return: None (does nothing for models without a loss history)
    """
    if not history or not history.get('train'):
        return
    epochs = np.arange(1, len(history['train']) + 1)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(epochs, history['train'], label='train')
    if history.get('val'):
        plt.plot(epochs, history['val'], label='validation (test split)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def save_plots(model, out_dir):
    """
    Save all available plots for a trained Model into out_dir:
    'predictions.png' (train + test scatter) always, 'training_curve.png'
    for models trained by epochs (PyTorch), and 'feature_importances.png' /
    'top_features.png' when the regressor provides feature importances
    (e.g. random forest).

    :param model: trained dnaffinity Model with predict() already called
    :param out_dir: directory to write the images to
    :return: pd.DataFrame with per-feature importances, or None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_predictions(model.y_test, model.y_pred, model.r2,
                     y_train=model.y_train, y_train_pred=model.y_train_pred,
                     r2_train=model.r2_train,
                     save_path=out_dir / 'predictions.png')
    plot_training_curve(model.history, save_path=out_dir / 'training_curve.png')
    return plot_feature_importances(model.features, model.l, save_dir=out_dir)
