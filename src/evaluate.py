"""
Evaluate trained models for credential theft detection on the test set.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# Import our feature engineering function
from src.features import engineer_features
from src.datasets import load_project_dataset

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_and_engineer_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw synthetic login data and engineer features.

    Returns:
        X: DataFrame with engineered features
        y: Series with target (is_attack)
    """
    df_raw = load_project_dataset()
    X, y = engineer_features(df_raw)
    return X, y


def time_based_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, test sets based on time (timestamp order).
    Assumes X and y are already sorted by timestamp.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    n = len(X)
    if n < 3:
        raise ValueError("At least three events are required for a train/validation/test split")
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)  # 0.7 + 0.15

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_models() -> dict:
    """
    Load the trained models and threshold information.

    Returns:
        Dictionary containing models and thresholds
    """
    models = {}
    # Load Logistic Regression pipeline
    models['logistic_regression'] = joblib.load("models/logistic_regression.joblib")
    # Load Random Forest
    models['random_forest'] = joblib.load("models/random_forest.joblib")
    # Load Isolation Forest
    models['isolation_forest'] = joblib.load("models/isolation_forest.joblib")

    # Load thresholds
    with open("models/threshold.json", "r") as f:
        threshold_info = json.load(f)

    return models, threshold_info


def get_model_scores(models: dict, X: pd.DataFrame) -> dict:
    """
    Get prediction scores from each model for the given data.

    Args:
        models: Dictionary of trained models
        X: Feature DataFrame

    Returns:
        Dictionary of scores for each model
    """
    scores = {}
    # Logistic Regression: probability of positive class
    scores['logistic_regression'] = models['logistic_regression'].predict_proba(X)[:, 1]
    # Random Forest: probability of positive class
    scores['random_forest'] = models['random_forest'].predict_proba(X)[:, 1]
    # Isolation Forest: convert decision_function to anomaly score (higher = more anomalous)
    # Decision function: higher means more like inliers (normal). We'll negate.
    scores['isolation_forest'] = -models['isolation_forest'].decision_function(X)
    return scores


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict:
    """
    Compute precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError:
        pr_auc = 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': [tn, fp, fn, tp]
    }


def plot_confusion_matrices(y_test: np.ndarray, scores_dict: dict, thresholds: dict, save_dir: str):
    """
    Plot confusion matrices for each model and save to file.

    Args:
        y_test: True test labels
        scores_dict: Dictionary of model scores
        thresholds: Dictionary of thresholds
        save_dir: Directory to save plots
    """
    n_models = len(scores_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, scores) in enumerate(scores_dict.items()):
        y_pred = (scores >= thresholds[model_name]).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_test: np.ndarray, scores_dict: dict, save_dir: str):
    """
    Plot ROC curves for each model and save to file.

    Args:
        y_test: True test labels
        scores_dict: Dictionary of model scores
        save_dir: Directory to save plots
    """
    plt.figure()
    for model_name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(y_test: np.ndarray, scores_dict: dict, save_dir: str):
    """
    Plot precision-recall curves for each model and save to file.

    Args:
        y_test: True test labels
        scores_dict: Dictionary of model scores
        save_dir: Directory to save plots
    """
    plt.figure()
    for model_name, scores in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        plt.plot(recall, precision, label=f'{model_name.replace("_", " ").title()} (AUC = {pr_auc:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_rf_feature_importance(model, feature_names: list, save_dir: str, top_n: int = 15):
    """
    Plot top-n feature importances for Random Forest model.

    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names
        save_dir: Directory to save plot
        top_n: Number of top features to display
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure()
    plt.barh(range(top_n), top_importances[::-1], align='center')
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Random Forest Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rf_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(y: np.ndarray, save_dir: str):
    """
    Plot bar chart of class distribution.

    Args:
        y: Array of labels
        save_dir: Directory to save plot
    """
    unique, counts = np.unique(y, return_counts=True)
    labels = ['Normal (0)', 'Attack (1)']

    plt.figure()
    plt.bar(labels, counts, color=['skyblue', 'salmon'])
    plt.ylabel('Count')
    plt.title('Class Distribution in Test Set')
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.01*max(counts), str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Loading data and engineering features for test set...")
    X, y = load_and_engineer_features()

    # Split into train, val, test (we only need test)
    _, _, X_test, _, _, y_test = time_based_split(X, y)

    print(f"Test set size: {len(X_test)}")
    print(f"Test attack rate: {y_test.mean():.4f}")

    # Load models and thresholds
    print("Loading trained models and thresholds...")
    models, threshold_info = load_models()

    # Get thresholds for each model
    lr_threshold = threshold_info['logistic_regression_threshold']
    rf_threshold = threshold_info['random_forest_threshold']
    iso_threshold = threshold_info['isolation_forest_threshold']

    thresholds = {
        'logistic_regression': lr_threshold,
        'random_forest': rf_threshold,
        'isolation_forest': iso_threshold
    }

    # Get prediction scores
    print("Getting prediction scores from models...")
    scores_dict = get_model_scores(models, X_test)

    # Compute metrics for each model
    print("\nComputing metrics on test set...")
    metrics_dict = {}
    for model_name in scores_dict.keys():
        metrics = compute_metrics(
            y_test.values,
            scores_dict[model_name],
            thresholds[model_name]
        )
        metrics_dict[model_name] = metrics

    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON ON TEST SET")
    print("="*80)
    print(f"{'Model':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10} {'PR-AUC':<10}")
    print("-"*80)
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name.replace('_', ' ').title():<25} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['roc_auc']:<10.4f} {metrics['pr_auc']:<10.4f}")
    print("-"*80)

    # Save metrics to CSV
    os.makedirs("reports", exist_ok=True)
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    metrics_df.to_csv("reports/metrics_comparison.csv")
    print(f"\nMetrics saved to reports/metrics_comparison.csv")

    # Generate and save plots
    print("\nGenerating plots...")
    plot_confusion_matrices(y_test.values, scores_dict, thresholds, "reports")
    plot_roc_curves(y_test.values, scores_dict, "reports")
    plot_pr_curves(y_test.values, scores_dict, "reports")
    plot_rf_feature_importance(models['random_forest'], list(X_test.columns), "reports", top_n=15)
    plot_class_distribution(y_test.values, "reports")
    print("Plots saved to reports/ directory")

    # Print note about accuracy
    print("\n" + "="*80)
    print("NOTE ON ACCURACY")
    print("="*80)
    always_normal_accuracy = (y_test == 0).mean()
    print(f"If we always predict normal (0), accuracy would be: {always_normal_accuracy:.4f}")
    print(f"Accuracy is misleading because the dataset is highly imbalanced "
          f"(only {y_test.mean()*100:.2f}% attacks).")
    print("A high accuracy could be achieved by simply predicting the majority class, "
          "which is useless for detecting attacks.")
    print("="*80)

    # Check for suspicious perfect scores
    print("\n" + "="*80)
    print("CHECKING FOR SUSPICIOUS PERFECT SCORES")
    print("="*80)
    suspicious = False
    for model_name, metrics in metrics_dict.items():
        if metrics['precision'] == 1.0 or metrics['recall'] == 1.0 or \
           metrics['f1'] == 1.0 or metrics['roc_auc'] == 1.0 or metrics['pr_auc'] == 1.0:
            print(f"WARNING: {model_name} has a perfect score of 1.0 in some metric.")
            suspicious = True
    if not suspicious:
        print("No model has suspicious perfect scores (all metrics < 1.0).")
    print("="*80)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
