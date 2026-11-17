"""
Train machine learning models for credential theft detection.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Import our feature engineering function
from src.features import engineer_features
from src.datasets import load_project_dataset


def load_and_engineer_features() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw synthetic login data and engineer features.

    Returns:
        X: DataFrame with engineered features
        y: Series with target (is_attack)
    """
    # Load raw data
    df_raw = load_project_dataset()
    # Engineer features
    X, y = engineer_features(df_raw)
    return X, y


def time_based_split(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train Logistic Regression model with balanced class weights and standard scaling.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Trained pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train Random Forest model with balanced_subsample and limited depth.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Trained Random Forest classifier
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,  # limited depth to avoid overfitting
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def train_isolation_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[IsolationForest, float]:
    """
    Train Isolation Forest on normal training events only.

    Args:
        X_train: Training features
        y_train: Training target (to filter normal events)

    Returns:
        Trained Isolation Forest and contamination level used
    """
    # Filter to normal events only
    X_train_normal = X_train[y_train == 0]
    # Compute contamination as the attack ratio in the training set
    attack_ratio = y_train.mean()
    contamination = attack_ratio  # IsolationForest expects contamination as the proportion of outliers

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_normal)
    return iso_forest, contamination


def tune_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Tune decision threshold on validation set to maximize F1 score.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores (probability of positive class or anomaly score)

    Returns:
        Optimal threshold that maximizes F1
    """
    # For probability-like scores where higher means more likely positive
    # Support probabilities and arbitrary anomaly scores on different scales.
    thresholds = np.unique(np.quantile(y_scores, np.linspace(0, 1, 101)))
    best_f1 = -1
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh)


def evaluate_model(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute evaluation metrics for a model.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        threshold: Decision threshold to convert scores to binary predictions

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_scores >= threshold).astype(int)
    # For metrics that need probabilities, we use y_scores directly
    # For binary metrics, we use y_pred
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # ROC AUC and PR AUC need scores
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = 0.5  # default if only one class
    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError:
        pr_auc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': [tn, fp, fn, tp]
    }


def main():
    print("Loading data and engineering features...")
    X, y = load_and_engineer_features()

    # Ensure data is sorted by timestamp (engineer_features already sorts)
    # But we'll sort again to be safe
    # Actually, engineer_features sorts by timestamp, so we assume it's sorted.

    print("Splitting data into train, validation, test (time-based)...")
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Train attack rate: {y_train.mean():.4f}")
    print(f"Validation attack rate: {y_val.mean():.4f}")
    print(f"Test attack rate: {y_test.mean():.4f}")

    # Train models
    print("\nTraining Logistic Regression...")
    lr_pipe = train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("Training Isolation Forest...")
    iso_forest, contamination = train_isolation_forest(X_train, y_train)

    # Get validation scores for threshold tuning
    print("\nGetting validation scores...")
    # For Logistic Regression, we need probability of positive class
    y_val_lr_scores = lr_pipe.predict_proba(X_val)[:, 1]
    # For Random Forest, probability of positive class
    y_val_rf_scores = rf_model.predict_proba(X_val)[:, 1]
    # For Isolation Forest, we need to convert decision_function to an anomaly score where higher means more anomalous
    # Decision function: higher means more like inliers (normal). We'll negate so that higher means more anomalous.
    y_val_iso_scores = -iso_forest.decision_function(X_val)

    # Tune threshold for each model on validation set (maximize F1)
    print("\nTuning thresholds on validation set...")
    lr_threshold = tune_threshold(y_val.values, y_val_lr_scores)
    rf_threshold = tune_threshold(y_val.values, y_val_rf_scores)
    iso_threshold = tune_threshold(y_val.values, y_val_iso_scores)

    print(f"Logistic Regression optimal threshold: {lr_threshold:.4f}")
    print(f"Random Forest optimal threshold: {rf_threshold:.4f}")
    print(f"Isolation Forest optimal threshold: {iso_threshold:.4f}")

    # Evaluate on validation set with tuned thresholds
    print("\nValidation set performance:")
    lr_metrics = evaluate_model(y_val.values, y_val_lr_scores, lr_threshold)
    rf_metrics = evaluate_model(y_val.values, y_val_rf_scores, rf_threshold)
    iso_metrics = evaluate_model(y_val.values, y_val_iso_scores, iso_threshold)

    print(f"Logistic Regression: F1={lr_metrics['f1']:.4f}, ROC-AUC={lr_metrics['roc_auc']:.4f}, PR-AUC={lr_metrics['pr_auc']:.4f}")
    print(f"Random Forest: F1={rf_metrics['f1']:.4f}, ROC-AUC={rf_metrics['roc_auc']:.4f}, PR-AUC={rf_metrics['pr_auc']:.4f}")
    print(f"Isolation Forest: F1={iso_metrics['f1']:.4f}, ROC-AUC={iso_metrics['roc_auc']:.4f}, PR-AUC={iso_metrics['pr_auc']:.4f}")

    # Evaluate on test set with the tuned thresholds
    print("\nTest set performance:")
    y_test_lr_scores = lr_pipe.predict_proba(X_test)[:, 1]
    y_test_rf_scores = rf_model.predict_proba(X_test)[:, 1]
    y_test_iso_scores = -iso_forest.decision_function(X_test)

    lr_test_metrics = evaluate_model(y_test.values, y_test_lr_scores, lr_threshold)
    rf_test_metrics = evaluate_model(y_test.values, y_test_rf_scores, rf_threshold)
    iso_test_metrics = evaluate_model(y_test.values, y_test_iso_scores, iso_threshold)

    print(f"Logistic Regression: F1={lr_test_metrics['f1']:.4f}, ROC-AUC={lr_test_metrics['roc_auc']:.4f}, PR-AUC={lr_test_metrics['pr_auc']:.4f}")
    print(f"Random Forest: F1={rf_test_metrics['f1']:.4f}, ROC-AUC={rf_test_metrics['roc_auc']:.4f}, PR-AUC={rf_test_metrics['pr_auc']:.4f}")
    print(f"Isolation Forest: F1={iso_test_metrics['f1']:.4f}, ROC-AUC={iso_test_metrics['roc_auc']:.4f}, PR-AUC={iso_test_metrics['pr_auc']:.4f}")

    # 5-fold TimeSeriesSplit cross-validation on training data for supervised models
    print("\nPerforming 5-fold TimeSeriesSplit cross-validation on training data...")
    tscv = TimeSeriesSplit(n_splits=5)

    lr_pr_aucs = []
    rf_pr_aucs = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Check if training data has both classes
        if len(np.unique(y_fold_train)) < 2:
            print(f"Fold {fold+1}: Training data has only one class. Skipping LR and RF for this fold.")
            continue

        # Logistic Regression
        lr_fold = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
        ])
        lr_fold.fit(X_fold_train, y_fold_train)
        y_fold_val_lr_scores = lr_fold.predict_proba(X_fold_val)[:, 1]
        lr_pr_aucs.append(average_precision_score(y_fold_val, y_fold_val_lr_scores))

        # Random Forest
        rf_fold = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        rf_fold.fit(X_fold_train, y_fold_train)
        y_fold_val_rf_scores = rf_fold.predict_proba(X_fold_val)[:, 1]
        rf_pr_aucs.append(average_precision_score(y_fold_val, y_fold_val_rf_scores))

        print(f"Fold {fold+1}: LR PR-AUC={lr_pr_aucs[-1]:.4f}, RF PR-AUC={rf_pr_aucs[-1]:.4f}")

    if len(lr_pr_aucs) == 0:
        print("Warning: No folds had two classes for Logistic Regression. Setting PR-AUC to 0.5.")
        lr_pr_aucs = [0.5]
    if len(rf_pr_aucs) == 0:
        print("Warning: No folds had two classes for Random Forest. Setting PR-AUC to 0.5.")
        rf_pr_aucs = [0.5]

    print(f"\nAverage PR-AUC over {len(lr_pr_aucs)} valid folds for LR and {len(rf_pr_aucs)} for RF:")
    print(f"Logistic Regression: {np.mean(lr_pr_aucs):.4f} +/- {np.std(lr_pr_aucs):.4f}")
    print(f"Random Forest: {np.mean(rf_pr_aucs):.4f} +/- {np.std(rf_pr_aucs):.4f}")

    # Save models and metadata
    print("\nSaving models and metadata...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_pipe, "models/logistic_regression.joblib")
    joblib.dump(rf_model, "models/random_forest.joblib")
    joblib.dump(iso_forest, "models/isolation_forest.joblib")

    # Save threshold and contamination
    threshold_info = {
        "logistic_regression_threshold": float(lr_threshold),
        "random_forest_threshold": float(rf_threshold),
        "isolation_forest_threshold": float(iso_threshold),
        "isolation_forest_contamination": float(contamination)
    }
    with open("models/threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=2)

    # Save feature columns
    feature_columns = list(X.columns)
    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
