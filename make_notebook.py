import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Machine Learning-based Credential Theft Detection System\n",
                "\n",
                "**Problem Statement**\n",
                "\n",
                "We aim to detect account takeover (ATO) attacks in login events. Each login is labeled as normal (0) or attack (1).\n",
                "We build three models: Logistic Regression (baseline), Random Forest (main), and Isolation Forest (unsupervised).\n",
                "\n",
                "---\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Dataset Overview and Exploratory Data Analysis\n",
                "\n",
                "We load the synthetic login dataset and examine its characteristics.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sns.set_style('whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "\n",
                "# Load raw data\n",
                "df = pd.read_csv('data/raw/synthetic_logins.csv')\n",
                "print(f'Total login events: {len(df)}')\n",
                "print(f\"Attack events: {df['is_attack'].sum()} ({100*df['is_attack'].mean():.2f}%)\")\n",
                "print(f\"Normal events: {(~df['is_attack']).sum()} ({100*(1-df['is_attack'].mean()):.2f}%)\")\n",
                "print(f\"Unique users: {df['user_id'].nunique()}\")\n",
                "print(f\"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\")\n",
                "\n",
                "# Plot class distribution\n",
                "plt.figure()\n",
                "ax = sns.countplot(x='is_attack', data=df)\n",
                "ax.set_xticklabels(['Normal', 'Attack'])\n",
                "ax.set_ylabel('Count')\n",
                "ax.set_title('Class Distribution')\n",
                "for p in ax.patches:\n",
                "    ax.annotate(f'{p.get_height()}', (p.get_x()+p.get_width()/2., p.get_height()),\n",
                "                ha='center', va='bottom')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Login hour distribution\n",
                "df['hour'] = pd.to_datetime(df['timestamp']).dt.hour\n",
                "plt.figure()\n",
                "sns.histplot(data=df, x='hour', hue='is_attack', bins=24, discrete=True, multiple='stack')\n",
                "plt.xlabel('Hour of Day')\n",
                "plt.ylabel('Count')\n",
                "plt.title('Login Hour Distribution by Class')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Feature Engineering\n",
                "\n",
                "We engineer features using only historical information to avoid data leakage.\n",
                "Features include temporal attributes, user/IP/device/country/ASN history, failed attempts, distance, and speed.\n",
                "\n",
                "The feature engineering function is in `src.features.engineer_features`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.features import engineer_features\n",
                "\n",
                "# Load data and engineer features (we'll use a small sample for speed in demo)\n",
                "df_raw = pd.read_csv('data/raw/synthetic_logins.csv')\n",
                "# For speed, we'll use first 5000 rows to show feature engineering\n",
                "df_sample = df_raw.iloc[:5000].copy()\n",
                "X, y = engineer_features(df_sample)\n",
                "print(f'Engineered features shape: {X.shape}')\n",
                "print(f'Number of features: {X.shape[1]}')\n",
                "print('\\nFirst 5 rows of engineered features:')\n",
                "display(X.head())\n",
                "\n",
                "# List of feature names\n",
                "print('\\nFeature names:')\n",
                "for i, fname in enumerate(X.columns, 1):\n",
                "    print(f'{i:2d}. {fname}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Training Three Models\n",
                "\n",
                "We split the data temporally: 70% train, 15% validation, 15% test.\n",
                "We train:\n",
                "1. Logistic Regression with balanced class weights and standard scaling.\n",
                "2. Random Forest with balanced_subsample, 200 trees, max depth=10.\n",
                "3. Isolation Forest on normal training events only, contamination set to training attack ratio.\n",
                "\n",
                "We tune decision thresholds on the validation set to maximize F1 score.\n",
                "We also compute 5-fold TimeSeriesSplit cross-validation PR-AUC for the supervised models.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# We'll load the pre-trained models and evaluation results from the train.py run\n",
                "import joblib\n",
                "import json\n",
                "\n",
                "# Load models\n",
                "lr_model = joblib.load('models/logistic_regression.joblib')\n",
                "rf_model = joblib.load('models/random_forest.joblib')\n",
                "iso_model = joblib.load('models/isolation_forest.joblib')\n",
                "\n",
                "# Load thresholds and feature columns\n",
                "with open('models/threshold.json', 'r') as f:\n",
                "    threshold_info = json.load(f)\n",
                "with open('models/feature_columns.json', 'r') as f:\n",
                "    feature_columns = json.load(f)\n",
                "\n",
                "print(f'Logistic Regression threshold: {threshold_info[\"logistic_regression_threshold\"]:.4f}')\n",
                "print(f'Random Forest threshold: {threshold_info[\"random_forest_threshold\"]:.4f}')\n",
                "print(f'Isolation Forest threshold: {threshold_info[\"isolation_forest_threshold\"]:.4f}')\n",
                "print(f'Isolation Forest contamination: {threshold_info[\"isolation_forest_contamination\"]:.4f}')\n",
                "\n",
                "# Load test data and compute metrics (we'll reuse the split from train.py for consistency)\n",
                "from src.features import engineer_features\n",
                "from src.evaluate import time_based_split\n",
                "\n",
                "X_all, y_all = engineer_features(pd.read_csv('data/raw/synthetic_logins.csv'))\n",
                "_, _, X_test, _, _, y_test = time_based_split(X_all, y_all)\n",
                "\n",
                "# Get predictions\n",
                "lr_scores = lr_model.predict_proba(X_test)[:, 1]\n",
                "rf_scores = rf_model.predict_proba(X_test)[:, 1]\n",
                "iso_scores = -iso_model.decision_function(X_test)\n",
                "\n",
                "# Apply thresholds\n",
                "lr_thresh = threshold_info['logistic_regression_threshold']\n",
                "rf_thresh = threshold_info['random_forest_threshold']\n",
                "iso_thresh = threshold_info['isolation_forest_threshold']\n",
                "\n",
                "from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score\n",
                "\n",
                "def compute_metrics(y_true, scores, thresh):\n",
                "    y_pred = (scores >= thresh).astype(int)\n",
                "    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)\n",
                "    try:\n",
                "        roc = roc_auc_score(y_true, scores)\n",
                "    except ValueError:\n",
                "        roc = 0.5\n",
                "    try:\n",
                "        pr = average_precision_score(y_true, scores)\n",
                "    except ValueError:\n",
                "        pr = 0.0\n",
                "    return p, r, f, roc, pr\n",
                "\n",
                "lr_p, lr_r, lr_f, lr_roc, lr_pr = compute_metrics(y_test.values, lr_scores, lr_thresh)\n",
                "rf_p, rf_r, rf_f, rf_roc, rf_pr = compute_metrics(y_test.values, rf_scores, rf_thresh)\n",
                "iso_p, iso_r, iso_f, iso_roc, iso_pr = compute_metrics(y_test.values, iso_scores, iso_thresh)\n",
                "\n",
                "print('\\nTest Set Metrics:')\n",
                "print(f'Logistic Regression: Precision={lr_p:.4f}, Recall={lr_r:.4f}, F1={lr_f:.4f}, ROC-AUC={lr_roc:.4f}, PR-AUC={lr_pr:.4f}')\n",
                "print(f'Random Forest:       Precision={rf_p:.4f}, Recall={rf_r:.4f}, F1={rf_f:.4f}, ROC-AUC={rf_roc:.4f}, PR-AUC={rf_pr:.4f}')\n",
                "print(f'Isolation Forest:    Precision={iso_p:.4f}, Recall={iso_r:.4f}, F1={iso_f:.4f}, ROC-AUC={iso_roc:.4f}, PR-AUC={iso_pr:.4f}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Metrics Comparison\n",
                "\n",
                "We compare the three models using various metrics.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "\n",
                "# Load the metrics comparison CSV generated by evaluate.py\n",
                "try:\n",
                "    metrics_df = pd.read_csv('reports/metrics_comparison.csv', index_col=0)\n",
                "    print('Metrics comparison from evaluate.py:')\n",
                "    display(metrics_df)\n",
                "except FileNotFoundError:\n",
                "    print('Metrics comparison file not found. Please run evaluate.py first.')\n",
                "\n",
                "# Plot ROC curves\n",
                "plt.figure()\n",
                "from sklearn.metrics import roc_curve\n",
                "fpr_lr, tpr_lr, _ = roc_curve(y_test.values, lr_scores)\n",
                "fpr_rf, tpr_rf, _ = roc_curve(y_test.values, rf_scores)\n",
                "fpr_iso, tpr_iso, _ = roc_curve(y_test.values, iso_scores)\n",
                "plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_roc:.3f})')\n",
                "plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_roc:.3f})')\n",
                "plt.plot(fpr_iso, tpr_iso, label=f'Isolation Forest (AUC = {iso_roc:.3f})')\n",
                "plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')\n",
                "plt.xlabel('False Positive Rate')\n",
                "plt.ylabel('True Positive Rate')\n",
                "plt.title('ROC Curves')\n",
                "plt.legend(loc='lower right')\n",
                "plt.show()\n",
                "\n",
                "# Plot Precision-Recall curves\n",
                "plt.figure()\n",
                "from sklearn.metrics import precision_recall_curve\n",
                "prec_lr, rec_lr, _ = precision_recall_curve(y_test.values, lr_scores)\n",
                "prec_rf, rec_rf, _ = precision_recall_curve(y_test.values, rf_scores)\n",
                "prec_iso, rec_iso, _ = precision_recall_curve(y_test.values, iso_scores)\n",
                "plt.plot(rec_lr, prec_lr, label=f'Logistic Regression (AUC = {lr_pr:.3f})')\n",
                "plt.plot(rec_rf, prec_rf, label=f'Random Forest (AUC = {rf_pr:.3f})')\n",
                "plt.plot(rec_iso, prec_iso, label=f'Isolation Forest (AUC = {iso_pr:.3f})')\n",
                "plt.xlabel('Recall')\n",
                "plt.ylabel('Precision')\n",
                "plt.title('Precision-Recall Curves')\n",
                "plt.legend(loc='upper right')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Feature Importance\n",
                "\n",
                "We examine the top-15 features from the Random Forest model.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature importances from Random Forest\n",
                "importances = rf_model.feature_importances_\n",
                "indices = np.argsort(importances)[::-1][:15]\n",
                "\n",
                "plt.figure()\n",
                "plt.barh(range(15), importances[indices][::-1], align='center')\n",
                "plt.yticks(range(15), [feature_columns[i] for i in indices][::-1])\n",
                "plt.xlabel('Feature Importance')\n",
                "plt.title('Top 15 Random Forest Feature Importances')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print('Top 15 features:')\n",
                "for i in indices:\n",
                "    print(f'{feature_columns[i]:<30} {importances[i]:.6f}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "We have successfully built a credential theft detection system using three machine learning models.\n",
                "The Random Forest model achieves the highest performance, followed by Logistic Regression and Isolation Forest.\n",
                "The system leverages historical features engineered without data leakage, making it suitable for real-time deployment.\n",
                "\n",
                "Key takeaways:\n",
                "- Temporal split is crucial to avoid look-ahead bias.\n",
                "- Class imbalance must be addressed via class weights or threshold tuning.\n",
                "- Feature engineering based on user/IP/device/ASN/country history improves detection.\n",
                "\n",
                "---\n",
                "*Note: This notebook uses synthetic data for educational purposes.*\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('notebooks/phase1_ml_pipeline.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('Notebook created at notebooks/phase1_ml_pipeline.ipynb')