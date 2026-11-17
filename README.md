# Synthetic Login Account Takeover Detection

Defensive machine learning pipeline that flags
potential account takeover (ATO) activity in login event logs. It generates
synthetic data and compares Logistic Regression, Random Forest, and Isolation
Forest models. It does not collect credentials or interact with real accounts.

## Requirements

- Python 3.9 or newer
- Install the packages in `requirements.txt`

From this project directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the pipeline

The ZIP includes the prepared `data/basic_featured_logins.csv` dataset, which
the training and evaluation scripts use automatically. To regenerate a
synthetic dataset instead:

```bash
python -m src.generate_data --events 100000 --attack-ratio 0.02 --seed 42
```

For a quick smaller generated dataset, use `--events 5000`. Then train the
models and evaluate the held-out chronological test period:

```bash
python -m src.train
python -m src.evaluate
```

Launch the dashboard with:

```bash
python -m streamlit run app.py
```

The dashboard reads the bundled dataset by default and also accepts a compatible
CSV upload. It shows login volume, attack rate, model metrics, and recent events.
The model training and evaluation commands use the bundled prepared dataset
automatically, so regenerating data is optional.

Training writes three model files plus `models/feature_columns.json` and
`models/threshold.json`. Evaluation writes metric tables and plots under
`reports/`. The checked-in models and reports are example artifacts; regenerate
them after changing the data, features, or training code.

To score an event from Python, call `src.predict.score_login(event_dict,
history_df)`. Supply the event fields shown in that function's docstring and a
DataFrame containing earlier login events for context. The scorer returns a
Random Forest attack probability, an Isolation Forest anomaly score, a risk
band, and the Random Forest's three globally most important features. The
bundled `python -m src.predict --demo` example uses the generated dataset and
trained model files.

## Pipeline outline

1. `src.generate_data` creates labeled synthetic login events for normal and
   attack patterns.
2. `src.features.engineer_features` computes numeric temporal, account, device,
   IP, and location features using only events at or before each login. It
   returns rows in the input order, so training and evaluation sort the raw
   events chronologically before engineering features.
3. `src.train` makes a 70/15/15 time split, fits the three planned models, tunes
   classification thresholds on the validation period, and reports test
   metrics. Five-fold `TimeSeriesSplit` PR-AUC is reported for supervised models.
4. `src.evaluate` reproduces the held-out comparison and saves confusion
   matrices, ROC and precision-recall curves, and feature importance.

The notebook in `notebooks/phase1_ml_pipeline.ipynb` documents the same flow.
Run it from the project directory after generating data and training models.

## Data and evaluation notes

All bundled and generated login data is synthetic. The results are for learning
and pipeline development, not production authentication decisions. Country
centroids are approximate, and the generated labels are simulated rather than
ground truth from real incidents. Use a representative, privacy-reviewed,
properly labeled dataset and operational review before considering deployment.

`src/generate_data.load_rba_dataset()` can load an optional compatible RBA CSV;
it does not download external data. The project never needs real passwords or
authentication secrets.
