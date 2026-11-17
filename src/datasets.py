"""Dataset discovery helpers for the included project CSVs."""

from pathlib import Path
from typing import Optional

import pandas as pd


DATASET_CANDIDATES = (
    Path("data/raw/synthetic_logins.csv"),
    Path("data/basic_featured_logins.csv"),
    Path("data/featured_logins_subset_5000.csv"),
)


def find_dataset(path: Optional[str] = None) -> Path:
    """Return an explicit dataset path or the best available bundled dataset."""
    if path:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Dataset not found: {candidate}")
        return candidate
    for candidate in DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No project dataset found. Add data/basic_featured_logins.csv or "
        "generate data/raw/synthetic_logins.csv with `python -m src.generate_data`."
    )


def load_project_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load project login events and normalize the common input fields."""
    resolved = find_dataset(path)
    df = pd.read_csv(resolved)
    if "timestamp" not in df:
        raise ValueError(f"Dataset {resolved} has no timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True)
    if "is_attack" not in df:
        raise ValueError(f"Dataset {resolved} has no is_attack labels")
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
