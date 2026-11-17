"""
Predict module for scoring login events using trained models.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

# Import our feature engineering function
from src.features import engineer_features
from src.datasets import load_project_dataset
import joblib


def score_login(event_dict: dict, history_df: pd.DataFrame) -> dict:
    """
    Score a single login event given historical data.

    Args:
        event_dict: Dictionary representing the login event with keys:
            user_id, timestamp, ip_address, ip_country, ip_asn,
            device_id, user_agent, login_success, session_id, is_vpn_or_tor
            (is_attack is not needed)
        history_df: DataFrame of historical login events (must have the same columns
                    as the raw data, sorted by timestamp). This should NOT include
                    the current event.

    Returns:
        Dictionary with keys:
            attack_probability: float from Random Forest (probability of attack)
            anomaly_score: float from Isolation Forest (higher = more anomalous)
            risk_band: string, one of "normal", "suspicious", "high-risk"
            top_contributing_features: list of top 3 feature names (by Random Forest importance)
    """
    # Load models and metadata
    rf_model = joblib.load("models/random_forest.joblib")
    iso_model = joblib.load("models/isolation_forest.joblib")
    with open("models/feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    with open("models/threshold.json", "r") as f:
        threshold_info = json.load(f)
    if history_df is None:
        history_df = pd.DataFrame(columns=[
            "user_id", "timestamp", "ip_address", "ip_country", "ip_asn",
            "device_id", "user_agent", "login_success", "session_id", "is_vpn_or_tor",
        ])

    # Prepare the event DataFrame
    event_df = pd.DataFrame([event_dict])
    # Ensure timestamp is datetime
    event_df['timestamp'] = pd.to_datetime(event_df['timestamp'])

    # Ensure history_df timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(history_df['timestamp']):
        history_df = history_df.copy()
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

    # History must not contain events later than the event being scored.
    event_time = event_df['timestamp'].iloc[0]
    history_df = history_df.loc[history_df['timestamp'] <= event_time].copy()
    combined_df = pd.concat([history_df, event_df], ignore_index=True)
    combined_df = combined_df.sort_values('timestamp', kind='mergesort').reset_index(drop=True)
    # Keep the current event last when historical events share its timestamp.
    current_event = combined_df.iloc[[-1]]
    prior_events = combined_df.iloc[:-1].sort_values('timestamp', kind='mergesort')
    combined_df = pd.concat([prior_events, current_event], ignore_index=True)

    # Engineer features (this will compute features for all rows, using only historical info due to shifting)
    featured_df, _ = engineer_features(combined_df)
    # The event was appended after all prior events, so its feature row is last.
    event_features = featured_df.iloc[[-1]]  # Keep as DataFrame for consistency

    # Ensure we have the expected feature columns (in case any missing)
    # Reindex to match the training feature columns, filling missing with 0
    event_features = event_features.reindex(columns=feature_columns, fill_value=0)

    # Get Random Forest attack probability
    attack_prob = rf_model.predict_proba(event_features)[0, 1]

    # Get Isolation Forest anomaly score (higher = more anomalous)
    # Decision function: higher means more like inliers (normal). We negate.
    anomaly_score = -iso_model.decision_function(event_features)[0]

    # Use the validation-tuned Random Forest threshold saved during training.
    threshold = float(threshold_info['random_forest_threshold'])
    if attack_prob < threshold * 0.5:
        risk_band = "normal"
    elif attack_prob < threshold:
        risk_band = "suspicious"
    else:
        risk_band = "high-risk"

    # Report the model's three globally most important features.
    importances = rf_model.feature_importances_
    # Get indices of top 3 importances
    top_indices = np.argsort(importances)[::-1][:3]
    top_features = [feature_columns[i] for i in top_indices]

    return {
        "attack_probability": float(attack_prob),
        "anomaly_score": float(anomaly_score),
        "risk_band": risk_band,
        "top_contributing_features": top_features
    }


def _create_demo_event(user_id: str, timestamp: str, ip_address: str, ip_country: str,
                       ip_asn: int, device_id: str, user_agent: str,
                       login_success: bool, session_id: str, is_vpn_or_tor: bool) -> dict:
    """Helper to create a demo event dictionary."""
    return {
        "user_id": user_id,
        "timestamp": timestamp,
        "ip_address": ip_address,
        "ip_country": ip_country,
        "ip_asn": ip_asn,
        "device_id": device_id,
        "user_agent": user_agent,
        "login_success": login_success,
        "session_id": session_id,
        "is_vpn_or_tor": is_vpn_or_tor
    }


def demo():
    """Run a demonstration scoring three hand-made logins."""
    print("="*60)
    print("DEMONSTRATION: Scoring three hand-made login events")
    print("="*60)

    # We need a history DataFrame. For demonstration, we'll use a small slice of the training data
    # as history. We'll load the raw data, engineer features, and take a subset as history.
    # However, to keep the demo self-contained, we'll create a minimal history from the raw data.
    # Let's load the raw data and take the first 1000 rows as history.
    print("Loading historical data for demo...")
    df_raw = load_project_dataset()
    # Take first 1000 rows as history
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], utc=True)
    df_raw = df_raw.sort_values('timestamp', kind='mergesort').reset_index(drop=True)
    history_raw = df_raw.iloc[:1000].copy()
    # Demo events must be later than their historical context.
    demo_time = (df_raw['timestamp'].max() + pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
    # We don't need to engineer features here because score_login will do it.
    # But we need to pass the raw history (as DataFrame) to score_login.
    history_df = history_raw

    # Event 1: Normal login (typical behavior for a user)
    # We'll pick a user from the history and mimic their usual pattern.
    # For simplicity, we'll create an event that looks normal.
    normal_event = _create_demo_event(
        user_id="user_00500",
        timestamp=demo_time,
        ip_address="192.168.1.100",
        ip_country="US",
        ip_asn=7018,
        device_id="device_0b88be42",  # a device seen before for this user (we assume)
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        login_success=True,
        session_id="sess_1234567",
        is_vpn_or_tor=False
    )

    # Event 2: Suspicious login (somewhat unusual but not clearly malicious)
    suspicious_event = _create_demo_event(
        user_id="user_00500",
        timestamp=(pd.Timestamp(demo_time) + pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S'),  # unusual hour
        ip_address="10.0.0.50",
        ip_country="US",
        ip_asn=1239,
        device_id="device_99999999",  # new device
        user_agent="curl/7.68.0",
        login_success=True,
        session_id="sess_9876543",
        is_vpn_or_tor=True
    )

    # Event 3: Clear attack (e.g., credential stuffing from Tor)
    attack_event = _create_demo_event(
        user_id="user_00500",
        timestamp=(pd.Timestamp(demo_time) + pd.Timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S'),
        ip_address="185.220.101.10",  # Tor exit node IP (example)
        ip_country="RU",
        ip_asn=14061,
        device_id="device_11111111",
        user_agent="Python-urllib/3.9",
        login_success=False,  # credential stuffing often fails
        session_id="",  # no session if failed
        is_vpn_or_tor=True
    )

    events = [
        ("Normal Login", normal_event),
        ("Suspicious Login", suspicious_event),
        ("Clear Attack", attack_event)
    ]

    for label, event in events:
        print(f"\n--- {label} ---")
        result = score_login(event, history_df)
        print(f"Attack Probability: {result['attack_probability']:.4f}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"Risk Band: {result['risk_band']}")
        print(f"Top Contributing Features: {', '.join(result['top_contributing_features'])}")

    print("\n" + "="*60)
    print("Demo complete.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score login events for credential theft detection.")
    parser.add_argument("--demo", action="store_true", help="Run a demonstration with three hand-made logins.")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        parser.print_help()
