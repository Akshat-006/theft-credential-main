"""Leakage-safe, chronological feature engineering for login events."""

from collections import defaultdict, deque
from typing import Tuple

import numpy as np
import pandas as pd


COUNTRY_CENTROIDS = {
    "US": (37.0902, -95.7129), "GB": (55.3781, -3.4360),
    "DE": (51.1657, 10.4515), "FR": (46.2276, 2.2137),
    "CA": (56.1304, -106.3468), "AU": (-25.2744, 133.7751),
    "JP": (36.2048, 138.2522), "IN": (20.5937, 78.9629),
    "BR": (-14.2350, -51.9253), "MX": (23.6345, -102.5528),
    "RU": (61.5240, 105.3188), "CN": (35.8617, 104.1954),
    "KP": (40.3399, 127.5101), "IR": (32.4279, 53.6880),
    "SY": (34.8021, 38.9960), "NL": (52.1326, 5.2913),
    "SE": (60.1282, 18.6438), "CH": (46.8182, 8.2275),
    "AT": (47.5162, 14.5501), "DK": (56.2639, 9.5018),
    "UA": (48.3794, 31.1656), "BY": (53.7098, 27.9534),
    "MD": (47.4116, 28.3699), "GE": (42.3154, 43.3569),
    "AM": (40.0691, 45.0382), "SG": (1.3521, 103.8198),
}

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "day_of_week", "is_weekend",
    "is_new_device", "is_new_country", "is_new_ip", "distance_km",
    "travel_speed_kmph", "failed_attempts_10min", "failed_attempts_24h",
    "ip_attempts_10min", "distinct_accounts_per_ip_10min",
    "time_since_last_login", "device_seen_count", "country_seen_count",
    "login_success", "is_vpn_or_tor", "session_ip_changed",
]

REQUIRED_COLUMNS = ["user_id", "timestamp", "ip_address", "login_success"]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres; accepts scalars or NumPy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return chronological numeric features and labels using event history only.

    The current event is not included in any historical count.  ``is_attack`` is
    optional so this same function can score unlabeled events in production.
    Results follow input row order; callers making chronological splits should
    sort their raw events before calling this function.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype="int64", name="is_attack")

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="raise", utc=True)
    work["login_success"] = work["login_success"].fillna(False).astype(bool)
    for col, default in (("ip_country", ""), ("device_id", ""),
                         ("is_vpn_or_tor", False), ("event_id", None)):
        if col not in work:
            work[col] = default
    work["is_vpn_or_tor"] = work["is_vpn_or_tor"].fillna(False).astype(bool)
    work["_original_order"] = np.arange(len(work))
    work = work.sort_values(["timestamp", "_original_order"], kind="mergesort").reset_index(drop=True)

    user_devices, user_countries, user_ips = defaultdict(set), defaultdict(set), defaultdict(set)
    user_prev, user_failed = {}, defaultdict(deque)
    ip_events = defaultdict(deque)
    feature_rows = []

    for row in work.itertuples(index=False):
        user = row.user_id
        ts = row.timestamp
        ip = row.ip_address
        country = row.ip_country
        device = row.device_id

        prior = user_prev.get(user)
        elapsed = max(0.0, (ts - prior[0]).total_seconds()) if prior else 0.0
        previous_country = prior[1] if prior else None
        if previous_country in COUNTRY_CENTROIDS and country in COUNTRY_CENTROIDS:
            distance = float(haversine_distance(*COUNTRY_CENTROIDS[previous_country], *COUNTRY_CENTROIDS[country]))
        else:
            distance = 0.0

        user_failures = user_failed[user]
        while user_failures and (ts - user_failures[0]).total_seconds() > 24 * 3600:
            user_failures.popleft()
        failed_24h = len(user_failures)
        failed_10m = sum((ts - fail_ts).total_seconds() <= 600 for fail_ts in user_failures)

        recent_ip_events = ip_events[ip]
        while recent_ip_events and (ts - recent_ip_events[0][0]).total_seconds() > 600:
            recent_ip_events.popleft()
        ip_attempts = len(recent_ip_events)
        distinct_users = len({seen_user for _, seen_user in recent_ip_events})

        hour = ts.hour
        feature_rows.append({
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week": ts.dayofweek,
            "is_weekend": int(ts.dayofweek >= 5),
            "is_new_device": int(device not in user_devices[user]),
            "is_new_country": int(country not in user_countries[user]),
            "is_new_ip": int(ip not in user_ips[user]),
            "distance_km": distance,
            "travel_speed_kmph": distance / (elapsed / 3600) if elapsed > 0 else 0.0,
            "failed_attempts_10min": failed_10m,
            "failed_attempts_24h": failed_24h,
            "ip_attempts_10min": ip_attempts,
            "distinct_accounts_per_ip_10min": distinct_users,
            "time_since_last_login": elapsed,
            "device_seen_count": len(user_devices[user]),
            "country_seen_count": len(user_countries[user]),
            "login_success": int(row.login_success),
            "is_vpn_or_tor": int(row.is_vpn_or_tor),
            "session_ip_changed": int(prior is not None and ip != prior[2]),
        })

        # Update history only after creating this event's features.
        user_devices[user].add(device)
        user_countries[user].add(country)
        user_ips[user].add(ip)
        user_prev[user] = (ts, country, ip)
        if not row.login_success:
            user_failures.append(ts)
        ip_events[ip].append((ts, user))

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS).astype(float)
    original_order = work["_original_order"].to_numpy()
    X["_original_order"] = original_order
    X = X.sort_values("_original_order").drop(columns="_original_order").reset_index(drop=True)
    if "is_attack" in work.columns:
        y = pd.Series(work["is_attack"].fillna(False).astype(int).to_numpy(), index=original_order)
        y = y.sort_index().reset_index(drop=True)
    else:
        y = pd.Series(np.zeros(len(work), dtype="int64"), name="is_attack")
    return X, y


if __name__ == "__main__":
    from pathlib import Path

    input_path = Path("data/raw/synthetic_logins.csv")
    if not input_path.exists():
        raise SystemExit("Dataset not found. Run `python -m src.generate_data` first.")
    X, y = engineer_features(pd.read_csv(input_path))
    print(f"Created {len(X)} feature rows with {X.shape[1]} features; attack rate={y.mean():.3%}")
