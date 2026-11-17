"""
Tests for feature engineering.
"""

import pandas as pd
import numpy as np
from src.features import engineer_features


def test_no_future_leakage():
    """
    Test that history features do not use future information.
    We'll create a small synthetic dataset where we know the answer.
    """
    # Create a simple DataFrame with known timestamps and values
    data = {
        'event_id': ['evt_00000000', 'evt_00000001', 'evt_00000002', 'evt_00000003', 'evt_00000004'],
        'user_id': ['user_1', 'user_1', 'user_1', 'user_2', 'user_2'],
        'timestamp': pd.to_datetime([
            '2026-01-01 10:00:00',
            '2026-01-01 10:05:00',
            '2026-01-01 10:10:00',
            '2026-01-01 10:00:00',
            '2026-01-01 10:15:00'
        ]),
        'ip_address': ['1.1.1.1', '1.1.1.1', '1.1.1.1', '2.2.2.2', '2.2.2.2'],
        'ip_country': ['US', 'US', 'US', 'CA', 'CA'],
        'device_id': ['device_a', 'device_a', 'device_b', 'device_c', 'device_c'],
        'login_success': [True, True, False, True, True],
        'is_vpn_or_tor': [False, False, False, False, False],
        'is_attack': [False, False, False, False, False]
    }
    df = pd.DataFrame(data)
    # Engineer features
    X, y = engineer_features(df)
    # For the first event of each user, history features should be zero/default
    # First event of user_1 (index 0)
    assert X.iloc[0]['time_since_last_login'] == 0
    assert X.iloc[0]['device_seen_count'] == 0
    assert X.iloc[0]['country_seen_count'] == 0
    assert X.iloc[0]['is_new_device'] == 1  # first device
    assert X.iloc[0]['is_new_country'] == 1  # first country
    assert X.iloc[0]['is_new_ip'] == 1  # first IP
    # Second event of user_1 (index 1)
    # time_since_last_login should be 5 minutes = 300 seconds
    assert X.iloc[1]['time_since_last_login'] == 300
    # device_seen_count: before this event, we have seen device_a once -> count 1
    assert X.iloc[1]['device_seen_count'] == 1
    # country_seen_count: before this event, we have seen US once -> count 1
    assert X.iloc[1]['country_seen_count'] == 1
    # is_new_device: device_a already seen -> 0
    assert X.iloc[1]['is_new_device'] == 0
    # is_new_country: US already seen -> 0
    assert X.iloc[1]['is_new_country'] == 0
    # is_new_ip: IP already seen -> 0
    assert X.iloc[1]['is_new_ip'] == 0
    # Third event of user_1 (index 2)
    # time_since_last_login: from 10:05 to 10:10 = 5 min = 300 sec
    assert X.iloc[2]['time_since_last_login'] == 300
    # device_seen_count: before this event, we have seen device_a (twice) and device_b? Wait, we have not seen device_b yet.
    # Actually, at index 2, the device is device_b (new). So before this event, we have seen device_a only.
    assert X.iloc[2]['device_seen_count'] == 1
    # country_seen_count: still US only -> 1
    assert X.iloc[2]['country_seen_count'] == 1
    # is_new_device: device_b is new -> 1
    assert X.iloc[2]['is_new_device'] == 1
    # is_new_country: US already seen -> 0
    assert X.iloc[2]['is_new_country'] == 0
    # is_new_ip: IP already seen -> 0
    assert X.iloc[2]['is_new_ip'] == 0
    # For user_2, first event (index 3)
    assert X.iloc[3]['time_since_last_login'] == 0
    assert X.iloc[3]['device_seen_count'] == 0
    assert X.iloc[3]['country_seen_count'] == 0
    assert X.iloc[3]['is_new_device'] == 1
    assert X.iloc[3]['is_new_country'] == 1
    assert X.iloc[3]['is_new_ip'] == 1
    # Second event of user_2 (index 4)
    # time_since_last_login: from 10:00 to 10:15 = 15 min = 900 sec
    assert X.iloc[4]['time_since_last_login'] == 900
    # device_seen_count: before this event, we have seen device_c once -> count 1
    assert X.iloc[4]['device_seen_count'] == 1
    # country_seen_count: before this event, we have seen CA once -> count 1
    assert X.iloc[4]['country_seen_count'] == 1
    # is_new_device: device_c already seen -> 0
    assert X.iloc[4]['is_new_device'] == 0
    # is_new_country: CA already seen -> 0
    assert X.iloc[4]['is_new_country'] == 0
    # is_new_ip: IP already seen -> 0
    assert X.iloc[4]['is_new_ip'] == 0


def test_hour_sin_cos_range():
    """
    Test that hour_sin and hour_cos are in the correct range [-1, 1].
    """
    df = pd.read_csv("data/raw/synthetic_logins.csv")
    X, y = engineer_features(df)
    # Check that hour_sin and hour_cos are between -1 and 1
    assert ((X['hour_sin'] >= -1) & (X['hour_sin'] <= 1)).all()
    assert ((X['hour_cos'] >= -1) & (X['hour_cos'] <= 1)).all()
    # Also check that sin^2 + cos^2 = 1 (approximately)
    np.testing.assert_allclose(X['hour_sin']**2 + X['hour_cos']**2, 1.0, rtol=1e-10)


def test_is_new_device_correctness():
    """
    Test that is_new_device correctly indicates whether the device is new for the user.
    We'll use a small deterministic dataset.
    """
    data = {
        'event_id': ['evt_00000000', 'evt_00000001', 'evt_00000002', 'evt_00000003', 'evt_00000004'],
        'user_id': ['user_a', 'user_a', 'user_a', 'user_b', 'user_b'],
        'timestamp': pd.to_datetime([
            '2026-01-01 10:00:00',
            '2026-01-01 10:05:00',
            '2026-01-01 10:10:00',
            '2026-01-01 10:00:00',
            '2026-01-01 10:15:00'
        ]),
        'device_id': ['dev1', 'dev1', 'dev2', 'dev3', 'dev3'],
        'login_success': [True, True, True, True, True],
        'is_vpn_or_tor': [False, False, False, False, False],
        'ip_address': ['1.1.1.1', '1.1.1.1', '1.1.1.1', '2.2.2.2', '2.2.2.2'],
        'ip_country': ['US', 'US', 'US', 'CA', 'CA'],
        'is_attack': [False, False, False, False, False]
    }
    df = pd.DataFrame(data)
    X, y = engineer_features(df)
    # For user_a:
    #   event 0: dev1 first -> is_new_device = 1
    #   event 1: dev1 seen -> is_new_device = 0
    #   event 2: dev2 first -> is_new_device = 1
    assert X.iloc[0]['is_new_device'] == 1
    assert X.iloc[1]['is_new_device'] == 0
    assert X.iloc[2]['is_new_device'] == 1
    # For user_b:
    #   event 0: dev3 first -> is_new_device = 1
    #   event 1: dev3 seen -> is_new_device = 0
    assert X.iloc[3]['is_new_device'] == 1
    assert X.iloc[4]['is_new_device'] == 0


if __name__ == "__main__":
    test_no_future_leakage()
    test_hour_sin_cos_range()
    test_is_new_device_correctness()
    print("All tests passed.")