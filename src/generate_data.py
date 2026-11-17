"""
Synthetic login data generation for credential theft detection system.
Generates realistic login events with 5 types of attack patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Tuple, Optional
import hashlib
import argparse


def generate_synthetic_logins(
    n_events: int = 100000,
    attack_ratio: float = 0.02,
    random_seed: int = 42,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate synthetic login dataset with controlled attack patterns.

    Args:
        n_events: Total number of login events to generate
        attack_ratio: Proportion of events that are attacks (0.01-0.03)
        random_seed: Seed for reproducibility
        save_path: Optional path to save CSV file

    Returns:
        DataFrame with synthetic login events
    """
    if n_events < 10:
        raise ValueError("n_events must be at least 10")
    if not 0 < attack_ratio < 1:
        raise ValueError("attack_ratio must be between 0 and 1")
    if int(n_events * attack_ratio) == 0:
        raise ValueError("n_events and attack_ratio must produce at least one attack event")

    np.random.seed(random_seed)

    # Calculate normal vs attack events
    n_attacks = int(n_events * attack_ratio)
    n_normal = n_events - n_attacks

    print(f"Generating {n_events} login events...")
    print(f"  Normal: {n_normal} ({100*(1-attack_ratio):.1f}%)")
    print(f"  Attacks: {n_attacks} ({100*attack_ratio:.1f}%)")

    # Generate base normal login data
    normal_data = _generate_normal_logins(n_normal, random_seed)

    # Generate attack data
    attack_data = _generate_attack_logins(n_attacks, random_seed)

    # Combine and shuffle
    df = pd.concat([normal_data, attack_data], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Add event IDs
    df['event_id'] = [f"evt_{i:08d}" for i in range(len(df))]

    # Reorder columns
    column_order = [
        'event_id', 'user_id', 'timestamp', 'ip_address', 'ip_country',
        'ip_asn', 'device_id', 'user_agent', 'login_success', 'session_id',
        'is_vpn_or_tor', 'is_attack'
    ]
    df = df[column_order]

    # Save if path provided
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved synthetic data to {save_path}")

    return df


def _generate_normal_logins(n_normal: int, random_seed: int) -> pd.DataFrame:
    """Generate normal login behavior."""
    np.random.seed(random_seed)

    # User population (power-law distribution)
    n_users = max(100, int(n_normal ** 0.5))
    user_ids = [f"user_{i:05d}" for i in range(n_users)]
    user_weights = np.random.zipf(1.5, n_users)
    user_weights = user_weights / user_weights.sum()

    # Time range: last 90 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)

    # Generate normal logins
    data = []
    for _ in range(n_normal):
        # Select user with power-law distribution
        user_idx = np.random.choice(len(user_ids), p=user_weights)
        user_id = user_ids[user_idx]

        # Timestamp with realistic patterns (more during business hours)
        time_offset = np.random.exponential(scale=90*24*3600/2)  # Bias toward recent
        time_offset = min(time_offset, 90*24*3600)
        timestamp = start_time + timedelta(seconds=time_offset)

        # Add circadian pattern (more likely 8am-6pm)
        hour_weight = 1.0
        hour = timestamp.hour
        if 8 <= hour <= 18:
            hour_weight = 2.0
        elif 22 <= hour or hour <= 6:
            hour_weight = 0.3

        if np.random.random() > hour_weight / 3.0:
            # Adjust timestamp slightly
            timestamp += timedelta(minutes=np.random.randint(-30, 30))

        # IP address (correlated with user - same ASN/country)
        base_ip = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        # Add small variation for same user
        ip_parts = base_ip.split('.')
        ip_parts[-1] = str(int(ip_parts[-1]) + np.random.randint(-5, 5))
        ip_address = '.'.join(ip_parts)

        # Country and ASN (realistic mapping)
        countries = ['US', 'GB', 'DE', 'FR', 'CA', 'AU', 'JP', 'IN', 'BR', 'MX']
        country = np.random.choice(countries, p=[0.3, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])

        asn_mapping = {
            'US': [7018, 7922, 1239, 209, 6327],
            'GB': [5089, 2856, 13238],
            'DE': [3320, 24940, 8560],
            'FR': [3215, 8422, 15557],
            'CA': [577, 852, 14593],
            'AU': [4804, 7575, 9443],
            'JP': [2516, 4713, 9370],
            'IN': [45528, 9498, 55836],
            'BR': [26599, 27699, 52868],
            'MX': [8151, 11888, 6503]
        }
        ip_asn = np.random.choice(asn_mapping[country])

        # Device ID (consistent per user with occasional changes)
        if np.random.random() < 0.95:  # 95% same device
            device_salt = hashlib.md5(user_id.encode()).hexdigest()[:8]
            device_id = f"device_{device_salt}"
        else:
            device_id = f"device_{np.random.randint(10000000, 99999999)}"

        # User agent (realistic distribution)
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox"
        ]
        user_agent = np.random.choice(user_agents)

        # Login success (mostly successful for normal)
        login_success = np.random.random() < 0.95

        # Session ID
        session_id = f"sess_{np.random.randint(1000000, 9999999)}"

        # VPN/TOR (rare for normal)
        is_vpn_or_tor = np.random.random() < 0.02

        # Not an attack
        is_attack = False

        data.append({
            'user_id': user_id,
            'timestamp': timestamp,
            'ip_address': ip_address,
            'ip_country': country,
            'ip_asn': ip_asn,
            'device_id': device_id,
            'user_agent': user_agent,
            'login_success': login_success,
            'session_id': session_id,
            'is_vpn_or_tor': is_vpn_or_tor,
            'is_attack': is_attack
        })

    return pd.DataFrame(data)


def _generate_attack_logins(n_attacks: int, random_seed: int) -> pd.DataFrame:
    """Generate attack login events with 5 distinct patterns."""
    np.random.seed(random_seed + 1)  # Different seed for attacks

    # Distribute attacks across 5 types
    attack_types = [
        'credential_stuffing',
        'brute_force',
        'phishing_ato',
        'session_hijacking',
        'impossible_travel'
    ]

    # Distribution: credential stuffing most common, impossible travel least
    type_weights = [0.35, 0.25, 0.20, 0.15, 0.05]
    attack_type_counts = np.random.multinomial(n_attacks, type_weights)

    attack_data = []

    # Generate each attack type
    for attack_type, count in zip(attack_types, attack_type_counts):
        if count > 0:
            if attack_type == 'credential_stuffing':
                attack_data.extend(_generate_credential_stuffing(count, random_seed))
            elif attack_type == 'brute_force':
                attack_data.extend(_generate_brute_force(count, random_seed))
            elif attack_type == 'phishing_ato':
                attack_data.extend(_generate_phishing_ato(count, random_seed))
            elif attack_type == 'session_hijacking':
                attack_data.extend(_generate_session_hijacking(count, random_seed))
            elif attack_type == 'impossible_travel':
                attack_data.extend(_generate_impossible_travel(count, random_seed))

    return pd.DataFrame(attack_data)


def _generate_credential_stuffing(n: int, random_seed: int) -> list:
    """Credential stuffing: known credentials tried across many IPs."""
    np.random.seed(random_seed)

    data = []
    # Use compromised credential pairs
    compromised_users = [f"user_{i:05d}" for i in range(0, 50)]  # First 50 users
    compromised_passwords = [f"pass_{i:03d}" for i in range(20)]  # Common passwords

    for _ in range(n):
        user_id = np.random.choice(compromised_users)

        # Timestamp: rapid fire attempts
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 30))
        timestamp = base_time + timedelta(seconds=np.random.randint(0, 86400))

        # IP: distributed across many different IPs (botnet)
        ip_address = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"

        # Country: often from high-risk countries
        high_risk_countries = ['RU', 'CN', 'KP', 'IR', 'SY']
        country = np.random.choice(high_risk_countries) if np.random.random() < 0.3 else np.random.choice(['US', 'GB', 'DE'])

        # ASN: often hosting/VPS providers
        vps_asns = [16276, 24940, 14061, 32475, 63949]
        ip_asn = np.random.choice(vps_asns)

        # Device: new/unknown device
        device_id = f"device_{np.random.randint(10000000, 99999999)}"

        # User agent: often headless/bot-like
        bot_agents = [
            "Python-urllib/3.9",
            "java/1.8.0_291",
            "curl/7.68.0",
            "Mozilla/5.0 (compatible; Googlebot/2.1)",
            "python-requests/2.25.1"
        ]
        user_agent = np.random.choice(bot_agents)

        # Login success: low rate (credential stuffing has low success)
        login_success = np.random.random() < 0.02

        # Session ID: only if successful
        session_id = f"sess_{np.random.randint(1000000, 9999999)}" if login_success else ""

        # VPN/TOR: often used
        is_vpn_or_tor = np.random.random() < 0.4

        # Attack flag
        is_attack = True

        data.append({
            'user_id': user_id,
            'timestamp': timestamp,
            'ip_address': ip_address,
            'ip_country': country,
            'ip_asn': ip_asn,
            'device_id': device_id,
            'user_agent': user_agent,
            'login_success': login_success,
            'session_id': session_id,
            'is_vpn_or_tor': is_vpn_or_tor,
            'is_attack': is_attack
        })

    return data


def _generate_brute_force(n: int, random_seed: int) -> list:
    """Brute force: many attempts on single account from few IPs."""
    np.random.seed(random_seed + 10)

    data = []
    target_users = [f"user_{i:05d}" for i in range(50, 100)]  # Different users

    for _ in range(n):
        user_id = np.random.choice(target_users)

        # Timestamp: concentrated in short bursts
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 60))
        # Burst of attempts over minutes/hours
        burst_start = base_time + timedelta(hours=np.random.randint(0, 24))
        timestamp = burst_start + timedelta(minutes=np.random.randint(0, 120))

        # IP: few IPs trying many passwords (rotating)
        base_ip = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        # Rotate through 1-3 IPs
        ip_octets = list(map(int, base_ip.split('.')))
        ip_octets[-1] = (ip_octets[-1] + np.random.randint(-2, 2)) % 256
        ip_address = '.'.join(map(str, ip_octets))

        # Country: usually attacker location
        country = np.random.choice(['RU', 'CN', 'BR', 'VE', 'PK'])

        # ASN: often residential or mobile
        ip_asn = np.random.choice([209, 7922, 1239, 6327, 7018])

        # Device: varied but consistent within attack
        device_id = f"device_{np.random.randint(1000000, 9999999)}"

        # User agent: common browsers/automation
        ua_list = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
        ]
        user_agent = np.random.choice(ua_list)

        # Login success: very low until lucky guess
        login_success = np.random.random() < 0.001

        # Session ID
        session_id = f"sess_{np.random.randint(1000000, 9999999)}" if login_success else ""

        # VPN/TOR: sometimes used
        is_vpn_or_tor = np.random.random() < 0.25

        # Attack flag
        is_attack = True

        data.append({
            'user_id': user_id,
            'timestamp': timestamp,
            'ip_address': ip_address,
            'ip_country': country,
            'ip_asn': ip_asn,
            'device_id': device_id,
            'user_agent': user_agent,
            'login_success': login_success,
            'session_id': session_id,
            'is_vpn_or_tor': is_vpn_or_tor,
            'is_attack': is_attack
        })

    return data


def _generate_phishing_ato(n: int, random_seed: int) -> list:
    """Phishing ATO: compromised credentials from phishing, used from new locations/devices."""
    np.random.seed(random_seed + 20)

    data = []
    # Target users who fell for phishing
    target_users = [f"user_{i:05d}" for i in range(100, 150)]

    for _ in range(n):
        user_id = np.random.choice(target_users)

        # Timestamp: delay after phishing (hours to days)
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 45))
        delay_hours = np.random.exponential(scale=24)  # Average 1 day delay
        timestamp = base_time + timedelta(hours=delay_hours)

        # IP: new location (different from user's normal)
        # Generate IP from different country/ASN
        country = np.random.choice(['US', 'GB', 'DE', 'FR', 'CA'])  # Victim's perceived location
        # But actually from attacker location
        attacker_countries = ['NL', 'SE', 'CH', 'AT', 'DK']
        ip_country = np.random.choice(attacker_countries) if np.random.random() < 0.7 else country

        # IP generation
        if ip_country in ['NL', 'SE', 'CH', 'AT', 'DK']:
            ip_asn = np.random.choice([14061, 20773, 5430, 29017, 56103])
            ip_base = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        else:
            ip_asn = np.random.choice([7018, 7922, 1239, 209, 6327])
            ip_base = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"

        ip_address = ip_base

        # Device: new device (phishing victim logging in from attacker's device)
        device_id = f"device_{np.random.randint(10000000, 99999999)}"  # Definitely new

        # User agent: might be different (attacker's setup)
        ua_list = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0; rv:86.0) Gecko/20100101 Firefox/86.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        user_agent = np.random.choice(ua_list)

        # Login success: high (valid credentials)
        login_success = np.random.random() < 0.85

        # Session ID
        session_id = f"sess_{np.random.randint(1000000, 9999999)}" if login_success else ""

        # VPN/TOR: sometimes used by attacker
        is_vpn_or_tor = np.random.random() < 0.3

        # Attack flag
        is_attack = True

        data.append({
            'user_id': user_id,
            'timestamp': timestamp,
            'ip_address': ip_address,
            'ip_country': ip_country,
            'ip_asn': ip_asn,
            'device_id': device_id,
            'user_agent': user_agent,
            'login_success': login_success,
            'session_id': session_id,
            'is_vpn_or_tor': is_vpn_or_tor,
            'is_attack': is_attack
        })

    return data


def _generate_session_hijacking(n: int, random_seed: int) -> list:
    """Session hijacking: stealing active cookies/sessions."""
    np.random.seed(random_seed + 30)

    data = []
    # Target active users
    target_users = [f"user_{i:05d}" for i in range(150, 200)]

    for _ in range(n):
        user_id = np.random.choice(target_users)

        # Timestamp: during or shortly after legitimate session
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 30))
        # Hijack happens during active period
        timestamp = base_time + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))

        # IP: attacker's location (often different)
        attacker_countries = ['UA', 'BY', 'MD', 'GE', 'AM']
        ip_country = np.random.choice(attacker_countries) if np.random.random() < 0.6 else np.random.choice(['US', 'GB'])

        ip_asn = np.random.choice([6855, 12389, 20857, 31334, 49505])  # Various
        ip_address = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"

        # Device: different from victim's usual
        device_id = f"device_{np.random.randint(10000000, 99999999)}"

        # User agent: might mimic victim or be different
        if np.random.random() < 0.3:
            # Try to mimic victim's UA
            user_agent = np.random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            ])
        else:
            user_agent = np.random.choice([
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)"
            ])

        # Login success: high (using valid session)
        login_success = np.random.random() < 0.9

        # Session ID: either stolen or new session
        if login_success and np.random.random() < 0.7:
            # Using stolen session
            session_id = f"sess_{np.random.randint(100000, 999999)}"  # 6-digit session ID (stolen)
        else:
            session_id = f"sess_{np.random.randint(1000000, 9999999)}"

        # VPN/TOR: often used to hide tracks
        is_vpn_or_tor = np.random.random() < 0.5

        # Attack flag
        is_attack = True

        data.append({
            'user_id': user_id,
            'timestamp': timestamp,
            'ip_address': ip_address,
            'ip_country': ip_country,
            'ip_asn': ip_asn,
            'device_id': device_id,
            'user_agent': user_agent,
            'login_success': login_success,
            'session_id': session_id,
            'is_vpn_or_tor': is_vpn_or_tor,
            'is_attack': is_attack
        })

    return data


def _generate_impossible_travel(n: int, random_seed: int) -> list:
    """Impossible travel: logins from geographically distant locations in short time."""
    np.random.seed(random_seed + 40)

    data = []
    # Users who establish normal pattern first
    target_users = [f"user_{i:05d}" for i in range(200, 250)]

    # Define distant location pairs
    location_pairs = [
        (('US', 'AS15169'), ('SG', 'AS4837')),   # US to Singapore
        (('GB', 'AS2856'), ('AU', 'AS4804')),    # UK to Australia
        (('DE', 'AS3320'), ('JP', 'AS2516')),    # Germany to Japan
        (('BR', 'AS26599'), ('FR', 'AS3215')),   # Brazil to France
        (('CA', 'AS577'), ('IN', 'AS45528'))     # Canada to India
    ]

    for _ in range(n):
        user_id = np.random.choice(target_users)

        # First establish normal pattern (legitimate login from location A)
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 50))
        # Choose a random location pair
        pair_idx = np.random.randint(0, len(location_pairs))
        (country_a, asn_a), (country_b, asn_b) = location_pairs[pair_idx]

        # Legitimate login from location A
        legit_time = base_time + timedelta(hours=np.random.randint(0, 12))
        legit_ip = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"

        # Impossible login from location B shortly after
        # Time gap too short for travel
        time_gap_hours = int(np.random.choice([1, 2, 3, 4, 5]))  # Impossible to travel
        attack_time = legit_time + timedelta(hours=time_gap_hours)

        # Use the attack time for this record
        timestamp = attack_time

        # IP from location B (impossible)
        ip_address = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        ip_country = country_b
        ip_asn = asn_b

        # Device: might be similar or different
        if np.random.random() < 0.5:
            # Same device (impossible unless cloned)
            device_salt = hashlib.md5(user_id.encode()).hexdigest()[:8]
            device_id = f"device_{device_salt}"
        else:
            device_id = f"device_{np.random.randint(10000000, 99999999)}"

        # User agent: might be similar
        if np.random.random() < 0.4:
            user_agent = np.random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            ])
        else:
            user_agent = np.random.choice([
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
            ])

        # Login success: variable
        login_success = np.random.random() < 0.6

        # Session ID
        session_id = f"sess_{np.random.randint(1000000, 9999999)}" if login_success else ""

        # VPN/TOR: often used to spoof location
        is_vpn_or_tor = np.random.random() < 0.6

        # Attack flag
        is_attack = True

        data.append({
            'user_id': user_id,
            'timestamp': timestamp,
            'ip_address': ip_address,
            'ip_country': ip_country,
            'ip_asn': ip_asn,
            'device_id': device_id,
            'user_agent': user_agent,
            'login_success': login_success,
            'session_id': session_id,
            'is_vpn_or_tor': is_vpn_or_tor,
            'is_attack': is_attack
        })

    return data


def load_rba_dataset(path: str) -> pd.DataFrame:
    """
    Load optional RBA (Risk-Based Authentication) dataset for comparison/testing.

    Args:
        path: Path to RBA dataset CSV

    Returns:
        DataFrame with RBA data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"RBA dataset not found at {path}")

    df = pd.read_csv(path)
    print(f"Loaded RBA dataset with {len(df)} records from {path}")

    # Ensure required columns exist
    required_cols = ['user_id', 'timestamp', 'ip_address', 'login_success']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add is_attack column if not present (assume all are normal for RBA data)
    if 'is_attack' not in df.columns:
        df['is_attack'] = False

    # Add event_id if not present
    if 'event_id' not in df.columns:
        df['event_id'] = [f"rba_evt_{i:08d}" for i in range(len(df))]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic login events for the defensive ATO detection pipeline.")
    parser.add_argument("--events", type=int, default=100000, help="number of login events (default: 100000)")
    parser.add_argument("--attack-ratio", type=float, default=0.02, help="fraction of events labeled as attacks")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output", default="data/raw/synthetic_logins.csv", help="output CSV path")
    args = parser.parse_args()

    df = generate_synthetic_logins(
        n_events=args.events,
        attack_ratio=args.attack_ratio,
        random_seed=args.seed,
        save_path=args.output
    )

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total events: {len(df)}")
    print(f"Attack events: {df['is_attack'].sum()} ({100*df['is_attack'].mean():.2f}%)")
    print(f"Normal events: {(~df['is_attack']).sum()} ({100*(1-df['is_attack'].mean()):.2f}%)")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Attack type distribution (approximate)
    print("\nAttack type distribution (approximate):")
    print("  Credential stuffing: ~35%")
    print("  Brute force: ~25%")
    print("  Phishing ATO: ~20%")
    print("  Session hijacking: ~15%")
    print("  Impossible travel: ~5%")
