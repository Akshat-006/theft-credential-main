"""Streamlit dashboard for the synthetic account takeover detection project."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.datasets import load_project_dataset


ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Sentinel | Login Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#0a1020; --panel:#111a2d; --panel2:#151f34; --line:#26334d;
            --muted:#91a0b9; --mint:#4ee0b2; --blue:#7ca8ff; }
    .stApp { background: radial-gradient(ellipse at 80% -10%, #172b49 0, var(--ink) 48%); color:#eef3ff; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f4f7ff; }
    [data-testid="stSidebar"] { background:#0c1425; border-right:1px solid var(--line); }
    [data-testid="stDecoration"] { display:none; }
    header[data-testid="stHeader"] { background:transparent; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,var(--panel2),var(--panel)); padding:18px 20px;
      border:1px solid var(--line); border-radius:16px; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { color:#f3f7ff; font-family:'Space Grotesk',sans-serif; }
    div[data-testid="stForm"] { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; }
    .hero { padding:24px 28px; border:1px solid #263752; border-radius:20px;
      background:linear-gradient(110deg,rgba(35,61,89,.78),rgba(17,26,45,.86)); margin:4px 0 22px; }
    .hero-kicker { color:var(--mint); font-size:12px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; }
    .hero-title { font:700 34px 'Space Grotesk',sans-serif; margin:8px 0 5px; color:#f4f7ff; }
    .hero-copy { color:#aab8ce; font-size:15px; margin:0; }
    .section-note { color:var(--muted); font-size:13px; margin-top:-8px; }
    .distribution { margin:24px 2px 12px; height:18px; display:flex; overflow:hidden; border-radius:12px;
      background:#1e2a40; border:1px solid #32425e; }
    .distribution-normal { background:#7898d0; }
    .distribution-attack { background:#ff7e8a; min-width:3px; }
    .legend { display:flex; gap:18px; color:#aab8ce; font-size:13px; margin:10px 0 4px; }
    .legend span { display:inline-flex; align-items:center; gap:7px; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; }
    .stButton button, .stFormSubmitButton button { border-radius:10px; border:0; background:var(--mint); color:#062219; font-weight:700; }
    .stButton button:hover, .stFormSubmitButton button:hover { background:#79edc8; color:#062219; }
    hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the prepared login dataset…")
def get_default_data():
    return load_project_dataset()


def normalize_uploaded_data(file) -> pd.DataFrame:
    frame = pd.read_csv(file)
    required = {"timestamp", "is_attack"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("CSV must include: " + ", ".join(sorted(required)))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
    frame["is_attack"] = frame["is_attack"].astype(bool)
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


with st.sidebar:
    st.markdown("## 🛡️ Sentinel")
    st.caption("LOGIN SECURITY")
    st.markdown("---")
    uploaded = st.file_uploader("Use another prepared dataset", type=["csv"])
    st.caption("CSV needs `timestamp` and `is_attack` columns.")
    st.markdown("---")
    st.markdown("**Detection models**")
    st.caption("Random Forest · Logistic Regression · Isolation Forest")
    st.markdown("**Data source**")
    st.caption("Prepared login activity")

try:
    data = normalize_uploaded_data(uploaded) if uploaded else get_default_data()
except Exception as error:
    st.error(f"Could not load the login dataset: {error}")
    st.stop()

if "is_attack" not in data:
    st.error("The selected dataset needs an `is_attack` label column.")
    st.stop()
data["is_attack"] = data["is_attack"].astype(bool)
if "login_success" in data:
    data["login_success"] = data["login_success"].astype(bool)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Security overview</div>
      <div class="hero-title">Login threat monitor</div>
      <p class="hero-copy">A clear view of account takeover signals, model performance, and recent login activity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

start_day = data["timestamp"].min().date()
end_day = data["timestamp"].max().date()
chosen = st.date_input("Activity period", value=(start_day, end_day), min_value=start_day, max_value=end_day)
if isinstance(chosen, tuple) and len(chosen) == 2:
    start_date, end_date = chosen
else:
    start_date = end_date = chosen
filtered = data[
    (data["timestamp"].dt.date >= start_date)
    & (data["timestamp"].dt.date <= end_date)
].copy()
if filtered.empty:
    st.info("No login events fall within this period.")
    st.stop()

total = len(filtered)
attacks = int(filtered["is_attack"].sum())
attack_pct = attacks / total if total else 0.0
failed = int((~filtered["login_success"]).sum()) if "login_success" in filtered else 0
accounts = filtered["user_id"].nunique() if "user_id" in filtered else "—"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Events analyzed", f"{total:,}")
k2.metric("Labeled attack events", f"{attacks:,}")
k3.metric("Failed logins", f"{failed:,}")
k4.metric("Accounts observed", f"{accounts:,}" if isinstance(accounts, int) else accounts)

st.markdown("### Activity")
st.markdown('<div class="section-note">Login volume and labeled attacks over time</div>', unsafe_allow_html=True)
daily = filtered.set_index("timestamp").resample("D").agg(
    logins=("is_attack", "size"), attacks=("is_attack", "sum")
).fillna(0)
left, right = st.columns([1.65, 1])
with left:
    st.caption("Daily login volume")
    st.area_chart(daily[["logins"]], color="#6686bd", height=270)
with right:
    st.caption("Event classification")
    st.line_chart(daily[["attacks"]], color="#ff7e8a", height=270)
    normal_pct = 100 * (total - attacks) / total
    attack_pct = 100 * attacks / total
    st.markdown(
        f'<div class="distribution"><div class="distribution-normal" style="width:{normal_pct:.3f}%"></div>'
        f'<div class="distribution-attack" style="width:{attack_pct:.3f}%"></div></div>'
        f'<div class="legend"><span><i class="dot" style="background:#7898d0"></i>Normal {normal_pct:.1f}%</span>'
        f'<span><i class="dot" style="background:#ff7e8a"></i>Attack {attack_pct:.1f}%</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("### Model performance")
st.markdown('<div class="section-note">Held-out test performance · thresholds selected on validation data</div>', unsafe_allow_html=True)
metrics_path = ROOT / "reports" / "metrics_comparison.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path, index_col=0)
    display_cols = [c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"] if c in metrics]
    if display_cols:
        column_names = {
            "precision": "Precision", "recall": "Recall", "f1": "F1 score",
            "roc_auc": "ROC AUC", "pr_auc": "PR AUC",
        }
        metrics.index = [str(name).replace("_", " ").title() for name in metrics.index]
        metrics = metrics.rename(columns=column_names)
        display_cols = [column_names[c] for c in display_cols]
        st.dataframe(
            metrics[display_cols].style.format("{:.3f}").highlight_max(axis=0, color="#153a36"),
            use_container_width=True,
        )
else:
    st.info("Train and evaluate the models to create `reports/metrics_comparison.csv`.")

st.markdown("### Recent login events")
event_cols = [c for c in ["timestamp", "user_id", "ip_country", "login_success", "is_vpn_or_tor", "is_attack"] if c in filtered]
recent = filtered.sort_values("timestamp", ascending=False).head(12)[event_cols].copy()
recent = recent.rename(columns={
    "timestamp": "Time (UTC)", "user_id": "Account", "ip_country": "Country",
    "login_success": "Login result", "is_vpn_or_tor": "VPN / Tor", "is_attack": "Classification",
})
recent["Time (UTC)"] = recent["Time (UTC)"].dt.strftime("%Y-%m-%d %H:%M")
if "Login result" in recent:
    recent["Login result"] = recent["Login result"].map({True: "Successful", False: "Failed"})
if "VPN / Tor" in recent:
    recent["VPN / Tor"] = recent["VPN / Tor"].map({True: "Yes", False: "No"})
if "Classification" in recent:
    recent["Classification"] = recent["Classification"].map({True: "Attack", False: "Normal"})
st.dataframe(recent, use_container_width=True, hide_index=True)

st.markdown("### Score a login")
st.markdown('<div class="section-note">Score an event against the trained model using available account history</div>', unsafe_allow_html=True)
with st.form("score_login_form"):
    accounts_list = sorted(data["user_id"].dropna().astype(str).unique()) if "user_id" in data else []
    c1, c2, c3 = st.columns(3)
    with c1:
        chosen_user = st.selectbox("Account", accounts_list) if accounts_list else st.text_input("Account", "demo_user")
        ip = st.text_input("IP address", "", placeholder="Enter an IP address")
    with c2:
        country = st.text_input("Country code", "US", max_chars=2).upper()
        device = st.text_input("Device ID", "", placeholder="Enter a device identifier")
    with c3:
        success = st.selectbox("Login result", ["Successful", "Failed"]) == "Successful"
        vpn = st.checkbox("VPN or Tor")
    submitted = st.form_submit_button("Analyze login")

if submitted:
    try:
        from src.predict import score_login

        event_time = pd.Timestamp(datetime.now(timezone.utc))
        event = {
            "user_id": chosen_user,
            "timestamp": event_time,
            "ip_address": ip.strip(),
            "ip_country": country.strip(),
            "ip_asn": 0,
            "device_id": device.strip(),
            "user_agent": "Sentinel Security Dashboard",
            "login_success": success,
            "session_id": "demo-session" if success else "",
            "is_vpn_or_tor": vpn,
        }
        recent_history = data[
            (data["timestamp"] < event_time)
            & ((data["user_id"].astype(str) == str(chosen_user)) | (data["timestamp"] >= event_time - timedelta(days=1)))
        ].tail(10000)
        result = score_login(event, recent_history)
        st.markdown("#### Model assessment")
        r1, r2, r3 = st.columns(3)
        r1.metric("Attack probability", f"{result['attack_probability']:.1%}")
        r2.metric("Risk band", result["risk_band"].replace("-", " ").title())
        r3.metric("Anomaly score", f"{result['anomaly_score']:.3f}")
        st.caption("Leading model features: " + ", ".join(result["top_contributing_features"]))
    except Exception as error:
        st.error(f"Could not score this event. Train the model first with `python -m src.train`. Details: {error}")

st.markdown("---")
st.caption("Sentinel · Login security monitoring")
