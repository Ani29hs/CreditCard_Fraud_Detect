import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FraudShield", page_icon="💳", layout="wide",
                   initial_sidebar_state="collapsed")

# ── Cache model loading ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model         = pickle.load(open("model/fraud_model.pkl",   "rb"))
    scaler_amount = pickle.load(open("model/scaler_amount.pkl", "rb"))
    scaler_time   = pickle.load(open("model/scaler_time.pkl",   "rb"))
    return model, scaler_amount, scaler_time

model, scaler_amount, scaler_time = load_models()

# ── Cache dataset ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv("Data/creditcard.csv")

df         = load_dataset()
fraud_rate = (df["Class"].sum() / len(df)) * 100
total_tx   = len(df)
fraud_count = int(df["Class"].sum())

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,700;1,300;1,700&display=swap');

:root {
  --bg:      #060a12;
  --text:    #f0f4ff;
  --muted:   #8a9ab8;
  --accent:  #3b82f6;
  --accent2: #8b5cf6;
  --mono:    'DM Mono', monospace;
}

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"],
.main, .block-container {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* Dot grid overlay */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(circle, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 32px 32px;
}
/* Ambient glow */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 50% at 15% 0%,   rgba(59,130,246,0.10) 0%, transparent 60%),
    radial-gradient(ellipse 55% 45% at 85% 100%, rgba(139,92,246,0.08) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; position: relative; z-index: 1; }
.element-container { margin-bottom: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* ── Buttons ── */
.stButton > button {
  all: unset !important;
  cursor: pointer !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 100% !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.03em !important;
  border-radius: 12px !important;
  height: 48px !important;
  padding: 0 1.5rem !important;
  box-sizing: border-box !important;
  transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
  color: #fff !important;
  box-shadow: 0 0 28px rgba(59,130,246,0.3), 0 4px 14px rgba(0,0,0,0.4) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 45px rgba(59,130,246,0.45), 0 8px 24px rgba(0,0,0,0.5) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
  background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ef4444) !important;
  border-radius: 999px !important;
  height: 10px !important;
}
.stProgress > div {
  background: rgba(255,255,255,0.06) !important;
  border-radius: 999px !important;
  height: 10px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── st.caption text ── */
[data-testid="stCaptionContainer"] p {
  color: #8a9ab8 !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.04em !important;
}

/* ── st.write text ── */
[data-testid="stMarkdownContainer"] p {
  color: #c8d8f0 !important;
  font-size: 0.9rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04) !important;
  backdrop-filter: blur(16px) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 14px !important;
  padding: 1.1rem 1.3rem !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
[data-testid="stMetricLabel"] p {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.62rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: #8a9ab8 !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Fraunces', serif !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  color: #f0f4ff !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  backdrop-filter: blur(14px) !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] {
  background: rgba(239,68,68,0.08) !important;
  border: 1px solid rgba(239,68,68,0.28) !important;
  color: #fca5a5 !important;
}
div.stSuccess {
  background: rgba(16,185,129,0.08) !important;
  border: 1px solid rgba(16,185,129,0.28) !important;
  color: #6ee7b7 !important;
}
div.stError {
  background: rgba(239,68,68,0.08) !important;
  border: 1px solid rgba(239,68,68,0.28) !important;
  color: #fca5a5 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:1.6rem 3rem 0;display:flex;align-items:center;justify-content:space-between;position:relative;z-index:2;">
  <div style="display:flex;align-items:center;gap:1rem;">
    <div style="width:48px;height:48px;border-radius:14px;flex-shrink:0;
      background:linear-gradient(135deg,#3b82f6,#8b5cf6);
      display:flex;align-items:center;justify-content:center;
      font-size:1.4rem;box-shadow:0 0 24px rgba(59,130,246,0.4);">💳</div>
    <div>
      <div style="font-family:'Fraunces',serif;font-size:1.55rem;font-weight:700;color:#f0f4ff;letter-spacing:-0.02em;line-height:1;">FraudShield</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#4a5a78;letter-spacing:0.14em;text-transform:uppercase;margin-top:3px;">Credit Card Fraud Detection System</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:0.45rem;font-family:'DM Mono',monospace;font-size:0.6rem;color:#10b981;letter-spacing:0.1em;background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.2);border-radius:999px;padding:0.3rem 0.9rem;">
    <span style="width:5px;height:5px;border-radius:50%;background:#10b981;display:inline-block;box-shadow:0 0 6px #10b981;"></span>MODEL ACTIVE
  </div>
</div>
<div style="margin:1rem 3rem 0;height:1px;background:linear-gradient(90deg,transparent,rgba(59,130,246,0.3),rgba(139,92,246,0.2),transparent);"></div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO STATS GLASS CARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="margin:0 3rem;padding:2rem 2.4rem;
  background:rgba(255,255,255,0.03);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:20px;
  box-shadow:0 8px 40px rgba(0,0,0,0.45),inset 0 1px 0 rgba(255,255,255,0.06);
  position:relative;overflow:hidden;">
  <div style="position:absolute;top:-50px;right:-50px;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(59,130,246,0.1),transparent 70%);pointer-events:none;"></div>

  <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.2em;text-transform:uppercase;color:#4a5a78;margin-bottom:1.2rem;">Dataset Overview</div>

  <div style="display:flex;gap:1rem;flex-wrap:wrap;">
    <div style="flex:1;min-width:120px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.18);border-radius:14px;padding:1.2rem 1.6rem;">
      <div style="font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:#93c5fd;line-height:1;">{total_tx:,}</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a6080;margin-top:0.35rem;">Total Transactions</div>
    </div>
    <div style="flex:1;min-width:120px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.18);border-radius:14px;padding:1.2rem 1.6rem;">
      <div style="font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:#fca5a5;line-height:1;">{fraud_rate:.3f}%</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#6a3a3a;margin-top:0.35rem;">Fraud Rate</div>
    </div>
    <div style="flex:1;min-width:120px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.18);border-radius:14px;padding:1.2rem 1.6rem;">
      <div style="font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:#6ee7b7;line-height:1;">{fraud_count:,}</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#0a3a28;margin-top:0.35rem;">Fraud Cases</div>
    </div>
    <div style="flex:1;min-width:120px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.18);border-radius:14px;padding:1.2rem 1.6rem;">
      <div style="font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:#c4b5fd;line-height:1;">{total_tx - fraud_count:,}</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#3a2a60;margin-top:0.35rem;">Legitimate</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ACTION BUTTON
# ══════════════════════════════════════════════════════════════════════════════
_, btn_col, _ = st.columns([0.25, 1, 5])
with btn_col:
    run = st.button("🎲 Analyse Random Transaction", use_container_width=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if run:

    sample = df.sample(1, weights=df["Class"].apply(lambda x: 0.6 if x==1 else 0.4))

    time   = sample["Time"].values[0]
    amount = sample["Amount"].values[0]
    v1     = sample["V1"].values[0]
    v2     = sample["V2"].values[0]
    v3     = sample["V3"].values[0]
    v4     = sample["V4"].values[0]
    v5     = sample["V5"].values[0]
    actual = sample["Class"].values[0]

    time_scaled   = scaler_time.transform([[time]])
    amount_scaled = scaler_amount.transform([[amount]])

    features = pd.DataFrame({
        "Time":   [time_scaled[0][0]],
        "Amount": [amount_scaled[0][0]],
        "V1": [v1], "V2": [v2], "V3": [v3], "V4": [v4], "V5": [v5]
    })

    try:
        prediction = model.predict(features)
        prob       = model.predict_proba(features)[0][1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    risk       = prob * 100
    confidence = max(prob, 1 - prob) * 100

    # ── Transaction snapshot ──────────────────────────────────────────────────
    st.markdown("""
    <div style="margin:0 3rem 12px;padding:1.2rem 1.8rem 0.8rem;
      background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);
      border:1px solid rgba(255,255,255,0.07);border-radius:16px;
      box-shadow:0 4px 24px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.05);">
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#6a7a98;margin-bottom:0.6rem;">Transaction Snapshot</div>
    </div>
    """, unsafe_allow_html=True)

    _, df_col, _ = st.columns([0.14, 1, 0.14])
    with df_col:
        st.dataframe(
            sample[["Time","Amount","V1","V2","V3","V4","V5","Class"]],
            use_container_width=True, hide_index=True
        )

    _, cap_col, _ = st.columns([0.14, 1, 0.14])
    with cap_col:
        st.caption("V1–V28 are PCA-transformed variables used to protect customer privacy.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Risk meter (components.html to avoid markdown parser issues) ──────────
    if risk < 30:
        risk_color = "#10b981"; risk_label = "LOW RISK"
        risk_bg = "rgba(16,185,129,0.08)";  risk_bdr = "rgba(16,185,129,0.25)"; risk_glow = "rgba(16,185,129,0.18)"
    elif risk < 70:
        risk_color = "#f59e0b"; risk_label = "MEDIUM RISK"
        risk_bg = "rgba(245,158,11,0.08)";  risk_bdr = "rgba(245,158,11,0.25)"; risk_glow = "rgba(245,158,11,0.18)"
    else:
        risk_color = "#ef4444"; risk_label = "HIGH RISK"
        risk_bg = "rgba(239,68,68,0.08)";   risk_bdr = "rgba(239,68,68,0.25)";  risk_glow = "rgba(239,68,68,0.22)"

    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@300;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>*{{margin:0;padding:0;box-sizing:border-box;}}</style>
</head>
<body style="background:transparent;padding:0 3rem;">
<div style="padding:1.8rem 2rem;background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.07);border-radius:16px;box-shadow:0 4px 24px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.05);position:relative;overflow:hidden;">
  <div style="position:absolute;top:-40px;right:-40px;width:160px;height:160px;border-radius:50%;background:radial-gradient(circle,{risk_glow},transparent 70%);pointer-events:none;"></div>

  <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#6a7a98;margin-bottom:1rem;">Fraud Risk Meter</div>

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;flex-wrap:wrap;gap:0.8rem;">
    <div>
      <div style="font-family:'Fraunces',serif;font-size:3rem;font-weight:700;color:{risk_color};line-height:1;">{risk:.1f}%</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#8a9ab8;margin-top:0.3rem;">Confidence: {confidence:.1f}%</div>
    </div>
    <div style="background:{risk_bg};border:1px solid {risk_bdr};border-radius:999px;padding:0.5rem 1.2rem;font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:{risk_color};display:flex;align-items:center;gap:0.5rem;">
      <span style="width:7px;height:7px;border-radius:50%;background:{risk_color};display:inline-block;box-shadow:0 0 8px {risk_color};"></span>{risk_label}
    </div>
  </div>

  <div style="background:rgba(255,255,255,0.05);border-radius:999px;height:12px;border:1px solid rgba(255,255,255,0.07);overflow:hidden;">
    <div style="height:100%;width:{risk:.1f}%;border-radius:999px;background:linear-gradient(90deg,#3b82f6,{risk_color});box-shadow:0 0 12px {risk_color};"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:0.5rem;">
    <span style="font-family:'DM Mono',monospace;font-size:0.54rem;color:#4a5a78;">0% — Safe</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.54rem;color:#6a3a3a;">100% — Fraud</span>
  </div>
</div>
</body></html>""", height=200, scrolling=False)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Prediction result cards ───────────────────────────────────────────────
    actual_label = "Fraud" if actual == 1 else "Legitimate"
    pred_label   = "Fraud" if prediction[0] == 1 else "Legitimate"
    actual_color = "#ef4444" if actual == 1 else "#10b981"
    pred_color   = "#ef4444" if prediction[0] == 1 else "#10b981"
    correct      = actual == prediction[0]
    match_label  = "✓ Correct" if correct else "✗ Mismatch"
    match_color  = "#10b981" if correct else "#f59e0b"

    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@300;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>*{{margin:0;padding:0;box-sizing:border-box;}}</style>
</head>
<body style="background:transparent;padding:0 3rem;">
<div style="padding:1.8rem 2rem;background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.07);border-radius:16px;box-shadow:0 4px 24px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.05);">
  <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#6a7a98;margin-bottom:1.2rem;">Prediction Result</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">

    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:1.3rem 1.5rem;text-align:center;">
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:#6a7a98;margin-bottom:0.6rem;">Actual Label</div>
      <div style="font-family:'Fraunces',serif;font-size:1.6rem;font-weight:700;color:{actual_color};">{actual_label}</div>
    </div>

    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:1.3rem 1.5rem;text-align:center;">
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:#6a7a98;margin-bottom:0.6rem;">Model Prediction</div>
      <div style="font-family:'Fraunces',serif;font-size:1.6rem;font-weight:700;color:{pred_color};">{pred_label}</div>
    </div>

    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:1.3rem 1.5rem;text-align:center;">
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:#6a7a98;margin-bottom:0.6rem;">Accuracy</div>
      <div style="font-family:'Fraunces',serif;font-size:1.6rem;font-weight:700;color:{match_color};">{match_label}</div>
    </div>

  </div>
</div>
</body></html>""", height=160, scrolling=False)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Final verdict ─────────────────────────────────────────────────────────
    if prediction[0] == 1:
        verdict_bg    = "rgba(239,68,68,0.08)"
        verdict_bdr   = "rgba(239,68,68,0.25)"
        verdict_glow  = "rgba(239,68,68,0.18)"
        verdict_color = "#fca5a5"
        verdict_icon  = "⚠️"
        verdict_title = "Fraudulent Transaction Detected"
        verdict_desc  = "This transaction has been flagged as potentially fraudulent. Immediate review is recommended."
    else:
        verdict_bg    = "rgba(16,185,129,0.08)"
        verdict_bdr   = "rgba(16,185,129,0.25)"
        verdict_glow  = "rgba(16,185,129,0.14)"
        verdict_color = "#6ee7b7"
        verdict_icon  = "✅"
        verdict_title = "Transaction Cleared"
        verdict_desc  = "This transaction appears legitimate. No suspicious activity was detected by the model."

    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@300;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>*{{margin:0;padding:0;box-sizing:border-box;}}</style>
</head>
<body style="background:transparent;padding:0 3rem 40px;">
<div style="padding:2rem 2.2rem;background:{verdict_bg};backdrop-filter:blur(20px);border:1px solid {verdict_bdr};border-radius:16px;box-shadow:0 4px 32px {verdict_glow},inset 0 1px 0 rgba(255,255,255,0.05);text-align:center;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-60px;left:50%;transform:translateX(-50%);width:220px;height:220px;border-radius:50%;background:radial-gradient(circle,{verdict_glow},transparent 65%);pointer-events:none;"></div>
  <div style="font-size:2.2rem;margin-bottom:0.75rem;">{verdict_icon}</div>
  <div style="font-family:'Fraunces',serif;font-size:1.65rem;font-weight:700;color:{verdict_color};margin-bottom:0.55rem;">{verdict_title}</div>
  <div style="font-size:0.88rem;color:#8a9ab8;max-width:440px;margin:0 auto;line-height:1.75;">{verdict_desc}</div>
  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3a4a60;margin-top:1.1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);letter-spacing:0.04em;">
    🩺 This tool is for research purposes only — not a substitute for professional financial review.
  </div>
</div>
</body></html>""", height=240, scrolling=False)