
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Initialize state
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = False

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="HyperSpectral Anomaly Detector",
    page_icon="🌾",
    layout="wide"
)
# ── CUSTOM CSS ──────────────────────────────
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 0.3em;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 1.2rem;
    color: #cbd5f5;
    margin-bottom: 2em;
}

/* Info Cards */
.info-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Section Heading */
.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 20px;
    color: #a5f3fc;
}

/* Success Message */
.success-box {
    background: rgba(34,197,94,0.1);
    padding: 10px;
    border-radius: 8px;
    color: #22c55e;
    font-weight: 500;
}



/* Remove white hover + background from button */
button {
    background-color: transparent !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────
st.markdown("""
<div class="main-title">
🔬 Hyperspectral Anomaly Detection
</div>

<div class="sub-title">
Hybrid Intelligence: RX Detector + Autoencoder for High-Precision Detection
</div>
<hr>
""", unsafe_allow_html=True)

#── SIDEBAR ──────────────────────────────
if st.session_state.sidebar_visible:
    with st.sidebar:
        st.markdown("## ⚙ SpectraGuard AI")

        st.markdown("Hybrid LSTM-Autoencoder & RX Detector Pipeline")

        st.markdown("### Select Application Domain")
        st.selectbox(
            "",
            ["Agriculture (Crop Health)", "Defense", "Medical"]
        )

        st.markdown("🟢 Model Status: Optimal")
        st.markdown("⚡ Hardware: GPU Accelerated")

# ── TEAM INFO ────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.markdown(
    "<h6 style='color:white;'>👥 Team<br><b>Infinity Crew</b></h6>",
    unsafe_allow_html=True
)

col2.markdown(
    "<h6 style='color:white;'>🪪 ID<br><b>FXH26-E-AIML-039</b></h6>",
    unsafe_allow_html=True
)

col3.markdown(
    "<h6 style='color:white;'>🏫 Institute<br><b>Sir MVIT</b></h6>",
    unsafe_allow_html=True
)
st.markdown("---")

# ── DOMAIN SELECTOR ──────────────────────────
st.markdown("### 🌍 Select Domain")
domain = st.selectbox(
    "Choose application domain:",
    [
        "🌾 Agriculture (Indian Pines)",
        "🛡️ Defence (Camouflage Detection)",
        "⛏️ Geology (Mineral Mapping)"
    ]
)

# ── LOAD MODELS ──────────────────────────────
@st.cache_resource
def load_models():
    ae  = tf.keras.models.load_model("model/autoencoder.h5",compile=False)
    ae.compile(optimizer='adam',loss='mse')
    with open("model/xgboost_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    return ae, xgb

@st.cache_data
def load_precomputed():
    data_pca     = np.load("data/data_pca.npy")
    fused_scores = np.load("data/fused_scores.npy")
    anomaly_mask = np.load("data/anomaly_mask.npy")
    rx_norm      = np.load("data/rx_norm.npy")
    ae_norm      = np.load("data/ae_norm.npy")
    labels       = np.load("data/labels.npy")
    with open("reports/anomaly_report.json") as f:
        report = json.load(f)
    return data_pca, fused_scores, anomaly_mask, rx_norm, ae_norm, labels, report

# ── AGRICULTURE MAPPER ───────────────────────
AGRICULTURE_MAPPER = {
    0:"Background", 1:"🌾 Alfalfa", 2:"🌽 Corn No-Till",
    3:"🌽 Corn Min-Till", 4:"🌽 Corn", 5:"🌿 Grass Pasture",
    6:"🌿 Grass Trees", 7:"🌿 Grass Mowed", 8:"🌾 Hay",
    9:"🌱 Oats", 10:"🫘 Soybean No-Till", 11:"🫘 Soybean Min-Till",
    12:"🫘 Soybean Clean", 13:"🌾 Wheat", 14:"🌲 Woods",
    15:"🏗️ Buildings", 16:"💧 Towers"
}

DEFENCE_MAPPER = {
    0:"Background", 1:"🛣️ Asphalt Clear", 2:"🚗 Vehicle Detected ⚠️",
    3:"🏗️ Rooftop Normal", 4:"🌳 Tree Cover Clear",
    5:"🎯 Camouflaged Target 🚨", 6:"💧 Water Body",
    7:"🏭 Metal Structure ⚠️", 8:"🛣️ Road Clear"
}

GEOLOGY_MAPPER = {
    0:"Background", 1:"⛏️ Alunite Zone", 2:"🪨 Buddingtonite ⚠️",
    3:"💎 Calcite — High Value 🚨", 4:"🌋 Kaolinite Normal",
    5:"⚡ Muscovite Detected", 6:"🟤 Montmorillonite",
    7:"🔵 Nontronite ⚠️", 8:"🪨 Pyrope Normal"
}
#── CSS of METRIC ──────────────────────────────
st.markdown("""
<style>

/* Make ALL text white */
html, body, [class*="css"]  {
    color: white !important;
}

/* Metric labels (Detection Summary) */
[data-testid="stMetricLabel"] {
    color: white !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: white !important;
}

/* Section headers */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* Selectbox text */
.stSelectbox label {
    color: white !important;
}

/* General text */
p {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# ── MAIN APP ─────────────────────────────────
with st.spinner("Loading models and data..."):
    ae_model, xgb_model = load_models()
    data_pca, fused_scores, anomaly_mask, rx_norm, ae_norm, labels, report = load_precomputed()

st.success("✅ Models loaded successfully!")

# ── METRICS ROW ──────────────────────────────
st.markdown('<div class="section-title">📊 Detection Summary</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

total     = len(fused_scores)
anomalies = int(anomaly_mask.sum())
pct       = round(float(anomaly_mask.mean()*100), 2)

m1.metric("Total Pixels",      f"{total:,}")
m2.metric("Anomalies Found",   f"{anomalies:,}", delta="🚨 Detected")
m3.metric("Anomaly Rate",      f"{pct}%")
m4.metric("Detection Method",  "Hybrid RX+AE")

st.markdown("---")

# ── HEATMAP VISUALIZATION ────────────────────
st.markdown("### 🗺️ Anomaly Heatmaps")

fused_map   = fused_scores.reshape(145, 145)
anomaly_map = anomaly_mask.reshape(145, 145)
rgb = data_pca.reshape(145, 145, 30)[:, :, :3]
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(rgb)
axes[0].set_title("Original Image\n(False Color)", fontsize=12)
axes[0].axis("off")

im = axes[1].imshow(fused_map, cmap="RdYlGn_r")
axes[1].set_title("Fused Anomaly Heatmap\n(RX + Autoencoder)", fontsize=12)
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)

axes[2].imshow(rgb)
axes[2].imshow(anomaly_map, cmap="Reds", alpha=0.6)
axes[2].set_title("Anomaly Overlay\n(Red = Anomaly)", fontsize=12)
axes[2].axis("off")

normal_p  = mpatches.Patch(color="green", alpha=0.5, label="Normal")
anomaly_p = mpatches.Patch(color="red",   alpha=0.8, label="Anomaly")
axes[2].legend(handles=[normal_p, anomaly_p], loc="lower right")

if "Agriculture" in domain:
    title = "Agriculture — Diseased/Stressed Crop Detection"
elif "Defence" in domain:
    title = "Defence — Camouflage Target Detection"
else:
    title = "Geology — Mineral Deposit Mapping"

plt.suptitle(title, fontsize=14, fontweight="bold")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── INDIVIDUAL SCORE COMPARISON ──────────────
st.markdown("### 📈 RX vs Autoencoder Score Comparison")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

rx_map = rx_norm.reshape(145, 145)
ae_map = ae_norm.reshape(145, 145)

im1 = axes2[0].imshow(rx_map, cmap="hot")
axes2[0].set_title("RX Detector\n(Classical Statistical)", fontsize=11)
axes2[0].axis("off")
plt.colorbar(im1, ax=axes2[0], fraction=0.046)

im2 = axes2[1].imshow(ae_map, cmap="hot")
axes2[1].set_title("Autoencoder\n(Deep Learning)", fontsize=11)
axes2[1].axis("off")
plt.colorbar(im2, ax=axes2[1], fraction=0.046)

plt.suptitle("Individual Method Comparison", fontsize=13)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# ── SEVERITY BREAKDOWN ───────────────────────
st.markdown("### ⚠️ Severity Breakdown")

if "summary" in report:
    s = report["summary"]["severity_breakdown"]
    c1, c2, c3 = st.columns(3)
    c1.error(  f"🚨 Critical: {s.get('CRITICAL', 0)}")
    c2.warning(f"⚠️ Medium:   {s.get('MEDIUM',   0)}")
    c3.info(   f"🔍 Low:      {s.get('LOW',       0)}")

st.markdown("---")

# ── TOP AFFECTED CROPS ───────────────────────
st.markdown("### 🌾 Top Affected Zones")

if "top_affected_crops" in report:
    cols = st.columns(len(report["top_affected_crops"]))
    for i, crop in enumerate(report["top_affected_crops"]):
        cols[i].metric(crop["crop"], f"{crop['count']} pixels")

st.markdown("---")

# ── SHAP EXPLANATION ─────────────────────────
st.markdown("### 🔍 XAI — Why These Are Anomalies?")
st.markdown("SHAP values explain which features drove each detection:")

if os.path.exists("reports/shap_plot.png"):
    st.image("reports/shap_plot.png",
             caption="SHAP Feature Importance",
             use_column_width=True)

st.markdown("---")

# ── METHODOLOGY ──────────────────────────────
st.markdown("### ⚙️ Pipeline Methodology")

steps = {
    "Step 1 — Input":      "Hyperspectral cube loaded (145×145×200 bands)",
    "Step 2 — PCA":        "Dimensionality reduction 200 → 30 bands",
    "Step 3 — RX Detector":"Statistical background modeling (Mahalanobis distance)",
    "Step 4 — Autoencoder":"Deep learning spectral reconstruction error",
    "Step 5 — Fusion":     "Weighted score fusion (50% RX + 50% AE)",
    "Step 6 — Output":     "Anomaly heatmap overlay on original image"
}

for step, desc in steps.items():
    st.markdown(f"**{step}:** {desc}")

st.markdown("---")

# ── FULL REPORT ──────────────────────────────
st.markdown("### 📋 Full Detection Report")

with st.expander("Click to view full JSON report"):
    st.json(report)

# ── FOOTER ───────────────────────────────────
st.markdown("""
<hr>
<div style='text-align:center; color:white; font-size:12px;'>
🔬 Hybrid Hyperspectral Anomaly Detection |
Infinity_Crew | FusionX Hackathon 2026 |
Sir M Visvesvarya Institute of Technology
</div>
""", unsafe_allow_html=True)
