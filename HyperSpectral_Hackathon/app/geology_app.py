
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle, json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="⛏️ Geology Anomaly Detection",
    page_icon="⛏️",
    layout="wide"
)

# ---- Load Models ----
@st.cache_resource
def load_all():
    ae        = tf.keras.models.load_model("model/autoencoder_geo.h5",compile=False)
    with open("model/xgboost_geo.pkl", "rb") as f:
        xgb   = pickle.load(f)
    threshold = float(np.load("model/threshold_geo.npy")[0])
    mse       = np.load("model/mse_scores_geo.npy")
    with open("reports/geology_report.json") as f:
        report = json.load(f)
    return ae, xgb, threshold, mse, report

ae_model, xgb_model, threshold, mse_scores, report = load_all()
anomaly_mask = (mse_scores > threshold).astype(int)

# ---- Mapper ----
def geology_mapper(score, thresh):
    ratio = score / thresh
    if   ratio < 0.5: return "⚪ Background",         "LOW",      "green"
    elif ratio < 1.0: return "🟡 Kaolinite Zone",     "LOW",      "yellow"
    elif ratio < 1.5: return "🟠 Alunite Zone",       "MEDIUM",   "orange"
    elif ratio < 2.5: return "🔴 Buddingtonite Zone", "HIGH",     "red"
    else:             return "💎 Calcite/Pyrope Zone", "CRITICAL", "purple"

# ===== HEADER =====
st.title("⛏️ Geological Surveying — Hyperspectral Anomaly Detection")
st.markdown("""
**Dataset:** Cuprite Nevada &nbsp;|&nbsp;
**Model:** Dense Autoencoder + XGBoost &nbsp;|&nbsp;
**Team:** INFINITY\_CREW
""")
st.divider()

# ===== METRICS =====
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔬 Pixels Analyzed",  f"{len(mse_scores):,}")
c2.metric("🚨 Anomalies Found",  f"{int(anomaly_mask.sum()):,}")
c3.metric("📊 Anomaly Rate",     f"{anomaly_mask.mean()*100:.2f}%")
c4.metric("📏 Threshold",        f"{threshold:.5f}")

st.divider()

# ===== HEATMAP =====
st.subheader("🗺️ Mineral Anomaly Heatmap")
n_show = st.slider("Pixels to display", 500, 5000, 1000, step=500)
scores_d = mse_scores[:n_show]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1 — MSE curve
axes[0].plot(scores_d, color="steelblue", linewidth=0.7, alpha=0.8)
axes[0].axhline(threshold, color="red", linestyle="--", linewidth=2)
axes[0].fill_between(range(len(scores_d)), scores_d, threshold,
                     where=(scores_d > threshold),
                     color="red", alpha=0.35, label="Anomaly")
axes[0].set_title("Reconstruction Error (MSE)")
axes[0].set_xlabel("Pixel Index")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2 — Severity colors
color_map = {"green":0, "yellow":1, "orange":2, "red":3, "purple":4}
bar_cols  = [geology_mapper(s, threshold)[2] for s in scores_d]
bar_vals  = [color_map[c] for c in bar_cols]
axes[1].bar(range(len(scores_d)), bar_vals, color=bar_cols, width=1.0)
axes[1].set_title("Mineral Zone Severity Map")
axes[1].set_yticks([0,1,2,3,4])
axes[1].set_yticklabels(["BG","Kaolin","Alunite","Budding","Calcite"])

# Panel 3 — Severity pie
sev_counts = {"LOW":0, "MEDIUM":0, "HIGH":0, "CRITICAL":0}
for s in scores_d:
    _, sev, _ = geology_mapper(s, threshold)
    sev_counts[sev] += 1
pie_labels = [k for k,v in sev_counts.items() if v > 0]
pie_vals   = [v for v in sev_counts.values()   if v > 0]
pie_colors = {"LOW":"#2ecc71","MEDIUM":"#f39c12","HIGH":"#e74c3c","CRITICAL":"#8e44ad"}
axes[2].pie(pie_vals,
            labels=pie_labels,
            colors=[pie_colors[l] for l in pie_labels],
            autopct="%1.1f%%",
            startangle=90)
axes[2].set_title("Severity Distribution")

plt.tight_layout()
st.pyplot(fig)
st.divider()

# ===== TOP ANOMALIES TABLE =====
st.subheader("🪨 Top Detected Mineral Zones")
top_idx = np.argsort(mse_scores)[::-1][:15]

cols = st.columns([1,2,3,2,3])
for h, t in zip(cols, ["Pixel","MSE Score","Mineral Zone","Severity","Action"]):
    h.markdown(f"**{t}**")

for idx in top_idx:
    zone, sev, _ = geology_mapper(mse_scores[idx], threshold)
    action_map = {
        "LOW":      "Monitor",
        "MEDIUM":   "⚠️ Investigate",
        "HIGH":     "🚨 Priority Survey",
        "CRITICAL": "💎 Urgent Extraction"
    }
    c = st.columns([1,2,3,2,3])
    c[0].write(str(int(idx)))
    c[1].write(f"{mse_scores[idx]:.5f}")
    c[2].write(zone)
    c[3].write(sev)
    c[4].write(action_map.get(sev, ""))

st.divider()

# ===== SHAP PLOT =====
st.subheader("🧠 SHAP Explainability — Why Minerals are Detected")
if os.path.exists("reports/shap_geology.png"):
    st.image("reports/shap_geology.png", use_column_width=True)
else:
    st.warning("SHAP plot not found — run Srija notebook first.")

st.divider()

# ===== JSON REPORT =====
st.subheader("📋 Full Survey Report")
if st.checkbox("Show JSON Report"):
    st.json(report)

# ===== FOOTER =====
st.success("✅ Geological Survey Complete! Flagged zones prioritized for ground-truth verification.")
st.markdown("---")
st.markdown("**INFINITY_CREW** | Hyperspectral Anomaly Detection | Geological Domain ⛏️")
