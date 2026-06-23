
import numpy as np
import pickle
import json
import tensorflow as tf
from datetime import datetime

print("=" * 50)
print("🚀 RUNNING FULL PIPELINE")
print("=" * 50)

# Step 1 — Load all data
print("\n[1/5] Loading data...")
data_pca     = np.load("data/data_pca.npy")
fused_scores = np.load("data/fused_scores.npy")
anomaly_mask = np.load("data/anomaly_mask.npy")
print(f"✅ Data loaded! Pixels: {len(data_pca)}")

# Step 2 — Load models
print("\n[2/5] Loading models...")
ae_model = tf.keras.models.load_model("model/autoencoder.h5",compile=False)
with open("model/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
print("✅ Models loaded!")

# Step 3 — Run autoencoder
print("\n[3/5] Running autoencoder...")
reconstructed = ae_model.predict(data_pca, verbose=0)
ae_scores     = np.mean(
    np.power(data_pca - reconstructed, 2), axis=1
)
print(f"✅ AE scores computed!")

# Step 4 — Load report
print("\n[4/5] Loading report...")
with open("reports/anomaly_report.json") as f:
    report = json.load(f)
print("✅ Report loaded!")

# Step 5 — Summary
print("\n[5/5] Pipeline complete!")
print("=" * 50)
print(f"Total pixels:    {len(fused_scores):,}")
print(f"Anomalies found: {int(anomaly_mask.sum()):,}")
print(f"Anomaly rate:    {anomaly_mask.mean()*100:.2f}%")
print(f"Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)
print("✅ PIPELINE COMPLETE!")
