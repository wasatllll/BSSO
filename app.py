import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(page_title="BSSO Fracture Pattern Inverse Planning System", layout="wide")

# --- 1. Load ML Assets ---
@st.cache_resource
def load_ml_assets():
    try:
        # Ensure these files exist in the 'result' folder
        model = joblib.load("result/svm_model.pkl")
        scaler = joblib.load("result/scaler.pkl")
        le = joblib.load("result/label_encoder.pkl")
        return model, scaler, le
    except Exception as e:
        st.error(f"Failed to load model files. Please check the 'result' folder. Error: {e}")
        return None, None, None

model, scaler, le = load_ml_assets()

# --- 2. Interface Title & Description ---
st.title("Bilateral Sagittal Split Osteotomy (BSSO) Lingual Fracture Pattern Prediction")
st.markdown("""
This system calculates high-probability parameter intervals, means, and standard deviations to assist in surgical planning by analyzing the influence of surgical variables on fracture patterns.
""")

# --- 3. Sidebar Settings ---
st.sidebar.header("⚙️ Calculation Settings")
grid_res = st.sidebar.select_slider("Grid Density (Total Points)", options=[50, 100, 150, 200], value=100)
prob_threshold = st.sidebar.slider("Probability Threshold", 0.0, 1.0, 0.6, 0.05)

st.divider()

# --- 4. Input Variables ---
col_in1, col_in2 = st.columns(2)

with col_in1:
    st.subheader("📌 Fixed Anatomical Variables (Constants)")
    pmbt = st.number_input("PMBT (Mandibular Ramus Thickness 1)", value=3.28, format="%.2f")
    mrt = st.number_input("MRT (Mandibular Ramus Thickness 2)", value=6.41, format="%.2f")

with col_in2:
    st.subheader("🔍 Search Surgical Variables (Range)")
    # Variable names are kept exactly as in the training set for consistency
    depth_range = st.slider("Depth  of A (Medial Horizontal Cut Depth) Range", 0.0, 5.0, (0.5, 3.0), step=0.01)
    llbce_range = st.slider("LLBCE (Lateral Bone Cut Position) Range", -5.0, 8.0, (-4.0, 6.0), step=0.01)

# --- 5. Core Calculation & Results ---
if st.button("🚀 Run Inverse Prediction & Statistical Analysis", use_container_width=True):
    if model:
        with st.spinner("Generating grid data and performing statistical analysis..."):
            # 1. Generate Meshgrid
            x_vals = np.linspace(depth_range[0], depth_range[1], grid_res)
            y_vals = np.linspace(llbce_range[0], llbce_range[1], grid_res)
            X, Y = np.meshgrid(x_vals, y_vals)
            total_points = X.size

            # 2. Construct Feature Matrix (Aligned with training order and double-space naming)
            # Order: LLBCE, PMBT, MRT, Depth  of A
            grid_flat = np.c_[
                Y.ravel(),                   # LLBCE
                np.full(total_points, pmbt), # PMBT
                np.full(total_points, mrt),  # MRT
                X.ravel()                    # Depth  of A
            ]
            
            exact_features = ['LLBCE', 'PMBT', 'MRT', 'Depth  of A']
            input_df = pd.DataFrame(grid_flat, columns=exact_features)

            # 3. Prediction
            try:
                grid_scaled = scaler.transform(input_df)
                probs = model.predict_proba(grid_scaled)
            except ValueError as ve:
                st.error(f"Feature name mismatch! Error detail: {ve}")
                st.stop()

            # 4. Display Results
            st.success(f"Calculation Complete. Total Grid Points: {total_points}")
            classes = le.classes_
            cols = st.columns(len(classes))

            for i, class_label in enumerate(classes):
                with cols[i]:
                    st.markdown(f"### 🚩 TYPE {class_label}")
                    
                    # Extract probability matrix for this class
                    Z = probs[:, i].reshape(X.shape)
                    
                    # Probability Heatmap
                    fig, ax = plt.subplots(figsize=(5, 4))
                    cp = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
                    plt.colorbar(cp)
                    ax.set_xlabel('Depth  of A')
                    ax.set_ylabel('LLBCE')
                    ax.set_title(f"Prob of Type {class_label}")
                    st.pyplot(fig)
                    
                    # --- Statistical Logic ---
                    mask = Z >= prob_threshold
                    high_prob_count = np.sum(mask)
                    percentage = (high_prob_count / total_points) * 100
                    
                    if high_prob_count > 0:
                        valid_depth = X[mask]
                        valid_llbce = Y[mask]
                        
                        # Mean and Standard Deviation
                        d_mean, d_std = valid_depth.mean(), valid_depth.std()
                        l_mean, l_std = valid_llbce.mean(), valid_llbce.std()
                        
                        # Statistical Summary
                        st.write(f"**High Prob. Points:** {high_prob_count} / {total_points} ({percentage:.2f}%)")
                        
                        st.markdown("**Parameter Range:**")
                        st.write(f"- Depth  of A: `[{valid_depth.min():.3f}, {valid_depth.max():.3f}]`")
                        st.write(f"- LLBCE: `[{valid_llbce.min():.3f}, {valid_llbce.max():.3f}]`")
                        
                        st.markdown("**Mean ± SD:**")
                        st.info(f"Depth  of A: **{d_mean:.3f} ± {d_std:.3f}**")
                        st.info(f"LLBCE: **{l_mean:.3f} ± {l_std:.3f}**")
                    else:
                        st.warning(f"No points found exceeding {prob_threshold} for TYPE {class_label}")

st.divider()
st.caption("Disclaimer: Based on SVM ML Model. Mean ± SD represents the central tendency of parameters within the high-probability region.")