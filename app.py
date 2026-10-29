"""
Streamlit Web Dashboard: Cell Nuclei Instance Segmentation Benchmark.
Interactive comparison between Mask R-CNN, StarDist, and Cellpose.
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.dataset import NucleiDatasetGenerator
from src.metrics import full_evaluation_summary
from src.visualizer import render_instances_overlay
from src.models.mask_rcnn_detector import MaskRCNNDetector
from src.models.stardist_detector import StarDistDetector
from src.models.cellpose_detector import CellposeDetector

st.set_page_config(
    page_title="Cell Nuclei Instance Segmentation",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Cell Nuclei Instance Segmentation Benchmark")
st.markdown(
    """
    **Comparative Study: Mask R-CNN vs. StarDist vs. Cellpose**  
    Evaluate how deep learning instance-segmentation architectures detect, separate, and count overlapping or touching cell nuclei in biological microscopy images.
    """
)

# Sidebar settings
st.sidebar.header("⚙️ Configuration & Inputs")
input_source = st.sidebar.radio(
    "Select Image Input Mode:",
    ["Generate Synthetic Microscopy Sample", "Upload Custom Microscopy Image"],
)

stain_type = st.sidebar.selectbox(
    "Microscopy Modality / Stain:",
    ["fluorescence", "he"],
    format_func=lambda x: "Fluorescence (DAPI / Hoechst)" if x == "fluorescence" else "H&E (Histopathology)",
)

st.sidebar.subheader("Model Hyper-parameters")
mask_rcnn_conf = st.sidebar.slider("Mask R-CNN Score Threshold", 0.1, 0.9, 0.45, 0.05)
stardist_prob = st.sidebar.slider("StarDist Centroid Prob Threshold", 0.1, 0.9, 0.50, 0.05)
cellpose_thresh = st.sidebar.slider("Cellpose Cell Prob Threshold", 0.1, 0.9, 0.45, 0.05)

# Session state initialization for sample
if "sample" not in st.session_state:
    gen = NucleiDatasetGenerator(seed=42)
    st.session_state["sample"] = gen.generate_sample(
        min_nuclei=30, max_nuclei=50, stain_type=stain_type
    )

if input_source == "Generate Synthetic Microscopy Sample":
    if st.sidebar.button("🎲 Generate New Sample"):
        gen = NucleiDatasetGenerator(seed=int(np.random.randint(0, 10000)))
        st.session_state["sample"] = gen.generate_sample(
            min_nuclei=30, max_nuclei=55, stain_type=stain_type
        )
    sample = st.session_state["sample"]
    raw_img = sample["image"]
    gt_mask = sample["instance_mask"]
    gt_count = sample["num_nuclei"]
else:
    uploaded = st.sidebar.file_uploader("Upload Image (PNG, JPG, TIFF):", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gt_mask = None
        gt_count = None
    else:
        sample = st.session_state["sample"]
        raw_img = sample["image"]
        gt_mask = sample["instance_mask"]
        gt_count = sample["num_nuclei"]

# Detectors
mask_rcnn = MaskRCNNDetector(score_threshold=mask_rcnn_conf)
stardist = StarDistDetector(prob_threshold=stardist_prob)
cellpose = CellposeDetector(cellprob_threshold=cellpose_thresh)

# Execute inference
with st.spinner("Executing instance segmentation across all 3 architectures..."):
    res_mrcnn = mask_rcnn.predict(raw_img)
    res_stardist = stardist.predict(raw_img)
    res_cellpose = cellpose.predict(raw_img)

# Main display columns
st.subheader("🖼️ Comparative Segmentation Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original Microscopy Image**")
    st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    if gt_mask is not None:
        st.markdown(f"**Ground Truth Reference (Count: {gt_count})**")
        st.image(render_instances_overlay(raw_img, gt_mask), use_column_width=True)
    else:
        st.info("Upload mode active: ground truth mask not available.")

# Model rows
st.markdown("---")
mcol1, mcol2, mcol3 = st.columns(3)

models_data = [
    ("Mask R-CNN (Two-stage proposals)", res_mrcnn, mcol1),
    ("StarDist (Star-convex polygons)", res_stardist, mcol2),
    ("Cellpose (Topological flow fields)", res_cellpose, mcol3),
]

metrics_summary = []

for title, res, col in models_data:
    with col:
        st.markdown(f"### {title}")
        n_count = res["num_nuclei"]
        lat = res["latency_ms"]
        st.markdown(f"**Detected Nuclei:** `{n_count}` | **Latency:** `{lat:.1f} ms`")

        overlay = render_instances_overlay(raw_img, res["instance_mask"])
        st.image(overlay, use_column_width=True)

        if gt_mask is not None:
            m = full_evaluation_summary(gt_mask, res["instance_mask"], latency_ms=lat)
            metrics_summary.append({
                "Model": res["model_name"],
                "Count": n_count,
                "Count Error": m["count_error"],
                "F1 @ 0.5": m["f1_at_50"],
                "AJI (Jaccard)": m["aji"],
                "Panoptic Qual": m["panoptic_quality"],
                "Latency (ms)": m["latency_ms"],
            })

# Quantitative comparison table
if metrics_summary:
    st.markdown("---")
    st.subheader("📊 Quantitative Performance Metrics")
    df_metrics = pd.DataFrame(metrics_summary)
    st.dataframe(df_metrics.set_index("Model"), use_container_width=True)

    # Bar chart
    st.subheader("📈 Metric Comparison Chart")
    chart_data = df_metrics.melt(id_vars=["Model"], value_vars=["F1 @ 0.5", "AJI (Jaccard)", "Panoptic Qual"], var_name="Metric", value_name="Score")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot_df = df_metrics.set_index("Model")[["F1 @ 0.5", "AJI (Jaccard)", "Panoptic Qual"]]
    pivot_df.plot(kind="bar", ax=ax, colormap="viridis", rot=0)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)
