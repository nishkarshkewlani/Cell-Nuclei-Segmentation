# Cell Nuclei Instance Segmentation: Mask R-CNN vs. StarDist vs. Cellpose

A comparative deep learning evaluation framework designed to detect, separate, and count individual and touching cell nuclei in biological microscopy images.

---

## 🔬 1. Project Overview & Scope: "What Will Your Project Do?"

In high-throughput biological microscopy (fluorescence, confocal, brightfield, and H&E histopathology), cell nuclei frequently clump, press together, and overlap due to high cellular density and 3D tissue sectioning. 

Standard semantic segmentation architectures (like standard U-Net or FCN) produce single contiguous binary blobs where touching nuclei are merged together. This makes automated cell counting, morphological profiling, and spatial transcriptomics impossible.

This project implements an end-to-end comparative instance-segmentation system that:
1. **Detects every individual nucleus**, even when cells are densely packed and touch across shared borders.
2. **Predicts distinct pixel-accurate instance masks** enclosing each nucleus with zero border leakage.
3. **Assigns a unique integer label ID** to every detected nucleus.
4. **Calculates the precise cell count** and estimates count error relative to ground truth.
5. **Benchmarks three leading instance-segmentation paradigms**:
   - **Mask R-CNN**: General-purpose proposal-based two-stage detector.
   - **StarDist**: Specialized star-convex radial polygon regressor.
   - **Cellpose**: Specialized topological gradient vector flow field tracker.
6. **Quantifies performance** across biological metrics:
   - Intersection over Union (IoU)
   - Precision, Recall, and $F_1$-score ($\text{IoU} \ge 0.5$ and $\text{IoU} \ge 0.75$)
   - Mean Average Precision ($\text{mAP@0.5:0.95}$)
   - **Aggregated Jaccard Index (AJI)** (the biomedical gold standard penalizing over/under-segmentation)
   - **Panoptic Quality (PQ)** decomposing into Segmentation Quality (SQ) and Recognition Quality (RQ)
   - Inference latency (ms) and processing throughput (FPS).

---

## 🧠 2. Architectural Comparison: How Each Model Works

| Architecture | Paradigm | Representation | Handling of Touching Nuclei | Primary Trade-Off |
| :--- | :--- | :--- | :--- | :--- |
| **Mask R-CNN** *(He et al., ICCV 2017)* | Two-Stage Proposal Detector (RPN + RoIAlign) | Axis-aligned bounding box + $28 \times 28$ binary mask | **Struggles on dense clumps**: Rectangular boxes overlap; NMS suppresses valid adjacent nuclei or merges them into one box. | Fast on sparse objects; struggles when objects lack clear rectangular separation. |
| **StarDist** *(Schmidt et al., MICCAI 2018)* | Star-Convex Radial Distance Regression | Centroid probability $P(x,y)$ + $N$ radial distances $r_i$ | **Excellent on convex/round nuclei**: Predicts rays pointing outward from center; polygon NMS prevents border merges. | Highly accurate for nuclei; cannot represent non-star-convex or convoluted morphologies. |
| **Cellpose** *(Stringer et al., Nature Methods 2021)* | Spatial Vector Flow Fields & Diffusion Dynamics | Horizontal ($v_x$) and vertical ($v_y$) inward gradient fields | **Superior on arbitrary shapes & dense clusters**: Border vectors point in opposite directions; particles converge to distinct sinks. | Highest boundary fidelity; higher computational cost due to iterative Euler integration. |

---

## 📊 3. Quantitative Evaluation Metrics

### Aggregated Jaccard Index (AJI)
Unlike standard IoU which evaluates matched pairs independently, AJI computes global intersection and union over all ground truth and predicted nuclei, heavily penalizing under-segmentation (two cells merged) and over-segmentation (one cell split):
$$\text{AJI} = \frac{\sum_{i=1}^{N_{gt}} |G_i \cap P_{j^*(i)}|}{\sum_{i=1}^{N_{gt}} |G_i \cup P_{j^*(i)}| + \sum_{k \in U} |P_k|}$$
where $P_{j^*(i)}$ is the predicted nucleus with maximum overlap to ground truth $G_i$, and $U$ is the set of false positive unassigned predictions.

### Panoptic Quality (PQ)
Combines recognition accuracy and boundary alignment:
$$\text{PQ} = \underbrace{\frac{\sum_{(p, g) \in TP} \text{IoU}(p, g)}{|TP|}}_{\text{Segmentation Quality (SQ)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Recognition Quality (RQ)}}$$

---

## 🚀 4. Quickstart Guide

### 1. Requirements
Ensure Python 3.10+ and PyTorch are installed:
```bash
pip install -r requirements.txt
```

### 2. Run Comparative Benchmark CLI
Execute automated benchmark across multi-modal microscopy samples (fluorescence and H&E):
```bash
python benchmark.py
```
This generates:
- `benchmark_results/comparison_results.png` (Side-by-side visual panels)
- `benchmark_results/benchmark_metrics.csv` (Full quantitative summary table)

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Use the web UI to:
- Generate synthetic microscopy samples or upload your own images (PNG, TIFF, JPG).
- Adjust detection thresholds interactively.
- Inspect predicted masks, labeled nucleus counts, and metric charts.

---

## 📁 5. Directory Structure

```
cell_nuclei_segmentation/
├── app.py                      # Interactive Streamlit Web UI dashboard
├── benchmark.py                # Comparative benchmark CLI script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & architectural guide
├── benchmark_results/          # Exported visualizations and CSV tables
│   ├── comparison_results.png
│   └── benchmark_metrics.csv
└── src/
    ├── dataset.py              # Microscopy generator & touching nuclei loader
    ├── metrics.py              # IoU, AJI, PQ, mAP, and Count Error calculations
    ├── visualizer.py           # Multi-panel visualization & overlay engine
    └── models/
        ├── mask_rcnn_detector.py  # Two-stage Mask R-CNN detector
        ├── stardist_detector.py   # Star-convex polygon StarDist detector
        └── cellpose_detector.py   # Vector gradient flow Cellpose detector
```
