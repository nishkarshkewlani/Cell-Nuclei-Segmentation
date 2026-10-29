"""
High-Resolution Visualization Module for Comparative Nuclei Instance Segmentation.
Plots side-by-side multi-model comparisons:
- Raw Image
- Ground Truth Masks
- Mask R-CNN Predictions
- StarDist Predictions
- Cellpose Predictions
- Performance Comparison Summary Table / Charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from typing import Dict, Any, List


def create_label_colormap(num_labels: int, seed: int = 101) -> mcolors.ListedColormap:
    """Creates a distinct randomized colormap with background at index 0 as black."""
    rng = np.random.default_rng(seed)
    colors = [(0, 0, 0)]  # Background is black
    for _ in range(max(num_labels, 1)):
        # High saturation / bright colors
        c = rng.uniform(0.25, 1.0, size=3)
        colors.append(tuple(c))
    return mcolors.ListedColormap(colors)


def render_instances_overlay(
    image: np.ndarray,
    instance_mask: np.ndarray,
    alpha: float = 0.45,
    draw_contours: bool = True,
    draw_ids: bool = True,
) -> np.ndarray:
    """
    Renders colored instance masks and boundary contours overlaid on the original image.
    """
    h, w = instance_mask.shape
    if len(image.shape) == 2:
        base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = base_rgb.copy().astype(np.float32)
    unique_ids = [i for i in np.unique(instance_mask) if i != 0]

    rng = np.random.default_rng(42)
    id_color_map = {}
    for uid in unique_ids:
        color = rng.integers(70, 255, size=3, dtype=np.uint8)
        id_color_map[uid] = color

    colored_mask = np.zeros_like(base_rgb, dtype=np.float32)
    for uid in unique_ids:
        colored_mask[instance_mask == uid] = id_color_map[uid]

    fg = instance_mask > 0
    overlay[fg] = (1.0 - alpha) * base_rgb[fg] + alpha * colored_mask[fg]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if draw_contours:
        for uid in unique_ids:
            bin_m = (instance_mask == uid).astype(np.uint8)
            contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

            if draw_ids:
                # Find centroid
                moments = cv2.moments(bin_m)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.putText(
                        overlay,
                        str(uid),
                        (cx - 5, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

    return overlay


def plot_side_by_side_comparison(
    raw_image: np.ndarray,
    gt_mask: np.ndarray,
    model_results: List[Dict[str, Any]],
    output_path: str = "comparison_results.png",
) -> str:
    """
    Renders and saves a 2x3 panel comparison figure of Ground Truth and all three models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    fig.patch.set_facecolor("#111827")  # Slate dark background

    # 1. Raw Image
    img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Input Microscopy Image\n(Fluorescence / DAPI)", color="white", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Ground Truth
    gt_overlay = render_instances_overlay(raw_image, gt_mask)
    gt_count = len([i for i in np.unique(gt_mask) if i != 0])
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title(f"Ground Truth Annotations\n(Total Nuclei: {gt_count})", color="#10B981", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    # Positions for 3 models in subplots
    positions = [(0, 2), (1, 0), (1, 1)]

    for idx, res in enumerate(model_results):
        r, c = positions[idx]
        pred_mask = res["instance_mask"]
        model_name = res["model_name"]
        pred_count = res.get("num_nuclei", len(np.unique(pred_mask)) - 1)
        latency = res.get("latency_ms", 0.0)
        f1 = res.get("metrics", {}).get("f1_at_50", 0.0)
        aji = res.get("metrics", {}).get("aji", 0.0)

        overlay = render_instances_overlay(raw_image, pred_mask)
        axes[r, c].imshow(overlay)
        axes[r, c].set_title(
            f"{model_name} Prediction\nCount: {pred_count} | F1: {f1:.3f} | AJI: {aji:.3f}\nLatency: {latency:.1f}ms",
            color="#38BDF8",
            fontsize=12,
            fontweight="bold",
        )
        axes[r, c].axis("off")

    # 6th panel: Metric Comparison Bar Chart
    ax_table = axes[1, 2]
    ax_table.set_facecolor("#1F2937")

    model_names = [res["model_name"] for res in model_results]
    f1_scores = [res.get("metrics", {}).get("f1_at_50", 0.0) for res in model_results]
    aji_scores = [res.get("metrics", {}).get("aji", 0.0) for res in model_results]
    pq_scores = [res.get("metrics", {}).get("panoptic_quality", 0.0) for res in model_results]

    x = np.arange(len(model_names))
    width = 0.25

    b1 = ax_table.bar(x - width, f1_scores, width, label="F1 @ 0.5", color="#38BDF8")
    b2 = ax_table.bar(x, aji_scores, width, label="AJI (Jaccard)", color="#10B981")
    b3 = ax_table.bar(x + width, pq_scores, width, label="Panoptic Qual (PQ)", color="#F59E0B")

    ax_table.set_ylabel("Score (0.0 - 1.0)", color="white", fontsize=11)
    ax_table.set_title("Instance Segmentation Accuracy", color="white", fontsize=13, fontweight="bold")
    ax_table.set_xticks(x)
    ax_table.set_xticklabels(model_names, color="white", fontsize=11, fontweight="bold")
    ax_table.set_ylim(0, 1.05)
    ax_table.tick_params(colors="white")
    ax_table.legend(facecolor="#374151", edgecolor="none", labelcolor="white")
    ax_table.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()

    return output_path
