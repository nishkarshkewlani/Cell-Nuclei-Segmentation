"""
Comparative Benchmark Pipeline for Cell Nuclei Instance Segmentation.
Runs Mask R-CNN, StarDist, and Cellpose on test images, evaluates metrics,
and produces comparative reports and visualizations.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import NucleiDatasetGenerator
from src.metrics import full_evaluation_summary
from src.visualizer import plot_side_by_side_comparison
from src.models.mask_rcnn_detector import MaskRCNNDetector
from src.models.stardist_detector import StarDistDetector
from src.models.cellpose_detector import CellposeDetector


def run_benchmark(num_test_samples: int = 5, output_dir: str = "benchmark_results"):
    """
    Executes comparative benchmark across all 3 models.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("CELL NUCLEI INSTANCE SEGMENTATION: COMPARATIVE BENCHMARK")
    print("Comparing: Mask R-CNN  vs.  StarDist  vs.  Cellpose")
    print("=" * 75)

    # Instantiate models
    print("\n[1/4] Initializing detection models...")
    models = [
        {"name": "Mask R-CNN", "detector": MaskRCNNDetector()},
        {"name": "StarDist", "detector": StarDistDetector()},
        {"name": "Cellpose", "detector": CellposeDetector()},
    ]

    generator = NucleiDatasetGenerator(image_size=(512, 512), seed=42)
    aggregated_metrics = {m["name"]: [] for m in models}

    print(f"\n[2/4] Running inference & metric evaluation over {num_test_samples} test samples...")

    sample_results_for_vis = None

    for s_idx in range(num_test_samples):
        # Generate sample with challenging touching nuclei clusters
        sample = generator.generate_sample(
            min_nuclei=30,
            max_nuclei=55,
            cluster_probability=0.75,
            stain_type="fluorescence" if s_idx % 2 == 0 else "he",
        )
        img = sample["image"]
        gt_mask = sample["instance_mask"]
        gt_count = sample["num_nuclei"]

        print(f"\n--- Test Sample #{s_idx + 1} (GT Nuclei: {gt_count}, Stain: {sample['stain_type']}) ---")

        current_sample_preds = []

        for m_info in models:
            m_name = m_info["name"]
            detector = m_info["detector"]

            pred_res = detector.predict(img)
            pred_mask = pred_res["instance_mask"]
            latency = pred_res["latency_ms"]

            metrics = full_evaluation_summary(gt_mask, pred_mask, latency_ms=latency)
            aggregated_metrics[m_name].append(metrics)

            print(
                f"  [{m_name:10s}] Found: {metrics['pred_nuclei']:2d} | "
                f"F1@0.5: {metrics['f1_at_50']:.3f} | "
                f"AJI: {metrics['aji']:.3f} | "
                f"PQ: {metrics['panoptic_quality']:.3f} | "
                f"Count Err: {metrics['count_error']:2d} | "
                f"Latency: {metrics['latency_ms']:5.1f}ms"
            )

            pred_res["metrics"] = metrics
            current_sample_preds.append(pred_res)

        if s_idx == 0:
            sample_results_for_vis = (img, gt_mask, current_sample_preds)

    # 3. Save comparative visualization
    print("\n[3/4] Generating side-by-side comparison visualization...")
    vis_img, vis_gt, vis_preds = sample_results_for_vis
    vis_path = os.path.join(output_dir, "comparison_results.png")
    plot_side_by_side_comparison(vis_img, vis_gt, vis_preds, output_path=vis_path)
    print(f"  Saved figure: {vis_path}")

    # 4. Compute Aggregate Table
    print("\n[4/4] Aggregating Benchmark Results Across All Test Images...")
    summary_rows = []
    for m_name, list_metrics in aggregated_metrics.items():
        df_m = pd.DataFrame(list_metrics)
        summary_rows.append(
            {
                "Model Architecture": m_name,
                "F1 Score (@0.5)": f"{df_m['f1_at_50'].mean():.3f} +/- {df_m['f1_at_50'].std():.3f}",
                "mAP (0.50:0.95)": f"{df_m['mAP_50_95'].mean():.3f} +/- {df_m['mAP_50_95'].std():.3f}",
                "AJI (Jaccard)": f"{df_m['aji'].mean():.3f} +/- {df_m['aji'].std():.3f}",
                "Panoptic Qual (PQ)": f"{df_m['panoptic_quality'].mean():.3f} +/- {df_m['panoptic_quality'].std():.3f}",
                "Mean Count Error": f"{df_m['count_error'].mean():.1f} ({df_m['count_pct_error'].mean():.1f}%)",
                "Mean Latency (ms)": f"{df_m['latency_ms'].mean():.1f} ms",
                "Throughput (FPS)": f"{df_m['fps'].mean():.1f} FPS",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "benchmark_metrics.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("FINAL COMPARATIVE BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)
    print(f"\nAll benchmark results exported to: {output_dir}/")

    return summary_df


if __name__ == "__main__":
    run_benchmark(num_test_samples=4)
