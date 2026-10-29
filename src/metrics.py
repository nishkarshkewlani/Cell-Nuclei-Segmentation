"""
Evaluation Metrics for Cell Nuclei Instance Segmentation.
Includes:
- Intersection over Union (IoU)
- Precision, Recall, F1-Score (IoU @ 0.5 and @ 0.75)
- Mean Average Precision (mAP@0.5:0.95)
- Aggregated Jaccard Index (AJI) - Biological Gold Standard
- Panoptic Quality (PQ = SQ * RQ)
- Nucleus Count Error
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any, List, Tuple


def compute_iou_matrix(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Computes pairwise IoU between all ground truth and predicted nucleus instances.

    Args:
        gt_mask: 2D array of ground truth instance IDs (0 is background).
        pred_mask: 2D array of predicted instance IDs (0 is background).

    Returns:
        iou_matrix: 2D array of shape [n_gt, n_pred].
        gt_ids: List of unique ground truth IDs.
        pred_ids: List of unique predicted IDs.
    """
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred)), gt_ids, pred_ids

    # Fast overlapping pixel tally
    # Encode pair IDs into single int64
    max_id = max(max(gt_ids), max(pred_ids)) + 1
    encoded = gt_mask.astype(np.int64) * max_id + pred_mask.astype(np.int64)

    # Filter out background pairs
    valid = (gt_mask > 0) & (pred_mask > 0)
    overlap_pairs, counts = np.unique(encoded[valid], return_counts=True)

    gt_areas = {gid: np.sum(gt_mask == gid) for gid in gt_ids}
    pred_areas = {pid: np.sum(pred_mask == pid) for pid in pred_ids}

    gt_idx_map = {gid: idx for idx, gid in enumerate(gt_ids)}
    pred_idx_map = {pid: idx for idx, pid in enumerate(pred_ids)}

    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

    for pair, inter in zip(overlap_pairs, counts):
        gid = int(pair // max_id)
        pid = int(pair % max_id)
        if gid in gt_idx_map and pid in pred_idx_map:
            union = gt_areas[gid] + pred_areas[pid] - inter
            iou_matrix[gt_idx_map[gid], pred_idx_map[pid]] = inter / union if union > 0 else 0.0

    return iou_matrix, gt_ids, pred_ids


def evaluate_detection_at_threshold(iou_matrix: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Matches GT to predicted instances using optimal bipartite matching (Hungarian algorithm).
    """
    n_gt, n_pred = iou_matrix.shape
    if n_gt == 0 and n_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_gt == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 0, "fp": n_pred, "fn": 0}
    if n_pred == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": n_gt}

    # Solve maximum weight bipartite matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    tp = 0
    matched_ious = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            tp += 1
            matched_ious.append(iou_matrix[r, c])

    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_iou": mean_iou,
    }


def compute_map(iou_matrix: np.ndarray) -> float:
    """
    Computes mean Average Precision (mAP) averaged over IoU thresholds 0.50 to 0.95 with step 0.05.
    """
    thresholds = np.linspace(0.50, 0.95, 10)
    aps = []
    for t in thresholds:
        res = evaluate_detection_at_threshold(iou_matrix, threshold=t)
        # Average precision under balanced single class approximates F1/Precision at threshold
        aps.append(res["precision"])
    return float(np.mean(aps))


def compute_aggregated_jaccard_index(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Computes Aggregated Jaccard Index (AJI) commonly used in pathology (e.g. MoNuSeg).
    Penalizes both over-segmentation and under-segmentation.
    """
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]

    if not gt_ids and not pred_ids:
        return 1.0
    if not gt_ids or not pred_ids:
        return 0.0

    iou_matrix, _, _ = compute_iou_matrix(gt_mask, pred_mask)

    overall_intersection = 0
    overall_union = 0
    used_pred_indices = set()

    gt_areas = [np.sum(gt_mask == gid) for gid in gt_ids]
    pred_areas = [np.sum(pred_mask == pid) for pid in pred_ids]

    for i in range(len(gt_ids)):
        if iou_matrix.shape[1] > 0:
            max_iou_idx = int(np.argmax(iou_matrix[i, :]))
            best_iou = iou_matrix[i, max_iou_idx]
        else:
            best_iou = 0.0
            max_iou_idx = -1

        if best_iou > 0 and max_iou_idx != -1:
            # Calculate true intersection and union between GT_i and Pred_best
            g_bin = gt_mask == gt_ids[i]
            p_bin = pred_mask == pred_ids[max_iou_idx]
            inter = np.sum(g_bin & p_bin)
            union = np.sum(g_bin | p_bin)
            overall_intersection += inter
            overall_union += union
            used_pred_indices.add(max_iou_idx)
        else:
            overall_union += gt_areas[i]

    # Add unmatched false positive predicted areas to union
    for j in range(len(pred_ids)):
        if j not in used_pred_indices:
            overall_union += pred_areas[j]

    if overall_union == 0:
        return 0.0
    return float(overall_intersection / overall_union)


def compute_panoptic_quality(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).
    """
    iou_matrix, gt_ids, pred_ids = compute_iou_matrix(gt_mask, pred_mask)
    res = evaluate_detection_at_threshold(iou_matrix, threshold=0.5)

    tp, fp, fn = res["tp"], res["fp"], res["fn"]
    if tp == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0}

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    tp_ious = [iou_matrix[r, c] for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= 0.5]

    sq = float(np.mean(tp_ious))
    rq = float(tp / (tp + 0.5 * fp + 0.5 * fn))
    pq = float(sq * rq)

    return {"pq": pq, "sq": sq, "rq": rq}


def full_evaluation_summary(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    latency_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Consolidates all instance segmentation metrics into a single dictionary.
    """
    gt_count = len([i for i in np.unique(gt_mask) if i != 0])
    pred_count = len([i for i in np.unique(pred_mask) if i != 0])
    count_error = abs(pred_count - gt_count)
    count_pct_error = (count_error / gt_count * 100.0) if gt_count > 0 else 0.0

    iou_matrix, _, _ = compute_iou_matrix(gt_mask, pred_mask)
    det_50 = evaluate_detection_at_threshold(iou_matrix, threshold=0.5)
    det_75 = evaluate_detection_at_threshold(iou_matrix, threshold=0.75)
    m_ap = compute_map(iou_matrix)
    aji = compute_aggregated_jaccard_index(gt_mask, pred_mask)
    pq_dict = compute_panoptic_quality(gt_mask, pred_mask)

    return {
        "gt_nuclei": gt_count,
        "pred_nuclei": pred_count,
        "count_error": count_error,
        "count_pct_error": round(count_pct_error, 2),
        "f1_at_50": round(det_50["f1"], 4),
        "precision_at_50": round(det_50["precision"], 4),
        "recall_at_50": round(det_50["recall"], 4),
        "f1_at_75": round(det_75["f1"], 4),
        "mAP_50_95": round(m_ap, 4),
        "aji": round(aji, 4),
        "panoptic_quality": round(pq_dict["pq"], 4),
        "segmentation_quality": round(pq_dict["sq"], 4),
        "recognition_quality": round(pq_dict["rq"], 4),
        "latency_ms": round(latency_ms, 2),
        "fps": round(1000.0 / latency_ms, 2) if latency_ms > 0 else 0.0,
    }
