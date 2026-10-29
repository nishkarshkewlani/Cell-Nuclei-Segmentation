"""
Cellpose: Anatomical Instance Segmentation via Spatial Gradient Flow Fields.
Implements:
1. Simulated heat diffusion / vector flow field generation (v_x, v_y).
2. Euler integration dynamics tracking virtual particles to topological sinks.
3. Sink clustering and unique instance assignment.
4. Robust separation of clustered and touching cell nuclei.
"""

import time
import numpy as np
import cv2
from typing import Dict, Any, Tuple


class CellposeDetector:
    """
    Cellpose gradient flow-field based instance segmentation detector.
    Tracks pixels along predicted vector flow fields toward topological centers.
    """

    def __init__(
        self,
        cellprob_threshold: float = 0.45,
        flow_threshold: float = 0.4,
        n_iter: int = 120,
        step_size: float = 0.8,
    ):
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.n_iter = n_iter
        self.step_size = step_size

    def _compute_vector_flows(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes cell probability and gradient flow fields (v_y, v_x)
        representing inward directional diffusion toward cell centroids.
        """
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

        # Foreground probability
        p_thresh = np.percentile(blurred, 55)
        fg_mask = (blurred > p_thresh).astype(np.uint8)

        # Distance transform serves as the potential function V(x, y)
        dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)

        # Flow fields are the spatial gradients of the potential function
        # grad_y, grad_x = dV/dy, dV/dx
        grad_y = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)

        # Normalize flow vectors
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        flow_y = (grad_y / magnitude) * fg_mask
        flow_x = (grad_x / magnitude) * fg_mask

        cellprob = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

        return cellprob, flow_y, flow_x

    def _euler_integration(
        self,
        flow_y: np.ndarray,
        flow_x: np.ndarray,
        fg_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Simulates particle dynamics by Euler integration along flow vectors.
        p_{t+1} = p_t + dt * v(p_t)
        """
        h, w = flow_y.shape
        pos = fg_coords.astype(np.float32).copy()  # [N, 2] (y, x)

        for _ in range(self.n_iter):
            # Nearest neighbor sampling of flow fields
            iy = np.clip(np.round(pos[:, 0]).astype(np.int32), 0, h - 1)
            ix = np.clip(np.round(pos[:, 1]).astype(np.int32), 0, w - 1)

            vy = flow_y[iy, ix]
            vx = flow_x[iy, ix]

            # Update coordinates along inward gradient
            pos[:, 0] += self.step_size * vy
            pos[:, 1] += self.step_size * vx

            # Clip within boundaries
            pos[:, 0] = np.clip(pos[:, 0], 0, h - 1)
            pos[:, 1] = np.clip(pos[:, 1], 0, w - 1)

        return pos

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Performs Cellpose instance segmentation.

        Args:
            image: Microscopy image.

        Returns:
            Dictionary containing:
                - 'instance_mask': [H, W] int32 label map
                - 'num_nuclei': count of segmented instances
                - 'flow_y', 'flow_x': flow field maps
                - 'cellprob': probability map
                - 'latency_ms': inference latency
        """
        start_time = time.perf_counter()

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.dtype != np.float32:
            gray = gray.astype(np.float32) / 255.0

        # Auto-invert brightfield / H&E images
        if np.mean(gray) > 0.5:
            gray = 1.0 - gray

        h, w = gray.shape

        # 1. Flow field & Cell Probability computation
        cellprob, flow_y, flow_x = self._compute_vector_flows(gray)

        # 2. Select pixels above threshold
        mask_fg = cellprob > self.cellprob_threshold
        ys, xs = np.where(mask_fg)

        instance_mask = np.zeros((h, w), dtype=np.int32)

        if len(ys) == 0:
            return {
                "instance_mask": instance_mask,
                "num_nuclei": 0,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                "model_name": "Cellpose",
            }

        fg_coords = np.column_stack((ys, xs))

        # 3. Particle Tracking via Euler Integration
        endpoints = self._euler_integration(flow_y, flow_x, fg_coords)
        end_y = np.round(endpoints[:, 0]).astype(np.int32)
        end_x = np.round(endpoints[:, 1]).astype(np.int32)

        # 4. Sink Density Accumulation
        sink_map = np.zeros((h, w), dtype=np.int32)
        np.add.at(sink_map, (end_y, end_x), 1)

        # Smooth sink map to consolidate cluster peaks
        sink_blurred = cv2.GaussianBlur(sink_map.astype(np.float32), (7, 7), 1.5)
        local_max = cv2.dilate(sink_blurred, np.ones((9, 9), np.uint8))
        sinks = (sink_blurred == local_max) & (sink_blurred > 8.0)

        num_sinks, sink_labels = cv2.connectedComponents(sinks.astype(np.uint8))

        if num_sinks <= 1:
            # Fallback if no distinct sinks
            num_sinks, sink_labels = cv2.connectedComponents(mask_fg.astype(np.uint8))
            instance_mask = sink_labels
            return {
                "instance_mask": instance_mask,
                "num_nuclei": num_sinks - 1,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                "model_name": "Cellpose",
            }

        # 5. Map each foreground pixel to the nearest sink cluster
        sink_ys, sink_xs = np.where(sinks)
        sink_centroids = np.column_stack((sink_ys, sink_xs))

        # Associate particle endpoints with closest sink
        from scipy.spatial import cKDTree

        tree = cKDTree(sink_centroids)
        _, nearest_sink_idx = tree.query(endpoints)

        # Assign unique instance ID (1-indexed)
        assigned_ids = nearest_sink_idx + 1
        instance_mask[ys, xs] = assigned_ids

        # Post-processing: remove tiny noisy fragments (<25 pixels)
        unique_ids, counts = np.unique(instance_mask, return_counts=True)
        cleaned_mask = np.zeros_like(instance_mask)
        final_id = 1
        for uid, count in zip(unique_ids, counts):
            if uid == 0 or count < 25:
                continue
            cleaned_mask[instance_mask == uid] = final_id
            final_id += 1

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "instance_mask": cleaned_mask,
            "num_nuclei": final_id - 1,
            "flow_y": flow_y,
            "flow_x": flow_x,
            "cellprob": cellprob,
            "latency_ms": latency_ms,
            "model_name": "Cellpose",
        }
