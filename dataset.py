"""
Synthetic and Real Microscopy Image Dataset Utilities for Cell Nuclei Segmentation.
Generates realistic multi-nuclei images with touching/clumped cells, chromatin textures,
and background noise for comparative evaluation.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Any


class NucleiDatasetGenerator:
    """
    Generates synthetic fluorescence / brightfield microscopy images with overlapping
    and touching cell nuclei, textured chromatin, and camera noise.
    """

    def __init__(self, image_size: Tuple[int, int] = (512, 512), seed: int = 42):
        self.height, self.width = image_size
        self.rng = np.random.default_rng(seed)

    def generate_sample(
        self,
        min_nuclei: int = 25,
        max_nuclei: int = 50,
        cluster_probability: float = 0.7,
        stain_type: str = "fluorescence",
    ) -> Dict[str, Any]:
        """
        Generates a synthetic microscopy image containing touching nuclei.

        Args:
            min_nuclei: Minimum number of nuclei to place.
            max_nuclei: Maximum number of nuclei to place.
            cluster_probability: Likelihood of nuclei forming touching clusters.
            stain_type: 'fluorescence' (bright nuclei on dark background) or
                        'he' (Hematoxylin & Eosin: purple nuclei on pink/white).

        Returns:
            Dictionary containing 'image', 'instance_mask', 'num_nuclei', 'boxes'.
        """
        n_nuclei = self.rng.integers(min_nuclei, max_nuclei + 1)
        instance_mask = np.zeros((self.height, self.width), dtype=np.int32)
        raw_signal = np.zeros((self.height, self.width), dtype=np.float32)

        boxes = []
        centers = []

        # Generate cluster centers
        num_clusters = max(2, n_nuclei // 8)
        cluster_centers = [
            (
                self.rng.uniform(60, self.width - 60),
                self.rng.uniform(60, self.height - 60),
            )
            for _ in range(num_clusters)
        ]

        nucleus_id = 1
        for _ in range(n_nuclei):
            # Decide if placed near cluster or scattered
            if self.rng.random() < cluster_probability and cluster_centers:
                c_idx = self.rng.integers(0, len(cluster_centers))
                cx = cluster_centers[c_idx][0] + self.rng.normal(0, 35)
                cy = cluster_centers[c_idx][1] + self.rng.normal(0, 35)
            else:
                cx = self.rng.uniform(40, self.width - 40)
                cy = self.rng.uniform(40, self.height - 40)

            cx = int(np.clip(cx, 30, self.width - 30))
            cy = int(np.clip(cy, 30, self.height - 30))
            centers.append((cx, cy))

            # Nucleus geometry: semi-major and semi-minor axes, rotation angle
            rx = self.rng.uniform(12, 26)
            ry = rx * self.rng.uniform(0.65, 0.95)
            angle = self.rng.uniform(0, 180)

            # Create polygon with slight boundary irregularities
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            radii = []
            for a in angles:
                # Ellipse radius equation
                cos_a = np.cos(a - np.radians(angle))
                sin_a = np.sin(a - np.radians(angle))
                base_r = (rx * ry) / np.sqrt((ry * cos_a) ** 2 + (rx * sin_a) ** 2)
                # Perturb radius slightly for organic shape
                perturb = self.rng.uniform(0.92, 1.08)
                radii.append(base_r * perturb)

            poly_pts = []
            for a, r in zip(angles, radii):
                px = int(cx + r * np.cos(a))
                py = int(cy + r * np.sin(a))
                poly_pts.append([px, py])
            poly_pts = np.array(poly_pts, dtype=np.int32)

            # Create individual nucleus mask
            single_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(single_mask, [poly_pts], 1)

            # Check if mask is non-empty
            if np.sum(single_mask) < 20:
                continue

            # Resolve overlaps: keep instance IDs separated
            # Where overlap occurs, if current pixel already has mask, assign boundary separation
            overlap = (instance_mask > 0) & (single_mask > 0)
            instance_mask[single_mask > 0] = nucleus_id
            # Slight separation barrier along touch edges
            if np.any(overlap):
                instance_mask[overlap] = nucleus_id

            # Internal texture (chromatin density, nucleoli spots)
            y_indices, x_indices = np.where(single_mask > 0)
            dist_to_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
            max_d = np.max(dist_to_center) + 1e-5
            intensity_falloff = 1.0 - 0.4 * (dist_to_center / max_d) ** 1.5
            texture_noise = self.rng.normal(1.0, 0.12, size=len(y_indices))
            chromatin_val = intensity_falloff * texture_noise * self.rng.uniform(0.75, 1.0)
            raw_signal[y_indices, x_indices] = np.maximum(
                raw_signal[y_indices, x_indices], chromatin_val
            )

            # Bounding box [x1, y1, x2, y2]
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

            nucleus_id += 1

        actual_nuclei_count = len(boxes)

        # Background gradient & illumination artifact (uneven field)
        gx, gy = np.meshgrid(
            np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height)
        )
        bg_gradient = 0.08 * (1.0 - 0.5 * (gx**2 + gy**2))
        raw_signal = raw_signal + bg_gradient

        # Add Gaussian and Poisson camera noise
        noise = self.rng.normal(0, 0.03, size=(self.height, self.width))
        raw_signal = np.clip(raw_signal + noise, 0.0, 1.0)

        if stain_type == "fluorescence":
            # Typical DAPI / Hoechst blue-cyan fluorescence image
            img_b = (raw_signal * 255).astype(np.uint8)
            img_g = (raw_signal * 180).astype(np.uint8)
            img_r = (raw_signal * 80).astype(np.uint8)
            image = np.stack([img_b, img_g, img_r], axis=-1)
        else:
            # H&E Stain (purple nuclei, pink cytoplasm/background)
            inv_signal = 1.0 - raw_signal * 0.8
            img_r = np.clip(inv_signal * 240 + 15, 0, 255).astype(np.uint8)
            img_g = np.clip(inv_signal * 190 + 20, 0, 255).astype(np.uint8)
            img_b = np.clip(inv_signal * 230 + 10, 0, 255).astype(np.uint8)
            image = np.stack([img_b, img_g, img_r], axis=-1)

        return {
            "image": image,
            "instance_mask": instance_mask,
            "num_nuclei": actual_nuclei_count,
            "boxes": boxes,
            "stain_type": stain_type,
        }
