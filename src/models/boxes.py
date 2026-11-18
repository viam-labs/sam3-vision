"""Turn SAM3 instance masks into detection records.

Kept free of Viam imports so unit tests can run without the SDK. Copied from
viam-labs/sam3-detector (src/models/boxes.py), trimmed to the mask path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RawDetection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    class_name: str
    object_id: Optional[int] = None


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask (H, W) to (x_min, y_min, x_max, y_max)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def detections_from_masks(
    masks: Sequence[np.ndarray],
    scores: Optional[np.ndarray],
    label: str,
    min_score: float = 0.0,
) -> Tuple[List[RawDetection], List[np.ndarray]]:
    """Axis-aligned boxes around each mask. Returns detections and the masks they came from, 1:1."""
    dets: List[RawDetection] = []
    kept: List[np.ndarray] = []
    for i, mask in enumerate(masks):
        score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        if score < min_score:
            continue
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        dets.append(
            RawDetection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=score,
                class_name=label,
                object_id=i,
            )
        )
        kept.append(mask)
    return dets, kept
