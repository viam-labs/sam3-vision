import io

import numpy as np
from PIL import Image as PILImage

from models.pick_render import FAILED_COLOR, OTHER_COLOR, TARGET_COLOR, render_pick

H, W = 240, 320


def _mask(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# Four segments, each 40x40 px, well apart so labels don't cover other masks' edges.
MASKS = [_mask(60, 100, 20, 60), _mask(60, 100, 100, 140), _mask(160, 200, 180, 220), _mask(160, 200, 260, 300)]
BOXES = [(20, 60, 59, 99), (100, 60, 139, 99), (180, 160, 219, 199), (260, 160, 299, 199)]
BACKGROUND = (40, 40, 120)


def _render(**kwargs):
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    jpeg = render_pick(image, MASKS, BOXES, ["a", "b", "c", "d"], **kwargs)
    return np.array(PILImage.open(io.BytesIO(jpeg)).convert("RGB")).astype(int)


def _near(px, color, tol=60):
    return np.abs(px - np.array(color)).max() < tol


def test_target_green_failed_orange_others_gray():
    img = _render(target=0, failed={1})
    # Bottom edge of each mask (away from its label, which sits above it).
    assert _near(img[99, 40], TARGET_COLOR)
    assert _near(img[99, 120], FAILED_COLOR)
    assert _near(img[199, 200], OTHER_COLOR)
    # Target is filled; a failed one is only outlined.
    assert img[85, 40, 1] > BACKGROUND[1] + 60
    assert _near(img[85, 120], BACKGROUND)


def test_low_confidence_segments_hidden_unless_tried():
    visible = [True, False, False, True]
    img = _render(target=None, failed={1}, visible=visible)
    # Segment 2 is hidden: no outline, no label above it.
    assert _near(img[199, 200], BACKGROUND)
    assert _near(img[150, 200], BACKGROUND)
    # Segment 1 is below the cut but was tried, so it is still drawn in orange.
    assert _near(img[99, 120], FAILED_COLOR)
    # Segment 3 is visible and plain.
    assert _near(img[199, 280], OTHER_COLOR)
