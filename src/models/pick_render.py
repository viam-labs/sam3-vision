"""Draw which SAM3 segment a pick targeted, for a per-pick snapshot image."""

from __future__ import annotations

import io
from typing import Collection, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

TARGET_COLOR = (0, 230, 0)
FAILED_COLOR = (255, 140, 0)
OTHER_COLOR = (200, 200, 200)

Box = Tuple[int, int, int, int]


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    """Pixels within `width` of the mask's edge, inside the mask."""
    inner = mask.copy()
    for _ in range(width):
        shrunk = inner.copy()
        shrunk[1:, :] &= inner[:-1, :]
        shrunk[:-1, :] &= inner[1:, :]
        shrunk[:, 1:] &= inner[:, :-1]
        shrunk[:, :-1] &= inner[:, 1:]
        inner = shrunk
    return mask & ~inner


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_pick(
    image: np.ndarray,
    masks: Sequence[np.ndarray],
    boxes: Sequence[Box],
    labels: Sequence[str],
    target: Optional[int],
    title: str = "",
    failed: Collection[int] = (),
    visible: Optional[Sequence[bool]] = None,
) -> bytes:
    """JPEG of `image` with segments outlined and numbered: `target` filled green and boxed,
    `failed` (tried but unreachable) outlined thick orange, the rest outlined thin gray.

    labels[i] is drawn next to segment i, which is shown as #(i + 1). Segments with
    visible[i] False are left out unless they are the target or failed.
    """
    def kind(i: int) -> str:
        if i == target:
            return "target"
        if i in failed:
            return "failed"
        return "other"

    shown = [
        i for i in range(len(masks))
        if kind(i) != "other" or visible is None or visible[i]
    ]
    # Draw the plain ones first so failed and target outlines and labels end up on top.
    order = {"other": 0, "failed": 1, "target": 2}
    shown.sort(key=lambda i: order[kind(i)])

    out = image.astype(np.float32).copy()
    for i in shown:
        mask = masks[i]
        k = kind(i)
        if k == "target":
            out[mask] = out[mask] * 0.5 + np.array(TARGET_COLOR) * 0.5
            out[_outline(mask, 3)] = TARGET_COLOR
        elif k == "failed":
            out[_outline(mask, 3)] = FAILED_COLOR
        else:
            out[_outline(mask, 1)] = OTHER_COLOR

    pil = PILImage.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    font = _font(max(12, pil.height // 30))
    colors = {"target": TARGET_COLOR, "failed": FAILED_COLOR, "other": OTHER_COLOR}
    for i in shown:
        box, label = boxes[i], labels[i]
        color = colors[kind(i)]
        x0, y0, x1, y1 = box
        if kind(i) == "target":
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        text = f"#{i + 1} {label}".strip()
        _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
        tx, ty = x0, max(0, y0 - (bottom - top) - 6)
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((tx, ty), text, fill=color, font=font)
    if title:
        bbox = draw.textbbox((4, 4), title, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((4, 4), title, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def parse_notes(notes: Optional[Mapping[str, object]], count: int) -> list:
    """Per-segment note text from a {"<index>": "text"} map; unknown indices are ignored."""
    texts = [""] * count
    for key, value in (notes or {}).items():
        try:
            i = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= i < count:
            texts[i] = str(value)
    return texts
