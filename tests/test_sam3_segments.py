import asyncio
import io
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image as PILImage
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.services.mlmodel import MLModel
from viam.utils import dict_to_struct

from models.sam3_segments import Sam3Segments, masks_from_tensor

H, W = 48, 64
FX = FY = 50.0
PPX, PPY = W / 2, H / 2
DEPTH_MM = 800


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class FakeCamera:
    def __init__(self, color: np.ndarray, depth: np.ndarray):
        self.color = color
        self.depth = depth

    async def get_images(self, **kwargs):
        return [
            NamedImage("color", _png(self.color), CameraMimeType.PNG),
            NamedImage("depth", _png(self.depth), CameraMimeType.PNG),
        ], None

    async def get_properties(self, **kwargs):
        return SimpleNamespace(
            intrinsic_parameters=SimpleNamespace(
                focal_x_px=FX, focal_y_px=FY, center_x_px=PPX, center_y_px=PPY
            )
        )


class FakeMLModel:
    """Mirrors the SAM3 ML model service: rejects a call while another is running."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
        self.active = 0

    async def infer(self, input_tensors, *, extra=None, timeout=None):
        if self.active:
            raise RuntimeError("busy")
        self.active += 1
        try:
            self.calls.append((input_tensors, extra))
            await asyncio.sleep(0)
            return self.outputs
        finally:
            self.active -= 1


def _prompt(inputs) -> str:
    assert inputs["prompt"].dtype == np.uint8
    assert inputs["prompt"].ndim == 1
    return inputs["prompt"].tobytes().decode("utf-8")


def _frame():
    color = np.zeros((H, W, 3), dtype=np.uint8)
    color[..., 0] = 200
    depth = np.full((H, W), DEPTH_MM, dtype=np.uint16)
    return color, depth


def _service(ml_outputs, **attrs):
    color, depth = _frame()
    camera = FakeCamera(color, depth)
    ml = FakeMLModel(ml_outputs)
    config = ComponentConfig(
        name="seg",
        attributes=dict_to_struct(
            {"camera_name": "cam", "mlmodel_name": "sam3", "label": "cup", "min_points": 1, **attrs}
        ),
    )
    deps = {
        Camera.get_resource_name("cam"): camera,
        MLModel.get_resource_name("sam3"): ml,
    }
    return Sam3Segments.new(config, deps), ml, color


def _half_res_mask(y0, y1, x0, x1):
    """Binary uint8 mask at half the camera resolution, as the SAM3 ML model service returns it."""
    m = np.zeros((H // 2, W // 2), dtype=np.uint8)
    m[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2] = 1
    return m


NO_MASKS = {"masks": np.zeros((0, H // 2, W // 2), dtype=np.uint8), "scores": np.zeros(0, np.float32)}


def test_sends_camera_frame_and_prompt_tensors():
    svc, ml, color = _service(NO_MASKS)
    asyncio.run(svc.get_detections_from_camera("cam"))
    (inputs, extra), = ml.calls
    assert extra is None
    assert inputs["image"].dtype == np.uint8
    assert inputs["image"].shape == (H, W, 3)
    np.testing.assert_array_equal(inputs["image"], color)
    assert _prompt(inputs) == "cup"


def test_set_label_changes_prompt():
    svc, ml, _ = _service(NO_MASKS)
    asyncio.run(svc.do_command({"command": "set_label", "label": "verre à vin"}))
    asyncio.run(svc.get_detections_from_camera("cam"))
    assert _prompt(ml.calls[-1][0]) == "verre à vin"


def test_concurrent_calls_do_not_hit_busy_model():
    svc, ml, _ = _service(NO_MASKS)

    async def run():
        await asyncio.gather(*(svc.get_detections_from_camera("cam") for _ in range(4)))

    asyncio.run(run())
    assert len(ml.calls) == 4


def test_low_score_proposals_dropped_from_many():
    masks = np.stack([_half_res_mask(10, 20, 10, 30)] * 200)
    scores = np.full(200, 0.1, dtype=np.float32)
    scores[137] = 0.8
    svc, _, _ = _service({"masks": masks, "scores": scores})
    dets = asyncio.run(svc.get_detections_from_camera("cam"))
    assert [round(d.confidence, 2) for d in dets] == [0.8]
    assert (dets[0].x_min, dets[0].y_min, dets[0].x_max, dets[0].y_max) == (10, 10, 29, 19)


def test_mask_and_score_count_mismatch_is_an_error():
    svc, _, _ = _service(
        {"masks": np.stack([_half_res_mask(10, 20, 10, 30)] * 3), "scores": np.ones(2, np.float32)}
    )
    with pytest.raises(ValueError, match="3 masks but 2 scores"):
        asyncio.run(svc.get_detections_from_camera("cam"))


def test_point_clouds_from_low_res_masks():
    masks = np.stack([_half_res_mask(10, 20, 10, 30), _half_res_mask(30, 40, 40, 60)])[None]
    svc, _, _ = _service({"masks": masks})
    objs = asyncio.run(svc.get_object_point_clouds("cam"))
    assert len(objs) == 2

    first = objs[0]
    assert first.geometries.reference_frame == "cam"
    geom = first.geometries.geometries[0]
    assert geom.label == "cup"
    assert geom.center.z == pytest.approx(DEPTH_MM)
    # Mask spans x 10..29 px → X = (u - ppx) * depth / fx.
    assert geom.box.dims_mm.x == pytest.approx((29 - 10) * DEPTH_MM / FX)

    header, _, body = first.point_cloud.partition(b"DATA binary\n")
    assert b"POINTS 200" in header  # 10 x 20 px mask
    x, y, z, _ = struct.unpack_from("ffff", body, 0)
    assert z == pytest.approx(DEPTH_MM / 1000.0)


def test_detections_boxes_at_camera_resolution():
    masks = _half_res_mask(10, 20, 10, 30)[None]
    svc, _, _ = _service({"masks": masks})
    dets = asyncio.run(svc.get_detections_from_camera("cam"))
    assert len(dets) == 1
    d = dets[0]
    assert (d.x_min, d.y_min, d.x_max, d.y_max) == (10, 10, 29, 19)
    assert d.class_name == "cup"
    assert d.confidence == 1.0


def test_optional_scores_filter_by_confidence():
    masks = np.stack([_half_res_mask(10, 20, 10, 30), _half_res_mask(30, 40, 40, 60)])
    svc, _, _ = _service(
        {"masks": masks, "scores": np.array([0.9, 0.2], dtype=np.float32)},
        confidence_threshold=0.5,
    )
    dets = asyncio.run(svc.get_detections_from_camera("cam"))
    assert [round(d.confidence, 2) for d in dets] == [0.9]


def test_missing_mask_tensor_is_an_error():
    svc, _, _ = _service({"something_else": np.zeros((1,), dtype=np.float32)})
    with pytest.raises(ValueError, match="masks"):
        asyncio.run(svc.get_detections_from_camera("cam"))


@pytest.mark.parametrize(
    "shape", [(3, 8, 8), (1, 3, 8, 8), (3, 1, 8, 8)]
)
def test_masks_from_tensor_shapes(shape):
    masks = masks_from_tensor(np.ones(shape, dtype=np.float32), (16, 16), 0.5)
    assert len(masks) == 3
    assert all(m.shape == (16, 16) and m.all() for m in masks)


def test_masks_from_tensor_uint8_0_255():
    t = np.zeros((1, 4, 4), dtype=np.uint8)
    t[0, :2] = 255
    (m,) = masks_from_tensor(t, (4, 4), 0.5)
    assert m[:2].all() and not m[2:].any()


def test_validate_requires_mlmodel_name():
    config = ComponentConfig(
        name="seg", attributes=dict_to_struct({"camera_name": "cam", "label": "cup"})
    )
    with pytest.raises(ValueError, match="mlmodel_name"):
        Sam3Segments.validate_config(config)


def test_validate_returns_dependencies():
    config = ComponentConfig(
        name="seg",
        attributes=dict_to_struct(
            {"camera_name": "cam", "mlmodel_name": "sam3", "label": "cup", "detector_name": "yolo"}
        ),
    )
    deps, _ = Sam3Segments.validate_config(config)
    assert deps == ["cam", "sam3", "yolo"]
