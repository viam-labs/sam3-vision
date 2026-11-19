"""
SAM3 Segments: text prompt → remote SAM3 ML model service → masks → depth → PointCloudObject.

The point-cloud pipeline is carried over from viam-labs/sam3-detector's sam3-segments
model. The difference is that SAM3 itself runs in a separate ML model service (OpenVINO
on an Intel machine); this service only sends the camera frame and prompt, then turns
the returned masks into detections and 3D point clouds.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    PointCloudObject,
    Pose,
    PoseInFrame,
    RectangularPrism,
    ResourceName,
    Vector3,
)
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.mlmodel import MLModel
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

from models.boxes import RawDetection, detections_from_masks
from models.pick_render import parse_notes, render_pick
from spatialmath import transform_points_with_pose

LOGGER = getLogger(__name__)

MAX_DEPTH_MM = 20000

DEFAULT_IMAGE_TENSOR = "image"
DEFAULT_PROMPT_TENSOR = "prompt"
DEFAULT_MASK_TENSOR = "masks"
SCORE_TENSOR = "scores"
MAX_SNAPSHOTS = 8


async def _connect_to_machine() -> Optional[RobotClient]:
    """Connect to the parent machine using env vars set by viam-server.
    Returns None if env vars are not set (e.g. running standalone)."""
    host = os.environ.get("VIAM_MACHINE_FQDN", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    api_key = os.environ.get("VIAM_API_KEY", "")
    missing = []
    if not host:
        missing.append("VIAM_MACHINE_FQDN")
    if not api_key_id:
        missing.append("VIAM_API_KEY_ID")
    if not api_key:
        missing.append("VIAM_API_KEY")
    if missing:
        LOGGER.warning(
            f"Frame system unavailable: missing environment variable(s): {', '.join(missing)}. "
            f"Point clouds will be returned in camera frame instead of world frame."
        )
        return None
    try:
        client = await RobotClient.at_address(
            host,
            RobotClient.Options(
                dial_options=DialOptions.with_api_key(api_key=api_key, api_key_id=api_key_id),
            ),
        )
        LOGGER.debug(f"Connected to machine at {host} for frame transforms")
        return client
    except Exception as e:
        LOGGER.warning(f"Could not connect to machine for frame transforms: {e}")
        return None


def viam_image_to_numpy(image: ViamImage) -> np.ndarray:
    """Convert a ViamImage to a numpy RGB array (H, W, 3)."""
    pil = PILImage.open(io.BytesIO(image.data)).convert("RGB")
    return np.array(pil)


def _depth_image_to_numpy(image: ViamImage) -> np.ndarray:
    """Convert a Viam depth image to a numpy array (H, W) of depth in mm."""
    mime = getattr(image, "mime_type", "")
    data = image.data
    if "viam" in str(mime) and "dep" in str(mime):
        depth_list = image.bytes_to_depth_array()
        return np.array(depth_list, dtype=np.float64)

    pil = PILImage.open(io.BytesIO(data))
    return np.array(pil, dtype=np.float64)


def _stack_masks(tensor: np.ndarray) -> np.ndarray:
    """Reshape any [..., h, w] mask tensor (e.g. [N, h, w], [1, N, h, w]) to [N, h, w]."""
    arr = np.asarray(tensor)
    if arr.ndim < 2 or arr.size == 0:
        return np.empty((0, 0, 0), dtype=arr.dtype)
    return arr.reshape(-1, arr.shape[-2], arr.shape[-1])


def masks_from_tensor(
    tensor: np.ndarray,
    image_hw: Tuple[int, int],
    threshold: float,
    keep: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Per-instance boolean masks at camera resolution, for only the `keep` indices if given."""
    arr = _stack_masks(tensor)
    if keep is not None:
        arr = arr[keep]
    height, width = image_hw
    masks: List[np.ndarray] = []
    for m in arr:
        binary = m > threshold
        if binary.shape != (height, width):
            # Depth is indexed with mask pixels, so masks must match the camera frame exactly.
            pil = PILImage.fromarray(binary.astype(np.uint8) * 255)
            binary = np.array(pil.resize((width, height), PILImage.NEAREST)) > 127
        masks.append(binary)
    return masks


def _encode_pcd_binary(points_mm: np.ndarray, colors: np.ndarray) -> bytes:
    """Encode points (N,3) in mm and colors (N,3) uint8 as binary PCD.
    PCD coordinates are in meters (Viam convention)."""
    n = len(points_mm)
    if n == 0:
        return b""

    header = (
        f"VERSION .7\n"
        f"FIELDS x y z rgb\n"
        f"SIZE 4 4 4 4\n"
        f"TYPE F F F F\n"
        f"COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    ).encode("ascii")

    points_m = (np.asarray(points_mm, dtype=np.float64) / 1000.0).astype("<f4")
    c = np.asarray(colors).astype(np.uint32)
    # PCD packs rgb as the raw bits of a float32, so build rows as uint32 to copy bits exactly.
    rows = np.empty((n, 4), dtype="<u4")
    rows[:, :3] = points_m.view("<u4")
    rows[:, 3] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    return header + rows.tobytes()


class Sam3Segments(Vision, EasyResource):
    """Vision service: text prompt → remote SAM3 masks → depth → 3D point clouds."""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "sam3-vision"), "sam3-segments")

    _camera: ResourceBase
    _mlmodel: ResourceBase
    _detector: Optional[ResourceBase] = None
    _detector_name: str = ""
    _camera_name: str = ""
    _mlmodel_name: str = ""
    _label: str = ""
    _image_tensor_name: str = DEFAULT_IMAGE_TENSOR
    _prompt_tensor_name: str = DEFAULT_PROMPT_TENSOR
    _mask_tensor_name: str = DEFAULT_MASK_TENSOR
    _infer_lock: asyncio.Lock
    _snapshots: "OrderedDict[str, Tuple[np.ndarray, List[_Segment]]]"
    _mask_threshold: float = 0.5
    _confidence_threshold: float = 0.5
    _depth_threshold_mm: int = 0
    _min_points: int = 50
    _highlighting_on: bool = False
    _highlight_color: Tuple[int, int, int] = (0, 255, 0)
    _debug: bool = False
    _robot_client: Optional[RobotClient] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)
        instance._robot_client = None
        instance._infer_lock = asyncio.Lock()
        instance._snapshots = OrderedDict()
        instance._apply_config(config, dependencies)
        return instance

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        self._apply_config(config, dependencies)
        LOGGER.info(
            f"Reconfigured {self.name} with camera {self._camera_name!r} "
            f"and ML model {self._mlmodel_name!r}"
        )

    def _apply_config(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = config.attributes.fields

        def num(key: str, default: float) -> float:
            return attrs[key].number_value if key in attrs else default

        def string(key: str, default: str) -> str:
            return attrs[key].string_value if key in attrs and attrs[key].string_value else default

        self._camera_name = attrs["camera_name"].string_value
        self._camera = dependencies[Camera.get_resource_name(self._camera_name)]
        self._mlmodel_name = attrs["mlmodel_name"].string_value
        self._mlmodel = dependencies[MLModel.get_resource_name(self._mlmodel_name)]
        self._label = attrs["label"].string_value
        LOGGER.info(f"Text prompt (label): {self._label!r}")

        self._detector_name = string("detector_name", "")
        self._detector = (
            dependencies[Vision.get_resource_name(self._detector_name)]
            if self._detector_name
            else None
        )

        self._image_tensor_name = string("image_tensor_name", DEFAULT_IMAGE_TENSOR)
        self._prompt_tensor_name = string("prompt_tensor_name", DEFAULT_PROMPT_TENSOR)
        self._mask_tensor_name = string("mask_tensor_name", DEFAULT_MASK_TENSOR)
        self._mask_threshold = num("mask_threshold", 0.5)
        self._confidence_threshold = num("confidence_threshold", 0.5)
        self._depth_threshold_mm = int(num("depth_threshold_mm", 0))
        self._min_points = int(num("min_points", 50))
        self._highlighting_on = attrs["highlighting_on"].bool_value if "highlighting_on" in attrs else False
        if "highlight_color" in attrs:
            color_fields = attrs["highlight_color"].struct_value.fields
            self._highlight_color = tuple(  # type: ignore[assignment]
                max(0, min(255, int(color_fields[c].number_value))) for c in ("r", "g", "b")
            )
        else:
            self._highlight_color = (0, 255, 0)
        self._debug = attrs["debug"].bool_value if "debug" in attrs else False

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = config.attributes.fields
        for key in ("camera_name", "mlmodel_name"):
            if key not in attrs or not attrs[key].string_value:
                raise ValueError(f"{key} is required")
        if "label" not in attrs or not attrs["label"].string_value:
            raise ValueError(
                "label is required — it is sent to the SAM3 ML model service as the "
                "open-vocabulary text prompt (e.g. 'stemless wine glass')."
            )
        deps = [attrs["camera_name"].string_value, attrs["mlmodel_name"].string_value]
        if "detector_name" in attrs and attrs["detector_name"].string_value:
            deps.append(attrs["detector_name"].string_value)
        return deps, []

    async def _sam3_text_prompt(
        self, color_np: np.ndarray
    ) -> Tuple[List[RawDetection], List[np.ndarray]]:
        """Run SAM3 remotely. Returns (detections, binary masks aligned 1:1)."""
        label = self._label
        inputs = {
            self._image_tensor_name: np.ascontiguousarray(color_np, dtype=np.uint8),
            self._prompt_tensor_name: np.frombuffer(label.encode("utf-8"), dtype=np.uint8),
        }
        # The SAM3 ML model service rejects concurrent Infer calls instead of queueing them.
        async with self._infer_lock:
            outputs = await self._mlmodel.infer(inputs)  # type: ignore[attr-defined]
        if self._mask_tensor_name not in outputs:
            raise ValueError(
                f"ML model {self._mlmodel_name!r} returned no {self._mask_tensor_name!r} tensor; "
                f"got {sorted(outputs)}"
            )
        raw_masks = outputs[self._mask_tensor_name]
        keep = None
        scores = outputs.get(SCORE_TENSOR)
        if scores is not None:
            scores = np.asarray(scores).reshape(-1)
            count = len(_stack_masks(raw_masks))
            if len(scores) != count:
                raise ValueError(
                    f"ML model {self._mlmodel_name!r} returned {count} masks but {len(scores)} scores"
                )
            keep = np.flatnonzero(scores >= self._confidence_threshold)
            scores = scores[keep]
        masks = masks_from_tensor(raw_masks, color_np.shape[:2], self._mask_threshold, keep)
        return detections_from_masks(masks, scores, label)

    async def _maybe_filter_with_upstream(
        self, image: ViamImage, dets: List[RawDetection], masks: List[np.ndarray]
    ) -> Tuple[List[RawDetection], List[np.ndarray]]:
        """If an optional upstream detector is configured, keep SAM3 instances
        that overlap an upstream box of the same (or any) label."""
        if self._detector is None:
            return dets, masks
        upstream = await self._detector.get_detections(image)  # type: ignore[attr-defined]
        kept_dets: List[RawDetection] = []
        kept_masks: List[np.ndarray] = []
        for det, mask in zip(dets, masks):
            for u in upstream:
                if u.confidence < self._confidence_threshold:
                    continue
                if not _boxes_overlap(
                    (det.x_min, det.y_min, det.x_max, det.y_max),
                    (u.x_min, u.y_min, u.x_max, u.y_max),
                ):
                    continue
                kept_dets.append(det)
                kept_masks.append(mask)
                break
        return kept_dets, kept_masks

    async def _get_images(self) -> Tuple[Optional[ViamImage], Optional[ViamImage]]:
        images, _ = await self._camera.get_images()  # type: ignore[attr-defined]
        color_img = None
        depth_img = None
        for img in images:
            name = getattr(img, "name", "") or getattr(img, "source_name", "") or ""
            if "color" in name.lower() or "rgb" in name.lower():
                color_img = img
            elif "depth" in name.lower():
                depth_img = img
        if color_img is None and len(images) >= 1:
            color_img = images[0]
        if depth_img is None and len(images) >= 2:
            depth_img = images[1]
        return color_img, depth_img

    async def _get_color_and_depth(self) -> Tuple[ViamImage, np.ndarray, np.ndarray]:
        color_img, depth_img = await self._get_images()
        if color_img is None or depth_img is None:
            raise ValueError("Camera must return both color and depth images")
        return color_img, viam_image_to_numpy(color_img), _depth_image_to_numpy(depth_img)

    def _overlay_masks(self, image: ViamImage, masks: List[np.ndarray]) -> ViamImage:
        color_np = viam_image_to_numpy(image)
        overlay = color_np.copy()
        color = np.array(self._highlight_color, dtype=np.uint8)
        for mask in masks:
            bool_mask = np.asarray(mask).astype(bool)
            if bool_mask.shape[:2] != overlay.shape[:2]:
                continue
            overlay[bool_mask] = (
                overlay[bool_mask].astype(np.float32) * 0.5 + color * 0.5
            ).astype(np.uint8)
        pil = PILImage.fromarray(overlay)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return ViamImage(buf.getvalue(), CameraMimeType.JPEG)

    def _mask_to_points_raw(
        self,
        mask: np.ndarray,
        color_np: np.ndarray,
        depth_np: np.ndarray,
        fx: float, fy: float, ppx: float, ppy: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        vs, us = np.where(mask)
        depths = depth_np[vs, us].astype(np.float64)
        valid = (depths > 0) & (depths < MAX_DEPTH_MM)
        vs, us, depths = vs[valid], us[valid], depths[valid]
        if len(depths) == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        xs = (us.astype(np.float64) - ppx) * depths / fx
        ys = (vs.astype(np.float64) - ppy) * depths / fy
        zs = depths
        points = np.stack([xs, ys, zs], axis=1)
        colors = color_np[vs, us]
        return points, colors

    def _clean_points(
        self, points: np.ndarray, colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._depth_threshold_mm <= 0 or len(points) == 0:
            return points, colors
        depths = points[:, 2]
        median = np.median(depths)
        within = np.abs(depths - median) <= self._depth_threshold_mm
        return points[within], colors[within]

    def _build_point_cloud_object(
        self, points: np.ndarray, colors: np.ndarray, label: str, ref_frame: str = ""
    ) -> PointCloudObject:
        pcd_bytes = _encode_pcd_binary(points, colors)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        dims = maxs - mins
        geometry = Geometry(
            center=Pose(
                x=float(center[0]), y=float(center[1]), z=float(center[2]),
                o_x=0, o_y=0, o_z=1, theta=0,
            ),
            box=RectangularPrism(dims_mm=Vector3(
                x=float(dims[0]), y=float(dims[1]), z=float(dims[2]),
            )),
            label=label,
        )
        return PointCloudObject(
            point_cloud=pcd_bytes,
            geometries=GeometriesInFrame(
                reference_frame=ref_frame,
                geometries=[geometry],
            ),
        )

    async def _ensure_robot_client(self):
        if self._robot_client is None:
            self._robot_client = await _connect_to_machine()
        return self._robot_client

    async def _transform_points_to_world(self, points_mm: np.ndarray) -> Tuple[np.ndarray, str]:
        client = await self._ensure_robot_client()
        if client is None:
            return points_mm, self._camera_name
        try:
            origin = PoseInFrame(
                reference_frame=self._camera_name,
                pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0),
            )
            world_pose_in_frame = await client.transform_pose(origin, "world")
            p = world_pose_in_frame.pose
            transformed = transform_points_with_pose(
                p.o_x, p.o_y, p.o_z, p.theta,
                p.x, p.y, p.z,
                points_mm,
            )
            return transformed, "world"
        except Exception as e:
            LOGGER.warning(f"Could not transform to world frame: {e}")
            return points_mm, self._camera_name

    async def _prompt_image(
        self, image: ViamImage
    ) -> Tuple[List[RawDetection], List[np.ndarray]]:
        color_np = viam_image_to_numpy(image)
        dets, masks = await self._sam3_text_prompt(color_np)
        return await self._maybe_filter_with_upstream(image, dets, masks)

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        results = []
        for seg in await self._segments():
            results.append(
                self._build_point_cloud_object(seg.points, seg.colors, seg.label, seg.ref_frame)
            )
            if self._debug:
                points_raw_world, raw_ref_frame = await self._transform_points_to_world(seg.raw_points)
                results.append(
                    self._build_point_cloud_object(
                        points_raw_world, seg.raw_colors, f"debug_{seg.label}_before_cleaning", raw_ref_frame
                    )
                )
        return results

    async def _segments(self) -> List[_Segment]:
        segments, _ = await self._segments_with_frame()
        return segments

    async def _segments_with_frame(self) -> Tuple[List[_Segment], np.ndarray]:
        """SAM3 instances as cleaned 3D points (world frame when available), in model order,
        plus the color frame they came from."""
        props = await self._camera.get_properties()  # type: ignore[attr-defined]
        intrinsics = props.intrinsic_parameters
        if intrinsics is None:
            raise ValueError("Camera must provide intrinsic parameters for 3D projection")
        fx = intrinsics.focal_x_px
        fy = intrinsics.focal_y_px
        ppx = intrinsics.center_x_px
        ppy = intrinsics.center_y_px

        color_viam, color_np, depth_np = await self._get_color_and_depth()
        dets, masks = await self._prompt_image(color_viam)
        LOGGER.debug(f"SAM3 text prompt {self._label!r} returned {len(dets)} instance(s)")

        segments: List[_Segment] = []
        for det, mask in zip(dets, masks):
            if not np.any(mask):
                continue
            points_raw, colors_raw = self._mask_to_points_raw(
                mask, color_np, depth_np, fx, fy, ppx, ppy
            )
            points, colors = self._clean_points(points_raw, colors_raw)
            if len(points) < self._min_points:
                LOGGER.debug(f"Skipping segment with {len(points)} points (min={self._min_points})")
                continue
            points, ref_frame = await self._transform_points_to_world(points)
            segments.append(
                _Segment(
                    label=det.class_name or self._label or "object",
                    confidence=det.confidence,
                    points=points,
                    colors=colors,
                    ref_frame=ref_frame,
                    raw_points=points_raw,
                    raw_colors=colors_raw,
                    mask=mask,
                    bbox_px=(det.x_min, det.y_min, det.x_max, det.y_max),
                )
            )
        return segments, color_np

    async def _object_summaries(self) -> Mapping[str, ValueTypes]:
        """Segments as plain dicts, best-first, in the shape snack-restocker's pick_snack reads.

        Also keeps the frame and masks under `snapshot_id` so `render_pick` can draw the chosen one.
        """
        segments, color_np = await self._segments_with_frame()
        segments.sort(key=lambda s: s.confidence, reverse=True)
        snapshot_id = uuid.uuid4().hex
        self._snapshots[snapshot_id] = (color_np, segments)
        while len(self._snapshots) > MAX_SNAPSHOTS:
            self._snapshots.popitem(last=False)

        summaries: List[Mapping[str, ValueTypes]] = []
        for seg in segments:
            mins = seg.points.min(axis=0)
            maxs = seg.points.max(axis=0)
            x_min, y_min, x_max, y_max = seg.bbox_px
            summaries.append(
                {
                    "label": seg.label,
                    "confidence": float(seg.confidence),
                    "reference_frame": seg.ref_frame,
                    "num_points": float(len(seg.points)),
                    "center_mm": _xyz((mins + maxs) / 2.0),
                    "dims_mm": _xyz(maxs - mins),
                    "min_mm": _xyz(mins),
                    "max_mm": _xyz(maxs),
                    "bbox_px": {
                        "x_min": float(x_min), "y_min": float(y_min),
                        "x_max": float(x_max), "y_max": float(y_max),
                    },
                }
            )
        height, width = color_np.shape[:2]
        return {
            "objects": summaries,
            "snapshot_id": snapshot_id,
            "image_width": float(width),
            "image_height": float(height),
        }

    def _render_pick(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        snapshot_id = str(command.get("snapshot_id", ""))
        if snapshot_id not in self._snapshots:
            return {"error": f"unknown or expired snapshot_id {snapshot_id!r}"}
        color_np, segments = self._snapshots[snapshot_id]
        target = command.get("target")
        target_index = int(target) if isinstance(target, (int, float)) and target >= 0 else None
        notes = parse_notes(command.get("notes"), len(segments))  # type: ignore[arg-type]
        labels = [
            f"{seg.confidence:.2f} {note}".strip() for seg, note in zip(segments, notes)
        ]
        failed_raw = command.get("failed") or []
        failed = {int(i) for i in failed_raw if isinstance(i, (int, float))}  # type: ignore[union-attr]
        min_confidence = command.get("min_confidence", 0.0)
        min_confidence = float(min_confidence) if isinstance(min_confidence, (int, float)) else 0.0
        jpeg = render_pick(
            color_np,
            [seg.mask for seg in segments],
            [seg.bbox_px for seg in segments],
            labels,
            target_index,
            title=str(command.get("title", "")),
            failed=failed,
            visible=[seg.confidence >= min_confidence for seg in segments],
        )
        return {"image_jpeg_b64": base64.b64encode(jpeg).decode("ascii")}

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        dets, _ = await self._prompt_image(image)
        return [_to_detection(d) for d in dets]

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        color_img, _ = await self._get_images()
        if color_img is None:
            return []
        return await self.get_detections(color_img, extra=extra, timeout=timeout)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        if camera_name not in (self._camera_name, ""):
            raise ValueError(
                f"Camera name '{camera_name}' does not match configured camera '{self._camera_name}'."
            )
        result = CaptureAllResult()
        color_viam, _ = await self._get_images()
        if color_viam is None:
            return result
        dets, masks = ([], [])
        if return_image or return_detections:
            dets, masks = await self._prompt_image(color_viam)
        if return_image:
            if self._highlighting_on and masks:
                result.image = self._overlay_masks(color_viam, masks)
            else:
                result.image = color_viam
        if return_detections:
            result.detections = [_to_detection(d) for d in dets]
        if return_object_point_clouds:
            result.objects = await self.get_object_point_clouds(
                camera_name, extra=extra, timeout=timeout
            )
        return result

    async def get_classifications_from_camera(
        self, camera_name: str, count: int, *, extra=None, timeout=None
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_classifications(
        self, image: ViamImage, count: int, *, extra=None, timeout=None
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_properties(self, *, extra=None, timeout=None) -> Vision.Properties:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=True,
        )

    async def close(self):
        if self._robot_client is not None:
            await self._robot_client.close()
            self._robot_client = None

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command", "")
        if cmd == "get_object_point_clouds":
            try:
                return await self._object_summaries()
            except Exception as e:
                LOGGER.warning(f"get_object_point_clouds command failed: {e}")
                return {"error": str(e)}
        if cmd == "render_pick":
            try:
                return self._render_pick(command)
            except Exception as e:
                LOGGER.warning(f"render_pick command failed: {e}")
                return {"error": str(e)}
        if cmd == "set_label":
            label = str(command.get("label", "")).strip()
            if not label:
                return {"error": "label is required, e.g. 'stemless wine glass'"}
            self._label = label
            LOGGER.info(f"Text prompt set to {label!r}")
            return {"status": f"label set to {label!r}"}
        if cmd == "status":
            return {
                "detector_name": self._detector_name,
                "camera_name": self._camera_name,
                "mlmodel_name": self._mlmodel_name,
                "label": self._label,
                "confidence_threshold": self._confidence_threshold,
                "mask_threshold": self._mask_threshold,
                "depth_threshold_mm": float(self._depth_threshold_mm),
                "min_points": float(self._min_points),
            }
        return {"error": f"unknown command: {cmd}"}


@dataclass
class _Segment:
    label: str
    confidence: float
    points: np.ndarray
    colors: np.ndarray
    ref_frame: str
    raw_points: np.ndarray
    raw_colors: np.ndarray
    mask: np.ndarray
    bbox_px: Tuple[int, int, int, int]


def _xyz(v: np.ndarray) -> Mapping[str, ValueTypes]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _to_detection(d: RawDetection) -> Detection:
    return Detection(
        x_min=d.x_min, y_min=d.y_min, x_max=d.x_max, y_max=d.y_max,
        confidence=d.confidence, class_name=d.class_name,
    )


def _boxes_overlap(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)
