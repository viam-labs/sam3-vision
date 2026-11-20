# sam3-vision

Vision service `viam:sam3-vision:sam3-segments`. It sends the camera frame and a text prompt to a
remote SAM3 ML model service (OpenVINO on an Intel machine), then turns the returned masks into
detections and 3D point clouds. The point-cloud pipeline is carried over from
[viam-labs/sam3-detector](https://github.com/viam-labs/sam3-detector)'s `sam3-segments` model.

## Config

```json
{
  "name": "sam3-segments",
  "api": "rdk:service:vision",
  "model": "viam:sam3-vision:sam3-segments",
  "attributes": {
    "camera_name": "realsense",
    "mlmodel_name": "sam3",
    "label": "stemless wine glass"
  }
}
```

The camera's `GetImages` must return a color image and a depth image (named with `color`/`rgb` and
`depth`, otherwise the first two images are used in that order).

| Attribute | Default | Notes |
|---|---|---|
| `camera_name` | required | RGB-D camera with aligned color + depth and intrinsics |
| `mlmodel_name` | required | The SAM3 ML model service (can live on a remote part) |
| `label` | required | Text prompt; change at runtime with `{"command": "set_label", "label": "..."}` |
| `image_tensor_name` | `image` | Input tensor name for the camera frame |
| `prompt_tensor_name` | `prompt` | Input tensor name for the UTF-8 text prompt |
| `mask_tensor_name` | `masks` | Output tensor name for the masks |
| `mask_threshold` | `0.5` | Mask pixels above this count as the object |
| `confidence_threshold` | `0.5` | Proposals scoring below this are dropped before masks are resized |
| `depth_threshold_mm` | `0` | Drop points farther than this from the segment's median depth (0 = off) |
| `min_points` | `50` | Skip segments with fewer points |
| `detector_name` | — | Optional vision service whose boxes filter SAM3 results |
| `highlighting_on`, `highlight_color` | off, green | Mask overlay on `capture_all_from_camera` images |
| `debug` | `false` | Also return each cloud before cleaning from `GetObjectPointClouds` |

## GetDetections3D

Returns one detection per segment, best confidence first. Each has:

- `transforms[0]`: root frame `<service>/object-<i>`, posed at the segment's axis-aligned bounding-box center in
  `world` (the camera frame when the frame system is unavailable), carrying that box as its geometry.
- `transforms[1]`: `<root>/points`, at the root's origin, carrying the segment's points relative to the root.
- `classifications`: the label and SAM3 confidence.
- `metadata`: `snapshot_id` and the pixel `bbox_px`.

The frame and masks are kept in memory under `snapshot_id` (last 8) for `render_pick`. Pass
`extra: {"snapshot_id": "..."}` to choose the id, so it is known even when nothing is detected.
Used by snack-restocker's `pick_snack`.

## DoCommand

- `{"command": "render_pick", "snapshot_id": "...", "target": i, "failed": [i, ...], "min_confidence": 0.75,
  "notes": {"<i>": "text"}, "title": "..."}` returns `{"image_jpeg_b64": "..."}`: that frame with detection `i`
  (by index in the `GetDetections3D` response) filled green, `failed` ones outlined orange, the rest outlined
  gray, and segments below `min_confidence` hidden unless targeted or failed.
- `{"command": "set_label", "label": "..."}` changes the text prompt.
- `{"command": "status"}` returns the current settings.

## Contract with the ML model service

Matches the `viam:openvino:sam3` ML model service design.

`Infer` request:
- `image`: `uint8`, shape `[H, W, 3]`, RGB, the raw camera frame at camera resolution.
- `prompt`: `uint8`, shape `[L]`, the label as UTF-8 bytes. Tokenization happens in the ML model service.

`Infer` response:
- `masks`: `uint8` `[N, h, w]`, values 0/1, one per proposal (any `[..., h, w]` shape is accepted).
  If `(h, w)` differs from the camera frame, masks are resized back with nearest-neighbour, so they
  must cover the whole frame (stretched, not letterboxed or cropped).
- `scores`: `float32` `[N]`, aligned with `masks`. Optional; without it every proposal has confidence 1.0.
  A count that differs from the number of masks is an error.

The ML model service rejects concurrent `Infer` calls, so this service sends one at a time and
overlapping requests wait their turn.

## Development

```bash
./setup.sh
.venv/bin/pip install pytest
.venv/bin/python -m pytest
./build.sh   # module.tar.gz
```

Hot reload onto a machine (adds the module only, no resources):

```bash
viam module reload-local --part-id <part-id>
```
