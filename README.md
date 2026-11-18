# sam3-vision

Vision service `viam:sam3-vision:sam3-segments`. It sends the camera frame and a text prompt to a
remote SAM3 ML model service (OpenVINO on an Intel machine), then turns the returned masks into
detections and 3D point clouds. The point-cloud pipeline is carried over from
[viam-labs/sam3-detector](https://github.com/viam-labs/sam3-detector)'s `sam3-segments` model.

## Config

```json
{
  "camera_name": "realsense",
  "mlmodel_name": "sam3",
  "label": "stemless wine glass"
}
```

| Attribute | Default | Notes |
|---|---|---|
| `camera_name` | required | RGB-D camera with aligned color + depth and intrinsics |
| `mlmodel_name` | required | The SAM3 ML model service (can live on a remote part) |
| `label` | required | Text prompt; change at runtime with `{"command": "set_label", "label": "..."}` |
| `image_tensor_name` | `image` | Input tensor name for the camera frame |
| `mask_tensor_name` | `masks` | Output tensor name for the masks |
| `mask_threshold` | `0.5` | Mask pixels above this count as the object |
| `confidence_threshold` | `0.5` | Only applies if the model also returns `scores` |
| `depth_threshold_mm` | `0` | Drop points farther than this from the segment's median depth (0 = off) |
| `min_points` | `50` | Skip segments with fewer points |
| `detector_name` | — | Optional vision service whose boxes filter SAM3 results |
| `highlighting_on`, `highlight_color` | off, green | Mask overlay on `capture_all_from_camera` images |
| `debug` | `false` | Also return each cloud before cleaning |

## Contract with the ML model service

`Infer` request:
- `image`: `uint8`, shape `[1, H, W, 3]`, RGB, the raw camera frame at camera resolution.
- `extra`: `{"prompt": "<label>"}`. Tokenization happens in the ML model service.

`Infer` response:
- `masks`: one mask per instance; any shape whose last two dims are `(h, w)` (e.g. `[N, h, w]`).
  `uint8` or `float32`. If `(h, w)` differs from the camera frame, masks are resized back with
  nearest-neighbour, so they must cover the whole frame (no letterboxing).
- `scores` (optional): `float32` `[N]`. Without it every instance has confidence 1.0.

## Development

```bash
./setup.sh
.venv/bin/python -m pytest
./build.sh   # module.tar.gz
```
