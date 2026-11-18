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
| `prompt_tensor_name` | `prompt` | Input tensor name for the UTF-8 text prompt |
| `mask_tensor_name` | `masks` | Output tensor name for the masks |
| `mask_threshold` | `0.5` | Mask pixels above this count as the object |
| `confidence_threshold` | `0.5` | Proposals scoring below this are dropped before masks are resized |
| `depth_threshold_mm` | `0` | Drop points farther than this from the segment's median depth (0 = off) |
| `min_points` | `50` | Skip segments with fewer points |
| `detector_name` | — | Optional vision service whose boxes filter SAM3 results |
| `highlighting_on`, `highlight_color` | off, green | Mask overlay on `capture_all_from_camera` images |
| `debug` | `false` | Also return each cloud before cleaning |

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
.venv/bin/python -m pytest
./build.sh   # module.tar.gz
```
