"""The vectorized point-cloud encoding and world transform must match the original per-point loops."""

import ctypes
import math
import struct
from ctypes import c_double

import numpy as np
import pytest

import spatialmath
from models.sam3_segments import _encode_pcd_binary


def _encode_pcd_binary_loop(points_mm, colors):
    n = len(points_mm)
    points_m = points_mm / 1000.0
    buf = bytearray(n * 16)
    for i in range(n):
        x, y, z = points_m[i]
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        rgb_int = (r << 16) | (g << 8) | b
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        struct.pack_into("ffff", buf, i * 16, float(x), float(y), float(z), rgb_float)
    return bytes(buf)


def _transform_loop(o_x, o_y, o_z, theta_deg, tx, ty, tz, points):
    lib = spatialmath._get_lib()
    ov = lib.new_orientation_vector(
        c_double(o_x), c_double(o_y), c_double(o_z), c_double(math.radians(theta_deg))
    )
    q = lib.quaternion_from_orientation_vector(ov)
    lib.free_orientation_vector_memory(ov)
    result = np.empty_like(points)
    for i in range(len(points)):
        v = spatialmath._Vector3(float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
        rv = lib.quaternion_rotate_vector(q, ctypes.byref(v))
        result[i] = [rv.contents.x + tx, rv.contents.y + ty, rv.contents.z + tz]
        lib.free_vector_memory(rv)
    lib.free_quaternion_memory(q)
    return result


@pytest.mark.parametrize("seed", range(5))
def test_pcd_encoding_is_byte_identical(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-5000, 5000, size=(2000, 3))
    colors = rng.integers(0, 256, size=(2000, 3), dtype=np.uint8)
    header, _, body = _encode_pcd_binary(points, colors).partition(b"DATA binary\n")
    assert b"POINTS 2000" in header
    assert body == _encode_pcd_binary_loop(points, colors)


@pytest.mark.parametrize(
    "pose",
    [
        (0, 0, 1, 0, 0, 0, 0),
        (0, 0, 1, 90, 100, -50, 25),
        (1, 0, 0, 45, 0, 0, 500),
        (0.3, -0.5, 0.8, 137, -250, 400, 1200),
        (0, 0, -1, 180, 10, 20, 30),
    ],
)
def test_world_transform_matches_per_point_rust(pose):
    rng = np.random.default_rng(0)
    points = rng.uniform(-2000, 2000, size=(500, 3))
    expected = _transform_loop(*pose, points)
    actual = spatialmath.transform_points_with_pose(*pose, points)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_world_transform_empty():
    out = spatialmath.transform_points_with_pose(0, 0, 1, 30, 1, 2, 3, np.empty((0, 3)))
    assert out.shape == (0, 3)
