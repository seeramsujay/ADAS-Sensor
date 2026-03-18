"""
calibration.py
==============
Sensor calibration and radar-to-camera projection utilities.

This module implements the geometric bridge between the 4D radar coordinate
frame and the 2D camera image plane.  Accurate spatial alignment is a
prerequisite for all subsequent fusion steps: a poorly-calibrated projection
will cause radar-guided spatial gates to cover the wrong pixels, making
denoising worse rather than better.

Mathematical Model
------------------
The projection follows the standard pinhole camera model:

1. **Rigid-body transform** (radar → camera 3-D frame)::

       X_cam = R · X_rad + T

   where ``R`` (3×3) is the rotation matrix and ``T`` (3×1) is the
   translation vector of the radar sensor expressed in the camera frame.

2. **Perspective projection** (3-D → 2-D pixel)::

       [u·z, v·z, z]ᵀ = K · X_cam,   then   u = u·z / z,   v = v·z / z

   where ``K`` is the 3×3 camera intrinsic matrix::

       K = [[fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]]

3. **Field-of-view filter** – only points with ``z > 0`` that project into
   the image rectangle ``[0, W) × [0, H)`` are retained.

Pipeline Location
-----------------
Phase 3 (Spatial Mapping and Calibration) of the cross-modal pipeline.

Functions
---------
- :func:`get_mock_calibration_matrices` – Return placeholder K, R, T matrices.
- :func:`project_radar_to_camera`       – Full 3-D → 2-D projection pipeline.
"""

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_mock_calibration_matrices() -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Return placeholder calibration matrices for development and testing.

    All values are reasonable approximations for a forward-facing automotive
    camera (1280 × 720) paired with a radar mounted on the same bumper bar,
    ~0.5 m below and ~1.0 m in front of the camera optical centre.

    In production, load these from your dataset's calibration JSON/YAML or a
    sensor-specific ROS parameter server.

    Returns:
        tuple:
            - **K** (*np.ndarray*, shape ``(3, 3)``): Camera intrinsic matrix.
              Focal lengths ``fx = fy = 1000 px``; principal point at the
              image centre ``(640, 360)``.
            - **R** (*np.ndarray*, shape ``(3, 3)``): Rotation matrix from
              radar to camera coordinate frame.  Identity here (sensors are
              assumed to share the same orientation for simplicity).
            - **T** (*np.ndarray*, shape ``(3, 1)``): Translation vector from
              radar to camera frame in metres.

    Example::

        K, R, T = get_mock_calibration_matrices()
        pts_2d, meta = project_radar_to_camera(radar_pc, K, R, T)
    """
    # ── Intrinsics ──────────────────────────────────────────────────────────
    # Using symmetric focal lengths equal to the image width for a ~53° FoV.
    K = np.array(
        [
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # ── Extrinsics (Radar → Camera) ─────────────────────────────────────────
    # Identity rotation: radar and camera share the same orientation.
    R = np.eye(3, dtype=np.float64)

    # Radar is 0.5 m below (−y) and 1.0 m further forward (+z) than the camera.
    T = np.array([[0.0], [-0.5], [1.0]], dtype=np.float64)

    return K, R, T


def project_radar_to_camera(
    radar_pc: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    img_shape: Tuple[int, int] = (720, 1280),
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a 4D radar point cloud onto the 2-D camera image plane.

    The 4th column (Doppler velocity) is **not** used in the projection
    geometry; it is preserved as metadata so that downstream modules can
    leverage velocity information for clutter rejection and object motion
    estimation.

    Args:
        radar_pc (np.ndarray): Point cloud of shape ``(N, 4)``.  Columns are
            ``[x, y, z, doppler_velocity]`` in the radar's own coordinate
            frame.  ``z`` is the forward-range axis.
        K (np.ndarray): Camera intrinsic matrix, shape ``(3, 3)``.
        R (np.ndarray): Rotation from radar to camera frame, shape ``(3, 3)``.
        T (np.ndarray): Translation from radar to camera frame, shape ``(3, 1)``,
            in metres.
        img_shape (tuple of int, optional): ``(height, width)`` of the target
            image.  Points projecting outside this rectangle are discarded.
            Defaults to ``(720, 1280)``.

    Returns:
        tuple:
            - **valid_points_2d** (*np.ndarray*, shape ``(V, 2)``, dtype
              ``int``): Pixel coordinates ``[u, v]`` of the radar returns that
              fall inside the image.  Returns shape ``(0, 2)`` when no points
              are visible.
            - **valid_metadata** (*np.ndarray*, shape ``(V, 2)``, dtype
              ``float64``): Per-point ``[depth_z, doppler_velocity]`` for the
              retained points.  Returns shape ``(0, 2)`` when no points are
              visible.

    Note:
        Points in front of the camera are defined by ``z_cam > 0``.  Any
        radar return with a negative or zero z-coordinate after transformation
        is behind the sensor plane and is silently dropped.

    Example::

        pts, meta = project_radar_to_camera(radar_pc, K, R, T)
        for (u, v), (z, vel) in zip(pts, meta):
            print(f"Pixel ({u},{v})  depth={z:.1f}m  vel={vel:.1f}m/s")
    """
    h, w = img_shape

    # ── Guard: empty point cloud ────────────────────────────────────────────
    if radar_pc.size == 0 or radar_pc.shape[0] == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=np.float64)

    # ── Step 1: Separate spatial and Doppler data ───────────────────────────
    xyz: np.ndarray = radar_pc[:, :3]       # (N, 3) – spatial positions
    velocity: np.ndarray = radar_pc[:, 3]   # (N,)   – radial Doppler velocities

    # ── Step 2: Rigid-body transform (radar frame → camera frame) ────────────
    # Result shape: (3, N)  –  each column is one point in the camera frame.
    xyz_cam: np.ndarray = (R @ xyz.T) + T

    # Discard points behind the camera optical plane (z_cam ≤ 0) to avoid
    # division-by-zero and phantom projections.
    front_mask: np.ndarray = xyz_cam[2, :] > 0.0
    xyz_cam = xyz_cam[:, front_mask]
    velocity = velocity[front_mask]

    # ── Step 3: Perspective projection K · X_cam ────────────────────────────
    uvz: np.ndarray = K @ xyz_cam  # (3, V) – homogeneous pixel coordinates

    # Normalise by depth to obtain pixel coordinates.
    u: np.ndarray = uvz[0, :] / uvz[2, :]
    v: np.ndarray = uvz[1, :] / uvz[2, :]
    depth: np.ndarray = uvz[2, :]

    # Round to nearest integer pixel position.
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # ── Step 4: Field-of-view filter ────────────────────────────────────────
    # Retain only points whose projected pixel falls inside the image bounds.
    img_mask: np.ndarray = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u_valid = u[img_mask]
    v_valid = v[img_mask]
    depth_valid = depth[img_mask]
    vel_valid = velocity[img_mask]

    # ── Guard: no points survived the FoV filter ────────────────────────────
    if len(u_valid) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=np.float64)

    valid_points_2d = np.stack((u_valid, v_valid), axis=1)    # (V, 2)
    valid_metadata = np.stack((depth_valid, vel_valid), axis=1)  # (V, 2)

    return valid_points_2d, valid_metadata


# ---------------------------------------------------------------------------
# Smoke-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    K, R, T = get_mock_calibration_matrices()

    # Generate 100 random radar points; force z to be positive so they are
    # in front of the sensor.
    N = 100
    mock_radar = np.random.randn(N, 4) * 5.0
    mock_radar[:, 2] = np.abs(mock_radar[:, 2]) + 5.0  # z in [5, ∞)

    pts_2d, metadata = project_radar_to_camera(mock_radar, K, R, T)
    print(f"Projected {len(pts_2d)} / {N} points inside the image plane.")
    if len(pts_2d) > 0:
        print(f"  Sample pixel : {pts_2d[0]}")
        print(f"  Depth / vel  : {metadata[0]}")
