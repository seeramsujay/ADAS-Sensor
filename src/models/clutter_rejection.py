"""
clutter_rejection.py
====================
Geometric multipath clutter rejection for 4D radar point clouds.

This module implements the **Camera → Radar** direction of the cross-modal
denoising loop.  Radar suffers from multipath propagation: signals bounce off
wet roads, metal guardrails, and large flat surfaces before reaching a real
object, producing **ghost detections** that appear at physically impossible
locations in the scene.

Rejection Strategy
------------------
Two complementary geometric heuristics are applied per radar return:

1. **Depth-Contradiction Test** (primary, when a depth map is available)

   If a monocular depth estimate suggests that a solid surface exists at
   distance ``d_visual``, then a radar return at ``d_radar > d_visual + ε``
   is physically behind a wall and must be a multipath ghost.  The tolerance
   ``ε`` (default 3 m) accounts for depth-estimation uncertainty.

2. **Field-of-View Validity Test** (always applied)

   Projected radar coordinates that fall outside the image rectangle
   ``[0, W) × [0, H)`` cannot be verified visually and are conservative-
   ly discarded.

   .. note::
      The edge-density variable (``edge_density``) is computed but reserved
      for a future heuristic that would flag points coinciding with dense
      structural boundaries as suspect.  It is deliberately left as a
      comment-documented placeholder for extension.

Pipeline Location
-----------------
Phase 6 (Clutter Rejection via Reciprocal Loop) of the cross-modal pipeline.

Functions
---------
- :func:`apply_clutter_rejection` – Filter multipath ghosts from a radar point cloud.
"""

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_clutter_rejection(
    radar_pts_2d: np.ndarray,
    radar_metadata: np.ndarray,
    edge_map: np.ndarray,
    depth_threshold_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Remove multipath radar ghost returns using visual geometry constraints.

    Iterates over all projected radar points and applies geometric heuristics
    to identify returns that are physically inconsistent with the structural
    layout observed in the camera image.  Inconsistent points are labelled as
    multipath clutter and excluded from the fused representation.

    Args:
        radar_pts_2d (np.ndarray): Projected radar pixel coordinates, shape
            ``(V, 2)``, dtype ``int``, columns ``[u, v]``.  Typically the
            output of :func:`~src.utils.calibration.project_radar_to_camera`.
        radar_metadata (np.ndarray): Per-point metadata, shape ``(V, 2)``,
            columns ``[depth_z, doppler_velocity]`` in metres and m/s
            respectively.
        edge_map (np.ndarray): Binary structural edge map ``(H, W)``,
            dtype ``uint8``, from
            :func:`~src.models.edge_detection.extract_structural_boundaries`.
            Used for local edge-density sampling (reserved for future
            heuristics).
        depth_threshold_map (np.ndarray, optional): Dense per-pixel depth
            map ``(H, W)`` in metres, e.g. from a monocular depth estimator
            (MiDaS, DepthAnything) or a stereo-camera disparity map.  When
            provided, the depth-contradiction heuristic is activated.
            Defaults to ``None`` (heuristic disabled).

    Returns:
        tuple:
            - **filtered_pts** (*np.ndarray*, shape ``(V_new, 2)``):
              Pixel coordinates of the retained (likely real) radar returns.
              Returns shape ``(0, 2)`` with correct dtype when all points are
              dropped.
            - **filtered_meta** (*np.ndarray*, shape ``(V_new, 2)``):
              Metadata rows corresponding to ``filtered_pts``.
            - **dropped_indices** (*list of int*): Zero-based indices of the
              input points that were identified as multipath clutter or lay
              outside the image bounds.

    Note:
        The depth-contradiction tolerance of ``3.0 m`` is a conservative
        default.  Tighten it (e.g. ``1.5 m``) on well-calibrated rigs with
        accurate depth maps; loosen it when using coarse monocular estimates.

    Example::

        clean_pts, clean_meta, ghosts = apply_clutter_rejection(
            pts_2d, metadata, edge_map, depth_threshold_map=depth_map
        )
        print(f"Dropped {len(ghosts)} ghost detections.")
    """
    valid_indices: List[int] = []
    dropped_indices: List[int] = []

    h, w = edge_map.shape

    for i, ((u, v), (z, vel)) in enumerate(zip(radar_pts_2d, radar_metadata)):

        # ── Heuristic 0: Field-of-view validity ─────────────────────────────
        # A point outside the image cannot be cross-validated visually.
        if u < 0 or u >= w or v < 0 or v >= h:
            dropped_indices.append(i)
            continue

        # ── Heuristic 1: Local edge density (reserved for future use) ────────
        # Compute the fraction of edge pixels in a 20×20 neighbourhood.  High
        # density near a point might indicate it coincides with a solid
        # structural boundary (potential pass-through ghost).  This variable
        # is available for downstream use or logging; a threshold-based
        # rejection rule can be added here.
        box = 10
        u_min, u_max = max(0, u - box), min(w, u + box)
        v_min, v_max = max(0, v - box), min(h, v + box)
        region = edge_map[v_min:v_max, u_min:u_max]
        edge_density: float = float(
            np.sum(region > 0) / ((u_max - u_min) * (v_max - v_min) + 1e-6)
        )
        # Future extension: drop if edge_density > DENSITY_THRESHOLD && z anomalous

        # ── Heuristic 2: Depth-contradiction test ─────────────────────────────
        # If the visual depth map says a solid surface is at depth d_visual,
        # any radar return claiming to be further away (z > d_visual + ε) is
        # physically behind that surface and is a multipath ghost.
        is_ghost = False
        if depth_threshold_map is not None:
            d_visual: float = float(depth_threshold_map[v, u])
            depth_tolerance_m: float = 3.0  # metres – tune per rig/algorithm
            if z > d_visual + depth_tolerance_m:
                is_ghost = True

        if is_ghost:
            dropped_indices.append(i)
        else:
            valid_indices.append(i)

    # ── Build output arrays ──────────────────────────────────────────────────
    if len(valid_indices) == 0:
        # Return zero-row arrays with correct column count to avoid downstream
        # shape mismatches when callers expect (V, 2) tensors.
        return (
            np.empty((0, 2), dtype=radar_pts_2d.dtype),
            np.empty((0, 2), dtype=radar_metadata.dtype),
            dropped_indices,
        )

    filtered_pts = radar_pts_2d[valid_indices]
    filtered_meta = radar_metadata[valid_indices]

    return filtered_pts, filtered_meta, dropped_indices


# ---------------------------------------------------------------------------
# Smoke-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Three radar returns: two real, one ghost (behind a visual wall).
    pts = np.array([[100, 100], [200, 200], [300, 300]])
    meta = np.array([[10.0, 1.5], [15.0, 0.0], [5.0, -1.0]])  # [z_m, vel_m/s]

    # Edge map with a dense boundary at the location of the second point.
    e_map = np.zeros((500, 500), dtype=np.uint8)
    e_map[190:210, 190:210] = 255

    # Depth map: most of the scene is open (50 m), but there is a wall at 5 m
    # exactly where the second radar point projects.
    depth_map = np.ones((500, 500), dtype=np.float64) * 50.0
    depth_map[190:210, 190:210] = 5.0

    f_pts, f_meta, dropped = apply_clutter_rejection(
        pts, meta, e_map, depth_threshold_map=depth_map
    )

    print(f"Input points    : {len(pts)}")
    # Point 1 (Z=15m) should be dropped — wall is at 5m, 15 > 5+3 = True.
    print(f"Dropped indices : {dropped}")
    print(f"Clean points    : {len(f_pts)}")
