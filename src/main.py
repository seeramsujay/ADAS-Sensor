"""
main.py
=======
End-to-end entry point for the Cross-Modal Denoising Early Fusion pipeline.

This script orchestrates the full forward pass through all pipeline phases for
a batch of camera–radar pairs.  It is intended as both a runnable
demonstration and a reference integration showing how each module fits
together.

Pipeline Execution Order
------------------------
::

    [EarlyFusionDataset]        Phase 1 – Data loading & synchronisation
          │
          ▼
    [project_radar_to_camera]   Phase 2 – Radar point cloud → image plane
          │
          ▼
    [generate_spatial_gate]     Phase 3 – Build soft radar-gated mask
    [apply_radar_gating]                  Amplify/suppress camera pixels
    [calculate_snr]                       Log per-frame SNR gain
          │
          ▼
    [extract_structural_boundaries] Phase 4 – Low-light edge map
          │
          ▼
    [apply_clutter_rejection]   Phase 5 – Remove multipath ghost returns
          │
          ▼
    [construct_early_fusion_tensor] Phase 6 – Build (1, C+2, H, W) tensor
          │
          ▼
    [SimpleEarlyFusionHead]     Phase 7 – Adapt tensor for object detector

Usage
-----
Run from the repository root with the source path on Python's path::

    PYTHONPATH=. python src/main.py

To iterate over the full dataset, change ``range(1)`` in :func:`run_pipeline`
to ``range(len(dataset))``.
"""

import time

import numpy as np

from src.data.loader import EarlyFusionDataset
from src.models.clutter_rejection import apply_clutter_rejection
from src.models.early_fusion import (
    SimpleEarlyFusionHead,
    construct_early_fusion_tensor,
)
from src.models.edge_detection import extract_structural_boundaries
from src.models.spatial_filtering import (
    apply_radar_gating,
    calculate_snr,
    generate_spatial_gate,
)
from src.utils.calibration import (
    get_mock_calibration_matrices,
    project_radar_to_camera,
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Execute the cross-modal denoising pipeline on a set of frames.

    Loads sensor data from :class:`~src.data.loader.EarlyFusionDataset`,
    runs every fusion phase in sequence, and prints latency and quality
    metrics for each frame.

    The function currently processes a single frame (``range(1)``) for quick
    demonstration.  Extend ``range(...)`` to cover the full dataset for
    benchmarking.

    Prints:
        Per-frame console output including:
        - Number of valid projected radar points.
        - SNR before and after radar-gated denoising (dB).
        - Number of structural edge pixels detected.
        - Number of multipath ghost returns dropped.
        - Shape of the fused tensor sent to the detector head.
        - End-to-end latency for the frame in milliseconds.

    Returns:
        None
    """
    print("=" * 60)
    print(" Cross-Modal Denoising ADAS Pipeline — Demo Run")
    print("=" * 60)

    # ── Phase 1: Initialise data loaders and calibration ─────────────────────
    dataset = EarlyFusionDataset(data_root=".", debug=True)
    K, R, T = get_mock_calibration_matrices()

    # Fusion head: maps (1, 5, H, W) → (1, 3, H, W) for downstream detectors.
    fusion_head = SimpleEarlyFusionHead(in_channels=5, out_channels=3)

    total_ms = 0.0

    for i in range(1):  # Expand to range(len(dataset)) for full evaluation.
        print(f"\n--- Frame {i} ---")
        t0 = time.perf_counter()

        # Phase 2: Sensor synchronisation is handled implicitly by the
        # dataset's pairing logic.  In production, call match_timestamps()
        # here and only proceed with successfully matched pairs.
        data = dataset[i]

        # Recover the uint8 numpy image from the normalised float tensor.
        camera_frame: np.ndarray = (
            data["camera"].permute(1, 2, 0).numpy() * 255.0
        ).astype(np.uint8)  # (H, W, 3)

        # Trim to valid (non-padded) radar points using the count stored at load time.
        n_valid = int(data["num_radar_points"])
        radar_pc: np.ndarray = (
            data["radar"].numpy()[:n_valid]   # (N, 4)
        )

        # ── Phase 3: Spatial Mapping ──────────────────────────────────────────
        pts_2d, metadata = project_radar_to_camera(
            radar_pc, K, R, T, img_shape=(720, 1280)
        )
        print(f"Projected radar points : {len(pts_2d)}")

        if len(pts_2d) == 0:
            print("  ⚠  No valid radar points in field of view. Skipping fusion.")
            continue

        # ── Phase 4: Radar-Gated Spatial Filtering ────────────────────────────
        gate_mask = generate_spatial_gate(
            (720, 1280), pts_2d, gate_radius=20
        )
        denoised_img = apply_radar_gating(
            camera_frame, gate_mask,
            amplification_factor=1.5,
            suppression_factor=0.2,
        )

        snr_before = calculate_snr(camera_frame, gate_mask)
        snr_after = calculate_snr(denoised_img, gate_mask)
        print(
            f"SNR  Before={snr_before:+.2f} dB  |  "
            f"After={snr_after:+.2f} dB  |  "
            f"Gain={snr_after - snr_before:+.2f} dB"
        )

        # ── Phase 5: Visual Edge Detection ───────────────────────────────────
        edge_map = extract_structural_boundaries(camera_frame)
        print(f"Structural edge pixels : {int(np.count_nonzero(edge_map))}")

        # ── Phase 6: Clutter Rejection ────────────────────────────────────────
        clean_pts, clean_meta, dropped = apply_clutter_rejection(
            pts_2d, metadata, edge_map
        )
        print(f"Ghost returns dropped  : {len(dropped)}")

        # ── Phase 7: Signal-Level Early Fusion ───────────────────────────────
        fused = construct_early_fusion_tensor(denoised_img, clean_pts, clean_meta)
        print(f"Fused tensor shape     : {tuple(fused.shape)}  (B, C+2, H, W)")

        # ── Phase 8: Fusion Head (detector adapter) ───────────────────────────
        detector_input = fusion_head(fused)
        print(
            f"Detector-ready shape   : {tuple(detector_input.shape)}  (B, 3, H, W)"
        )

        # ── Latency reporting ─────────────────────────────────────────────────
        frame_ms = (time.perf_counter() - t0) * 1_000
        total_ms += frame_ms
        print(f"Frame latency          : {frame_ms:.2f} ms")

    print(f"\n{'=' * 60}")
    print(f" Pipeline complete.  Total elapsed : {total_ms:.2f} ms")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
