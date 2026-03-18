"""
synchronization.py
==================
Temporal alignment utilities for multi-rate sensor streams.

In a real ADAS platform, the camera and the 4D radar operate at different
sampling frequencies (typically camera ≥ 30 Hz, radar ~10 Hz).  Before any
signal-level fusion can be performed, each camera frame must be matched to
the *closest* radar sweep in time.  This module provides that alignment logic
and a validation step to certify that the resulting temporal offsets fall
within a hard real-time bound.

Pipeline Location
-----------------
This module is invoked at **Phase 2 (Sensor Synchronisation)** of the
cross-modal denoising pipeline, immediately after raw data is loaded and
before spatial calibration.

Functions
---------
- :func:`match_timestamps`         – Nearest-neighbour timestamp matching.
- :func:`validate_synchronization` – Statistical validation of timing offsets.

Typical Usage
-------------
::

    import numpy as np
    from src.utils.synchronization import match_timestamps, validate_synchronization

    cam_ts  = np.arange(0, 1000, 33.3)   # ~30 Hz camera
    rad_ts  = np.arange(0, 1000, 100.0)  # ~10 Hz radar

    matches, offsets = match_timestamps(cam_ts, rad_ts, max_delay_ms=15.0)
    is_valid, stats  = validate_synchronization(offsets, max_delay_ms=15.0)
"""

from typing import Dict, List, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_timestamps(
    camera_timestamps: np.ndarray,
    radar_timestamps: np.ndarray,
    max_delay_ms: float = 10.0,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Match camera frames to the nearest radar sweep within a time threshold.

    For every camera frame, the function finds the radar sweep whose
    timestamp is closest in time.  If the absolute difference is within
    ``max_delay_ms``, the pair is considered a valid match.  Unmatched
    camera frames are silently discarded.

    This is an O(N·M) brute-force search which is sufficient for short
    inference windows.  For long offline sequences, replace the inner
    ``np.argmin`` with a sorted binary-search look-up.

    Args:
        camera_timestamps (np.ndarray): 1-D array of camera frame timestamps,
            in milliseconds, in ascending order.
        radar_timestamps (np.ndarray): 1-D array of radar sweep timestamps,
            in milliseconds, in ascending order.
        max_delay_ms (float, optional): Maximum acceptable absolute time
            difference (in ms) between a camera frame and its matched radar
            sweep.  Pairs exceeding this threshold are rejected.
            Defaults to ``10.0`` ms.

    Returns:
        tuple:
            - **matches** (*list of (int, int)*): Each element is a
              ``(camera_idx, radar_idx)`` pair identifying a valid match.
            - **offsets** (*list of float*): Signed temporal offsets
              ``camera_t − radar_t`` (ms) for each matched pair.  Positive
              values mean the camera frame is *later* than its radar match.

    Example::

        matches, offsets = match_timestamps(cam_ts, rad_ts, max_delay_ms=15.0)
        for c_idx, r_idx in matches:
            print(f"Camera frame {c_idx} ↔ Radar sweep {r_idx}")
    """
    matches: List[Tuple[int, int]] = []
    offsets: List[float] = []

    for c_idx, c_time in enumerate(camera_timestamps):
        # Compute absolute time differences from this camera frame to all radar sweeps.
        time_diffs = np.abs(radar_timestamps - c_time)

        # Select the radar sweep with the smallest time difference.
        best_r_idx: int = int(np.argmin(time_diffs))
        best_diff: float = float(time_diffs[best_r_idx])

        # Accept only matches that fall within the hard real-time constraint.
        if best_diff <= max_delay_ms:
            matches.append((c_idx, best_r_idx))
            # Record the signed offset so callers can plot temporal drift.
            offsets.append(float(c_time - radar_timestamps[best_r_idx]))

    return matches, offsets


def validate_synchronization(
    offsets: List[float],
    max_delay_ms: float = 10.0,
) -> Tuple[bool, Dict[str, Union[float, int, None]]]:
    """Validate that all matched pairs meet the real-time latency requirement.

    Computes descriptive statistics over the collection of signed temporal
    offsets returned by :func:`match_timestamps` and determines whether the
    worst-case absolute offset is acceptable for safety-critical operation.

    Args:
        offsets (list of float): Signed temporal offsets (ms) from a prior
            call to :func:`match_timestamps`.
        max_delay_ms (float, optional): Upper bound on the acceptable
            absolute temporal offset.  Defaults to ``10.0`` ms.

    Returns:
        tuple:
            - **is_valid** (bool): ``True`` if the maximum absolute offset
              is ≤ ``max_delay_ms``; ``False`` otherwise.
            - **stats** (dict): Summary statistics with keys:

              - ``"max_delay_ms"`` (float | None): Maximum absolute offset.
              - ``"mean_delay_ms"`` (float | None): Mean absolute offset.
              - ``"valid_count"`` (int): Number of matched pairs evaluated.

    Note:
        If ``offsets`` is empty (no pairs were matched), the function returns
        ``(False, {..., "valid_count": 0})`` to signal a total synchronisation
        failure upstream.

    Example::

        is_valid, stats = validate_synchronization(offsets)
        if not is_valid:
            raise RuntimeError(
                f"Synchronisation failure – max offset: {stats['max_delay_ms']:.1f} ms"
            )
    """
    if not offsets:
        return False, {"max_delay_ms": None, "mean_delay_ms": None, "valid_count": 0}

    abs_offsets = np.abs(offsets)
    max_offset: float = float(np.max(abs_offsets))
    mean_offset: float = float(np.mean(abs_offsets))

    # The system is time-valid only if *every* match stays within the bound.
    is_valid: bool = max_offset <= max_delay_ms

    stats: Dict[str, Union[float, int]] = {
        "max_delay_ms": max_offset,
        "mean_delay_ms": mean_offset,
        "valid_count": len(offsets),
    }

    return is_valid, stats


# ---------------------------------------------------------------------------
# Smoke-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a 1-second window: camera at ~30 Hz, radar at ~10 Hz.
    cam_time = np.arange(0, 1000, 33.3)
    rad_time = np.arange(0, 1000, 100.0)

    # Add small Gaussian jitter to radar timestamps to simulate clock drift.
    rad_time = rad_time + np.random.randn(len(rad_time)) * 2.0

    matches, offsets = match_timestamps(cam_time, rad_time, max_delay_ms=15.0)
    is_valid, stats = validate_synchronization(offsets, max_delay_ms=15.0)

    print(f"Total camera frames : {len(cam_time)}")
    print(f"Total radar sweeps  : {len(rad_time)}")
    print(f"Matched pairs       : {len(matches)}")
    print(f"Synchronisation OK  : {is_valid}")
    print(f"Timing stats        : {stats}")
