"""
edge_detection.py
=================
Structural boundary extraction from low-light and noisy camera images.

This module implements the **Camera → Radar** direction of the cross-modal
denoising reciprocal loop.  In zero-lux or fog conditions, a raw camera image
contains virtually no usable photon signal, yet structural geometry (lane
lines, curbs, guardrails, vehicles) can still be recovered by aggressively
enhancing local contrast before edge detection.

Method
------
The pipeline uses three sequential steps optimised for extreme low-light:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalisation)** –
   Operates on local 8×8 tiles with a clip limit of 2.0 to amplify faint
   detail without globally over-brightening noise.

2. **Gaussian Blur** – A 5×5 kernel removes impulsive (salt-and-pepper)
   noise *after* contrast enhancement so that Canny's gradient computation
   does not mistake noise spikes for edges.

3. **Canny Edge Detection** – The two-pass hysteresis thresholder produces a
   thin, clean binary edge map.  Thresholds are configurable so the caller
   can tune sensitivity for the current environment (e.g. lower thresholds
   in dense fog).

4. **Morphological Dilation** – A single iteration with a 3×3 kernel widens
   edges slightly, improving overlap with projected radar points in the
   subsequent clutter-rejection step.

Pipeline Location
-----------------
Phase 5 (Visual Edge Detection) of the cross-modal pipeline.

Functions
---------
- :func:`extract_structural_boundaries` – Full low-light edge-detection pipeline.
"""

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_structural_boundaries(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """Extract structural edges from a low-light / noisy camera image.

    The function applies CLAHE for contrast enhancement, Gaussian blur for
    noise pre-filtering, Canny edge detection, and a dilation pass to
    produce a thick, robust edge map suitable for geometric matching with
    the radar point cloud.

    Args:
        image (np.ndarray): Input image, either grayscale ``(H, W)`` or
            BGR colour ``(H, W, 3)``, dtype ``uint8``.  If a colour image
            is provided it is converted to grayscale internally.
        low_threshold (int, optional): Lower hysteresis threshold for the
            Canny detector.  Pixels with gradient magnitude below this value
            are rejected as non-edges.  Defaults to ``50``.
        high_threshold (int, optional): Upper hysteresis threshold for the
            Canny detector.  Pixels above this are accepted as definite edges;
            those between ``low_threshold`` and ``high_threshold`` are kept
            only if they are 8-connected to a strong edge.  Defaults to
            ``150``.

    Returns:
        np.ndarray: Binary edge map of shape ``(H, W)``, dtype ``uint8``,
        where ``255`` indicates an edge pixel and ``0`` background.

    Note:
        Lowering both thresholds (e.g. ``low=20, high=60``) increases
        sensitivity in very dark scenes at the cost of more spurious edges.
        Tune these per-environment for best clutter-rejection performance
        in Phase 6.

    Example::

        edge_map = extract_structural_boundaries(camera_frame,
                                                 low_threshold=30,
                                                 high_threshold=90)
        n_edge_px = np.count_nonzero(edge_map)
        print(f"Detected {n_edge_px} structural edge pixels.")
    """
    # ── Step 1: Grayscale conversion ─────────────────────────────────────────
    if len(image.shape) == 3:
        # Assumes BGR channel order (standard OpenCV convention).
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Already single-channel; make a copy so we don't mutate the input.
        gray = image.copy()

    # ── Step 2: CLAHE contrast enhancement ───────────────────────────────────
    # clipLimit=2.0 and tileGridSize=(8,8) are well-tested defaults for
    # automotive low-light imagery.  Increase clipLimit for denser fog.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)

    # ── Step 3: Gaussian blur (noise pre-filter) ──────────────────────────────
    # A 5×5 kernel suppresses salt-and-pepper sensor noise before computing
    # intensity gradients, preventing false edge detections.
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=0)

    # ── Step 4: Canny edge detection ─────────────────────────────────────────
    edges: np.ndarray = cv2.Canny(blurred, low_threshold, high_threshold)

    # ── Step 5: Morphological dilation ───────────────────────────────────────
    # One dilation pass thickens edges by 1 pixel in every direction.  This
    # widens the geometric footprint of structural boundaries so that
    # clutter-rejection logic has a more generous target to match against.
    kernel = np.ones((3, 3), dtype=np.uint8)
    thick_edges: np.ndarray = cv2.dilate(edges, kernel, iterations=1)

    return thick_edges


# ---------------------------------------------------------------------------
# Smoke-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    h, w = 720, 1280

    # Create a synthetic very-dark image (simulates a zero-lux scene)
    # with a faint rectangular structure (e.g. a distant vehicle outline).
    dark_img = np.ones((h, w), dtype=np.uint8) * 10

    # Slightly brighter rectangle to simulate a structural boundary.
    cv2.rectangle(dark_img, (400, 200), (800, 500), color=30, thickness=2)

    # Corrupt with salt-and-pepper noise to mimic high-ISO sensor output.
    salt_mask = np.random.randint(0, 2, (h, w), dtype=np.uint8)
    pepper_mask = np.random.randint(0, 2, (h, w), dtype=np.uint8)
    dark_img[salt_mask == 1] = 255
    dark_img[pepper_mask == 1] = 0

    edge_map = extract_structural_boundaries(dark_img)
    n_edges = int(np.count_nonzero(edge_map))
    print(f"Detected {n_edges} structural edge pixels in the synthetic scene.")
