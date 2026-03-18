"""
spatial_filtering.py
====================
Radar-guided spatial gating and SNR computation for camera images.

This module is the core of the **Radar → Camera** denoising direction in the
cross-modal reciprocal loop.  The key insight is that a 4D radar is nearly
immune to optical noise (darkness, fog, glare), and can therefore act as a
"trusted oracle" that informs the camera *where* real objects are.

Algorithm Overview
------------------
1. **Spatial Gate Generation** – For each radar return that has been projected
   onto the image plane, a soft circular region is drawn.  A Gaussian blur
   then transitions the mask smoothly from ``1.0`` (fully trusted) to
   ``0.0`` (void / untrusted).

2. **Radar-Gated Amplification** – Camera pixels inside trusted regions are
   multiplied by ``amplification_factor``; pixels in void regions are
   multiplied by the much smaller ``suppression_factor``.  The two responses
   are blended with the soft gate as an interpolation weight.

3. **SNR Calculation** – Signal-to-Noise Ratio is estimated in dB by treating
   the gated region as "signal" and the remaining area as "noise".  This
   metric is used to benchmark denoising gain across frames.

Pipeline Location
-----------------
Phase 4 (Radar-Gated Spatial Filtering) of the cross-modal pipeline.

Functions
---------
- :func:`generate_spatial_gate`  – Build a per-frame soft binary mask.
- :func:`apply_radar_gating`     – Apply the mask to denoise the camera frame.
- :func:`calculate_snr`          – Estimate per-frame SNR in dB.
"""

from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_spatial_gate(
    img_shape: Tuple[int, int],
    projected_radar_pts: np.ndarray,
    gate_radius: int = 15,
) -> np.ndarray:
    """Construct a soft spatial gate mask from projected radar point locations.

    Each validated radar return projects a circle of radius ``gate_radius``
    onto a blank float32 mask.  Overlapping circles accumulate to ``1.0``.
    A Gaussian blur then creates a smooth soft boundary so that the
    downstream gating step does not introduce hard pixel-level artefacts.

    Args:
        img_shape (tuple of int): ``(height, width)`` – dimensions of the
            output mask.  Must match the camera image dimensions.
        projected_radar_pts (np.ndarray): Projected 2-D radar positions from
            :func:`~src.utils.calibration.project_radar_to_camera`, shape
            ``(V, 2)`` with columns ``[u, v]``.  An empty array ``(0, 2)``
            is handled gracefully and returns a zero mask.
        gate_radius (int, optional): Pixel radius of the hard circle drawn
            around each radar return before Gaussian softening.
            Defaults to ``15``.

    Returns:
        np.ndarray: Float32 gate mask of shape ``(height, width)`` with
        values clipped to ``[0.0, 1.0]``.  ``1.0`` indicates a fully trusted
        region; intermediate values form the soft boundary.

    Note:
        The ``(int(u), int(v))`` cast is required because OpenCV rejects
        NumPy integer types in its drawing functions on some platforms.

    Example::

        gate = generate_spatial_gate((720, 1280), pts_2d, gate_radius=20)
        # gate.shape == (720, 1280)
    """
    gate_mask = np.zeros(img_shape, dtype=np.float32)

    for (u, v) in projected_radar_pts:
        # Cast to Python int – OpenCV rejects numpy integer types.
        cv2.circle(gate_mask, (int(u), int(v)), gate_radius, 1.0, thickness=-1)

    # Soften hard circle boundaries with a Gaussian kernel so the gating
    # transition is smooth and CNN-friendly.
    gate_mask = cv2.GaussianBlur(gate_mask, (15, 15), sigmaX=5.0)

    return np.clip(gate_mask, 0.0, 1.0)


def apply_radar_gating(
    noisy_img: np.ndarray,
    gate_mask: np.ndarray,
    amplification_factor: float = 1.5,
    suppression_factor: float = 0.2,
) -> np.ndarray:
    """Denoise a camera image by amplifying radar-trusted regions and suppressing voids.

    The denoised output is a per-pixel blend:

    .. code-block:: none

        output = gate · (img × α) + (1 − gate) · (img × β)

    where ``α = amplification_factor`` and ``β = suppression_factor``.
    In radar-verified regions the signal is boosted; in void regions the
    contribution is shrunk to near-black, substantially reducing the
    influence of high-ISO thermal and photon noise.

    Args:
        noisy_img (np.ndarray): Raw camera frame, shape ``(H, W)`` or
            ``(H, W, C)``, dtype ``uint8`` or convertible to ``float32``.
        gate_mask (np.ndarray): Soft spatial gate from
            :func:`generate_spatial_gate`, shape ``(H, W)`` with values in
            ``[0.0, 1.0]``.  The mask is **not** modified (a copy is made
            for the channel-expansion step).
        amplification_factor (float, optional): Scaling multiplier applied to
            pixels inside trusted regions.  Values ``> 1.0`` boost signal
            intensity. Defaults to ``1.5``.
        suppression_factor (float, optional): Scaling multiplier applied to
            pixels in void (untrusted) regions.  Should be ``< 1.0`` to
            suppress noise. Defaults to ``0.2``.

    Returns:
        np.ndarray: Denoised image of the same shape as ``noisy_img``, with
        dtype ``uint8`` and pixel values clipped to ``[0, 255]``.

    Example::

        denoised = apply_radar_gating(raw_frame, gate_mask,
                                      amplification_factor=1.5,
                                      suppression_factor=0.2)
    """
    # Copy the mask before expanding dims so we do not mutate the caller's array.
    mask = gate_mask.copy()

    # Promote the 2-D mask to match the image's channel dimension.
    if len(noisy_img.shape) == 3:
        mask = np.expand_dims(mask, axis=-1)  # (H, W) → (H, W, 1)

    img_float = noisy_img.astype(np.float32)

    # Amplified version – brightens radar-confirmed regions.
    amplified = img_float * amplification_factor

    # Suppressed version – dims noise-dominated void regions.
    suppressed = img_float * suppression_factor

    # Soft blend: gate controls the interpolation weight per pixel.
    fused = (mask * amplified) + ((1.0 - mask) * suppressed)

    return np.clip(fused, 0, 255).astype(np.uint8)


def calculate_snr(
    image: np.ndarray,
    signal_mask: np.ndarray,
) -> float:
    """Estimate the Signal-to-Noise Ratio (SNR) of an image region in dB.

    The "signal" is defined as the set of pixels where ``signal_mask > 0.5``
    (i.e. radar-verified regions), and "noise" as the complementary set.
    SNR is computed in decibels as:

    .. code-block:: none

        SNR_dB = 10 · log₁₀(P_signal / P_noise)

    where ``P_x = mean(pixels_x²)`` is the mean squared intensity.

    Args:
        image (np.ndarray): Grayscale ``(H, W)`` or colour ``(H, W, C)``
            image.  Any numeric dtype is accepted; values are cast to
            ``float32`` internally.
        signal_mask (np.ndarray): 2-D float mask ``(H, W)`` with values in
            ``[0.0, 1.0]``.  Pixels with value ``> 0.5`` are treated as
            signal; the rest are treated as noise.  The mask is **not**
            modified (a copy is made).

    Returns:
        float: SNR in dB.  Special values:

        - ``0.0``        – No signal pixels found (entire mask is zero).
        - ``+inf``       – No noise pixels found, or noise power is ``0``.
        - ``−inf``       – Signal power is ``0`` (dark signal region).

    Example::

        snr_before = calculate_snr(raw_frame,     gate_mask)
        snr_after  = calculate_snr(denoised_frame, gate_mask)
        print(f"SNR gain: {snr_after - snr_before:.2f} dB")
    """
    image_float = image.astype(np.float32)

    # Work on a copy of the mask to avoid side-effects in the caller.
    mask = signal_mask.copy()

    # Expand the 2-D mask to match colour channels for boolean indexing.
    if len(image.shape) == 3:
        mask = np.expand_dims(mask, axis=-1)  # (H, W) → (H, W, 1)

    # ── Signal pixels (inside the radar gate) ───────────────────────────────
    signal_pixels = image_float[mask > 0.5]
    if len(signal_pixels) == 0:
        return 0.0

    # ── Noise pixels (outside the radar gate) ───────────────────────────────
    noise_pixels = image_float[mask <= 0.5]
    if len(noise_pixels) == 0:
        return float("inf")

    signal_power = float(np.mean(signal_pixels**2))
    noise_power = float(np.mean(noise_pixels**2))

    # ── Edge-case guards ────────────────────────────────────────────────────
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")

    return 10.0 * float(np.log10(signal_power / noise_power))


# ---------------------------------------------------------------------------
# Smoke-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    h, w = 720, 1280

    # Synthetic noisy dark image (simulates zero-lux high-ISO capture).
    mock_img = np.random.randint(0, 100, (h, w, 3), dtype=np.uint8)

    # Inject a brighter "real object" patch to simulate a valid radar return.
    u_obj, v_obj = 640, 360
    cv2.circle(mock_img, (u_obj, v_obj), 30, (200, 200, 200), thickness=-1)

    # Radar returns that hit the real object location.
    pts = np.array([[u_obj, v_obj], [u_obj + 5, v_obj - 5]])

    gate = generate_spatial_gate((h, w), pts, gate_radius=40)
    denoised = apply_radar_gating(mock_img, gate)

    snr_before = calculate_snr(mock_img, gate)
    snr_after = calculate_snr(denoised, gate)

    print(f"SNR before gating : {snr_before:+.2f} dB")
    print(f"SNR after  gating : {snr_after:+.2f} dB")
    print(f"SNR improvement   : {snr_after - snr_before:+.2f} dB")
