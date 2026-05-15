"""
early_fusion.py
===============
Signal-level early fusion tensor construction and neural network frontend.

This module implements the final assembly step of the cross-modal denoising
pipeline: combining the denoised camera image with the cleaned 4D radar data
into a single, unified multi-channel tensor that can be fed directly into any
standard two-stage or single-stage object detector.

Design Philosophy: Early vs. Late Fusion
-----------------------------------------
Traditional ADAS systems fuse sensors at the **object** level (late fusion):
each modality independently detects objects, then the detections are
merged by confidence voting.  This accumulates errors from both detectors and
loses spatial precision.

This pipeline operates at the **signal** level (early fusion): raw data from
both modalities are combined *before* any detection network sees the scene.
The result is a richer, higher-SNR representation that a single detector can
reason about holistically.

Tensor Layout
-------------
The fused tensor has ``C + 2`` channels:

.. code-block:: none

    [R, G, B,  depth_z,  doppler_velocity]
     ──────────  ───────────────────────────
     from camera    from 4D radar (sparse → dense)

Sparse radar points are *rasterised* onto a dense grid (drawn as small filled
circles) so that convolutional layers can capture their spatial context.

Classes & Functions
-------------------
- :func:`construct_early_fusion_tensor`  – Build the (1, C+2, H, W) tensor.
- :class:`SimpleEarlyFusionHead`         – Lightweight conv adapter for detectors.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tensor Construction
# ---------------------------------------------------------------------------


def construct_early_fusion_tensor(
    denoised_img: np.ndarray,
    radar_pts_2d: np.ndarray,
    radar_metadata: np.ndarray,
) -> torch.Tensor:
    """Construct a multi-channel signal-level early fusion tensor.

    Combines the radar-denoised camera image (Phase 4 output) with dense
    radar feature channels derived from the cleaned, clutter-rejected point
    cloud (Phase 6 output).  The resulting tensor can be passed through
    :class:`SimpleEarlyFusionHead` and then into any standard detector
    (YOLOv8, DETR, PointPillars, etc.).

    Args:
        denoised_img (np.ndarray): Radar-gated camera frame, shape ``(H, W)``
            (grayscale) or ``(H, W, 3)`` (BGR colour), dtype ``uint8``.
        radar_pts_2d (np.ndarray): Cleaned 2-D radar positions, shape
            ``(V, 2)``, columns ``[u, v]``.  May be ``(0, 2)`` when all
            points were rejected as clutter.
        radar_metadata (np.ndarray): Per-point metadata, shape ``(V, 2)``,
            columns ``[depth_z (m), doppler_velocity (m/s)]``.

    Returns:
        torch.Tensor: Fused tensor of shape ``(1, C+2, H, W)`` where
        ``C`` is the number of image channels (1 for grayscale, 3 for RGB).
        The batch dimension ``1`` is added so the tensor is immediately
        compatible with PyTorch model ``forward()`` calls.  Dtype:
        ``float32``.

    Note:
        Radar feature channels are normalised implicitly through the detector's
        batch normalisation layers.  If your detector lacks BN, consider
        normalising depth and velocity channels explicitly before inference.

    Example::

        fused = construct_early_fusion_tensor(denoised_frame, pts, meta)
        # fused.shape → (1, 5, 720, 1280)  for a 3-channel + depth + vel tensor
        detector_out = fusion_head(fused)
    """
    h, w = denoised_img.shape[:2]

    # ── Camera channels ───────────────────────────────────────────────────────
    if len(denoised_img.shape) == 2:
        # Grayscale: (H, W) → (1, H, W) float32 in [0, 1]
        img_tensor = (
            torch.from_numpy(denoised_img).unsqueeze(0).float() / 255.0
        )
    else:
        # BGR colour → RGB colour: (H, W, 3) → (3, H, W) float32 in [0, 1]
        rgb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    # ── Radar feature channels ────────────────────────────────────────────────
    # Sparse point-cloud data is rasterised onto a dense float32 grid that
    # matches the image dimensions.  Each radar return is drawn as a filled
    # circle of radius 3 px so nearby convolutional filters can "see" it.
    depth_grid = np.zeros((h, w), dtype=np.float32)
    vel_grid = np.zeros((h, w), dtype=np.float32)

    for (u, v), (z, vel) in zip(radar_pts_2d, radar_metadata):
        # Cast to Python int: OpenCV rejects numpy int types on some platforms.
        cv2.circle(depth_grid, (int(u), int(v)), radius=3, color=float(z), thickness=-1)
        cv2.circle(vel_grid, (int(u), int(v)), radius=3, color=float(vel), thickness=-1)

    depth_tensor = torch.from_numpy(depth_grid).unsqueeze(0)  # (1, H, W)
    vel_tensor = torch.from_numpy(vel_grid).unsqueeze(0)       # (1, H, W)

    # ── Concatenate along channel axis ────────────────────────────────────────
    # Result: (C + 2, H, W)  →  unsqueeze(0) adds batch dim for inference.
    fused = torch.cat([img_tensor, depth_tensor, vel_tensor], dim=0)
    return fused.unsqueeze(0)  # (1, C+2, H, W)


# ---------------------------------------------------------------------------
# Neural Network Frontend
# ---------------------------------------------------------------------------


class SimpleEarlyFusionHead(nn.Module):
    """Lightweight convolutional adapter for multi-modal fused input tensors.

    Standard pre-trained object detectors (YOLO, DETR, EfficientDet, …) expect
    a 3-channel (RGB) input.  This minimal two-layer convolutional head maps
    the ``C + 2`` channel early-fusion tensor back to 3 channels while learning
    cross-modal feature interactions in the first layer.

    Unlike a naive channel-selection approach, the learnable weights in
    ``conv1`` allow the network to find the optimal linear combination of
    visual and radar channels for each spatial location.

    Architecture::

        Input  (B, C+2, H, W)
          │
          ▼
        Conv2d(C+2 → 16, 3×3, pad=1)  +  ReLU
          │
          ▼
        Conv2d(16  →  3, 1×1)
          │
          ▼
        Output (B, 3, H, W)  ← compatible with standard pre-trained detectors

    Args:
        in_channels (int, optional): Number of input channels.
            Must equal ``C + 2`` from :func:`construct_early_fusion_tensor`.
            Defaults to ``5`` (3 RGB + depth + velocity).
        out_channels (int, optional): Number of output channels passed to the
            downstream detector.  Defaults to ``3`` (RGB equivalent).

    Example::

        head = SimpleEarlyFusionHead(in_channels=5, out_channels=3)
        detector_input = head(fused_tensor)  # (1, 3, H, W)
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3) -> None:
        """Initialise conv layers for the fusion adapter head.

        Args:
            in_channels (int): Number of input channels (``C + 2``).
                Defaults to ``5`` (3 RGB + depth + velocity).
            out_channels (int): Output channels passed to the downstream
                detector.  Defaults to ``3`` (RGB-equivalent).
        """
        super(SimpleEarlyFusionHead, self).__init__()

        # Spatial 3×3 conv: learns cross-modal local correlations.
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        # 1×1 conv: point-wise channel projection from 16 → out_channels.
        # No spatial mixing; purely a linear channel recombination.
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fusion head forward pass.

        Args:
            x (torch.Tensor): Early-fusion tensor, shape ``(B, C+2, H, W)``.

        Returns:
            torch.Tensor: Projected tensor, shape ``(B, out_channels, H, W)``,
            ready to be passed to a standard object detector.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------
# End-to-End Training Model
# ---------------------------------------------------------------------------

class CameraFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN for feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class RadarFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # PointNet-like structure for radar point clouds (N points x 5 features)
        # Radar data from EarlyFusionDataset is [x, y, z, doppler_velocity], so 4 features.
        self.mlp1 = nn.Linear(4, 32)
        self.mlp2 = nn.Linear(32, 64)
        self.mlp3 = nn.Linear(64, 128)

    def forward(self, x):
        # x shape: [Batch, N, 4]
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        # Max pooling over the N points to get a global feature vector
        x, _ = torch.max(x, dim=1)
        return x

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.camera_net = CameraFeatureExtractor()
        self.radar_net = RadarFeatureExtractor()
        
        # Fusion Layers
        # Assuming flattened camera features + radar global vector
        # camera out shape approx: [Batch, 64, H/8, W/8] -> let's say 640x360 -> 80x45 -> 64*80*45 = 230400
        # To keep it lightweight for dummy, let's use adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # 64 * 4 * 4 = 1024
        
        self.fc1 = nn.Linear(1024 + 128, 512)
        self.fc2 = nn.Linear(512, 128)
        
        # Output head (dummy detection: predicting one bounding box per class probability)
        # Just to have a trainable loss
        self.out = nn.Linear(128, num_classes)

    def forward(self, images, radar_pc):
        cam_features = self.camera_net(images)
        cam_features = self.pool(cam_features)
        cam_features = torch.flatten(cam_features, 1) # [Batch, 1024]
        
        rad_features = self.radar_net(radar_pc) # [Batch, 128]
        
        # Concatenate Early Fusion
        fused = torch.cat((cam_features, rad_features), dim=1)
        
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        
        logits = self.out(x)
        return logits

# ---------------------------------------------------------------------------
# Smoke-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal synthetic inputs.
    mock_denoised = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_pts = np.array([[640, 360], [100, 100]])
    mock_meta = np.array([[50.0, 10.0], [20.0, -5.0]])

    fused = construct_early_fusion_tensor(mock_denoised, mock_pts, mock_meta)
    print(f"Fused tensor shape  : {tuple(fused.shape)}  (B, C+2, H, W)")

    head = SimpleEarlyFusionHead(in_channels=fused.shape[1], out_channels=3)
    out = head(fused)
    print(f"Detector-ready shape: {tuple(out.shape)}  (B, 3, H, W)")

