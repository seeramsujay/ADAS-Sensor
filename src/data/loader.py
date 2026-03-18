"""
loader.py
=========
Dataset and DataLoader utilities for the Cross-Modal Denoising Early Fusion pipeline.

This module defines :class:`EarlyFusionDataset`, a PyTorch-compatible ``Dataset``
that loads (or, in debug/mock mode, synthesises) temporally synchronised pairs of:

- **Camera frames** – represented as high-ISO noisy RGB tensors that simulate
  zero-lux or adverse-weather conditions.
- **4D Radar point clouds** – padded to a fixed maximum length so that
  PyTorch's default collate function can batch samples without errors.

Usage
-----
For real data, subclass :class:`EarlyFusionDataset` and override
:meth:`EarlyFusionDataset.load_metadata` to parse your dataset's index files
(e.g. nuScenes *scene* tokens or RADIATE CSV manifests).

Example::

    from torch.utils.data import DataLoader
    from src.data.loader import EarlyFusionDataset

    ds = EarlyFusionDataset(data_root="/data/radiate", split="train")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    for batch in loader:
        camera = batch["camera"]   # (B, 3, H, W)
        radar  = batch["radar"]    # (B, MAX_RADAR_POINTS, 4)
        n_pts  = batch["num_radar_points"]  # (B,) – valid rows in radar tensor

Dependencies
------------
- numpy
- opencv-python (cv2)
- torch / torchvision
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed upper bound on radar points per frame.
#: All point clouds are zero-padded to this size so samples can be stacked
#: into a uniform batch tensor without a custom collate function.
MAX_RADAR_POINTS: int = 150

#: Synthetic image resolution (height, width) used in mock / debug mode.
MOCK_IMG_SHAPE: tuple = (720, 1280)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EarlyFusionDataset(Dataset):
    """PyTorch Dataset for paired Camera and 4D Radar sensor streams.

    In **debug mode**, synthetic noisy images and random point clouds are
    generated on-the-fly, making it possible to validate the full pipeline
    without real sensor data.

    In **production mode**, override :meth:`load_metadata` to populate
    ``self.camera_paths`` and ``self.radar_paths`` from your dataset's index,
    then override :meth:`__getitem__` to load the actual files.

    Args:
        data_root (str): Absolute or relative path to the dataset root
            directory (e.g. ``"/data/nuscenes"``).
        split (str, optional): Dataset partition – one of ``"train"``,
            ``"val"``, or ``"test"``. Defaults to ``"train"``.
        debug (bool, optional): When ``True``, replaces real I/O with
            synthetic data generation so the pipeline can run without a
            dataset. Defaults to ``False``.

    Attributes:
        camera_paths (List[str]): Ordered list of camera image file paths.
        radar_paths (List[str]): Ordered list of radar scan file paths.
            Must be the same length as ``camera_paths`` after
            :meth:`load_metadata` is called.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        debug: bool = False,
    ) -> None:
        """Initialise the dataset and load file-path metadata.

        Args:
            data_root (str): Root directory of the sensor dataset.
            split (str): Data partition – ``"train"``, ``"val"``, or ``"test"``.
            debug (bool): If ``True``, generates synthetic data instead of
                reading from disk.  Useful for pipeline smoke-tests without a
                real dataset.
        """
        self.data_root: str = data_root
        self.split: str = split
        self.debug: bool = debug

        # Populated by load_metadata(); kept as instance attributes so
        # subclasses can inspect them for sanity-checks.
        self.camera_paths: List[str] = []
        self.radar_paths: List[str] = []

        self.load_metadata()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_metadata(self) -> None:
        """Populate ``camera_paths`` and ``radar_paths`` from dataset index.

        **Override this method** in a subclass to support a real dataset.
        The default implementation generates placeholder path strings when
        ``debug=True`` and leaves both lists empty otherwise.

        Note:
            Both lists must have identical lengths after this method returns;
            index ``i`` in ``camera_paths`` must correspond to the same point
            in time as index ``i`` in ``radar_paths``.
        """
        if self.debug:
            # Generate 10 synthetic sample identifiers for pipeline testing.
            self.camera_paths = [f"camera_{i}.png" for i in range(10)]
            self.radar_paths = [f"radar_{i}.npy" for i in range(10)]

    def __len__(self) -> int:
        """Return the number of synchronised camera–radar pairs."""
        return len(self.camera_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Load and return a single camera–radar sample.

        In debug mode, a synthetic salt-and-pepper noisy image and a random
        4D radar point cloud are generated.  The point cloud is zero-padded
        to :data:`MAX_RADAR_POINTS` rows so that samples can be stacked into
        a uniform batch tensor.

        Args:
            idx (int): Index of the sample to load (0-based).

        Returns:
            dict: A dictionary with the following keys:

            - ``"camera"`` (:class:`torch.Tensor`, shape ``(3, H, W)``,
              dtype ``float32``): Normalised [0, 1] noisy camera frame.
            - ``"radar"`` (:class:`torch.Tensor`, shape
              ``(MAX_RADAR_POINTS, 4)``, dtype ``float32``): Zero-padded
              point cloud.  Each row is ``[x, y, z, doppler_velocity]``.
            - ``"num_radar_points"`` (int): Number of *valid* (non-padded)
              rows in the radar tensor.
            - ``"frame_id"`` (str): Identifier string for the camera frame
              (e.g. a file path or dataset token).

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        # ------------------------------------------------------------------
        # 1. Camera – synthetic high-ISO noisy image
        # ------------------------------------------------------------------
        # Start from a fully-black image to simulate a zero-lux scene, then
        # blend in uniform random noise to mimic high-ISO salt-and-pepper
        # sensor noise (ISO 102400+ behaviour).
        camera_img = np.zeros((*MOCK_IMG_SHAPE, 3), dtype=np.uint8)
        noise = np.random.randint(0, 256, (*MOCK_IMG_SHAPE, 3), dtype=np.uint8)
        camera_img = cv2.addWeighted(camera_img, 0.5, noise, 0.5, 0)

        # ------------------------------------------------------------------
        # 2. Radar – synthetic 4D point cloud
        # ------------------------------------------------------------------
        # Columns: x (m), y (m), z (m), doppler_velocity (m/s).
        N_points = np.random.randint(50, MAX_RADAR_POINTS)
        radar_pc = (np.random.randn(N_points, 4) * 10).astype(np.float32)

        # Zero-pad to MAX_RADAR_POINTS so PyTorch's default collate_fn can
        # stack tensors from different samples into a uniform batch.
        padded_radar = np.zeros((MAX_RADAR_POINTS, 4), dtype=np.float32)
        padded_radar[:N_points] = radar_pc

        # ------------------------------------------------------------------
        # 3. Convert to tensors
        # ------------------------------------------------------------------
        # (H, W, C) uint8 → (C, H, W) float32 in [0, 1]
        camera_tensor = (
            torch.from_numpy(camera_img).permute(2, 0, 1).float() / 255.0
        )
        radar_tensor = torch.from_numpy(padded_radar)

        return {
            "camera": camera_tensor,
            "radar": radar_tensor,
            "num_radar_points": N_points,
            "frame_id": self.camera_paths[idx],
        }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = EarlyFusionDataset(data_root=".", debug=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch in loader:
        print(
            f"Camera batch shape : {batch['camera'].shape}  "
            f"(B, C, H, W)"
        )
        print(
            f"Radar  batch shape : {batch['radar'].shape}   "
            f"(B, MAX_RADAR_POINTS, 4)"
        )
        print(f"Valid radar points : {batch['num_radar_points'].tolist()}")
        break
