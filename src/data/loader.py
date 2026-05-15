"""
loader.py
=========
Dataset and DataLoader utilities for the Cross-Modal Denoising Early Fusion pipeline.

This module defines `EarlyFusionDataset`, a PyTorch-compatible ``Dataset``
that loads (or, in debug/mock mode, synthesises) temporally synchronised pairs of:

- **Camera frames** – represented as high-ISO noisy RGB tensors that simulate
  zero-lux or adverse-weather conditions.
- **4D Radar point clouds** – padded to a fixed maximum length so that
  PyTorch's default collate function can batch samples without errors.

It also contains loaders for standard datasets (KITTI, nuScenes).
"""

import os
import glob
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RADAR_POINTS: int = 150
MOCK_IMG_SHAPE: tuple = (720, 1280)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class EarlyFusionDataset(Dataset):
    """PyTorch Dataset for paired Camera and 4D Radar sensor streams.
    In debug mode, synthetic noisy images and random point clouds are generated.
    """
    def __init__(self, data_root: str, split: str = "train", debug: bool = False) -> None:
        self.data_root: str = data_root
        self.split: str = split
        self.debug: bool = debug

        self.camera_paths: List[str] = []
        self.radar_paths: List[str] = []
        self.load_metadata()

    def load_metadata(self) -> None:
        if self.debug:
            self.camera_paths = [f"camera_{i}.png" for i in range(10)]
            self.radar_paths = [f"radar_{i}.npy" for i in range(10)]

    def __len__(self) -> int:
        return len(self.camera_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        camera_img = np.zeros((*MOCK_IMG_SHAPE, 3), dtype=np.uint8)
        noise = np.random.randint(0, 256, (*MOCK_IMG_SHAPE, 3), dtype=np.uint8)
        camera_img = cv2.addWeighted(camera_img, 0.5, noise, 0.5, 0)

        N_points = np.random.randint(50, MAX_RADAR_POINTS)
        radar_pc = (np.random.randn(N_points, 4) * 10).astype(np.float32)

        padded_radar = np.zeros((MAX_RADAR_POINTS, 4), dtype=np.float32)
        padded_radar[:N_points] = radar_pc

        camera_tensor = (torch.from_numpy(camera_img).permute(2, 0, 1).float() / 255.0)
        radar_tensor = torch.from_numpy(padded_radar)

        return {
            "camera": camera_tensor,
            "radar": radar_tensor,
            "num_radar_points": N_points,
            "frame_id": self.camera_paths[idx],
        }

class KITTIDataset(Dataset):
    """Loader for KITTI Object Detection Dataset."""
    def __init__(self, root_dir, split='training'):
        self.root_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root_dir, "image_2")
        self.lidar_dir = os.path.join(self.root_dir, "velodyne")
        self.label_dir = os.path.join(self.root_dir, "label_2")
        
        if os.path.exists(self.image_dir):
            self.sample_ids = [f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.png')]
        else:
            self.sample_ids = []

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        img_path = os.path.join(self.image_dir, f"{sample_id}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 360))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        lidar_path = os.path.join(self.lidar_dir, f"{sample_id}.bin")
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        max_points = 1000
        if pc.shape[0] < max_points:
            padding = np.zeros((max_points - pc.shape[0], 4), dtype=np.float32)
            pc = np.vstack((pc, padding))
        else:
            pc = pc[:max_points, :]
        pc_tensor = torch.from_numpy(pc)

        # For compatibility with EarlyFusion format:
        return {
            "camera": image,
            "radar": pc_tensor,
            "num_radar_points": pc.shape[0],
            "frame_id": sample_id
        }

class NuScenesDataset(Dataset):
    """Placeholder for nuScenes."""
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return 0

class ADASDataset(Dataset):
    """The original Generic Dataset (kept for mock data compatibility)"""
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_paths = sorted(glob.glob(os.path.join(dataset_path, "images", "*.png")))
        self.radar_paths = sorted(glob.glob(os.path.join(dataset_path, "radar", "*.npy")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 360))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        radar = np.load(self.radar_paths[idx])
        radar_tensor = torch.from_numpy(radar).float()
        
        return {
            "camera": image,
            "radar": radar_tensor,
            "num_radar_points": radar.shape[0],
            "frame_id": os.path.basename(self.image_paths[idx])
        }

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dataloader(dataset_type, root_dir, batch_size=4, debug=False, shuffle=True):
    if debug and dataset_type == 'EarlyFusion':
        dataset = EarlyFusionDataset(data_root=root_dir, debug=True)
    elif dataset_type == 'KITTI':
        dataset = KITTIDataset(root_dir)
    elif dataset_type == 'nuScenes':
        dataset = NuScenesDataset(root_dir)
    elif dataset_type == 'EarlyFusion':
        dataset = EarlyFusionDataset(data_root=root_dir, debug=False)
    else:
        dataset = ADASDataset(root_dir)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    dataset = EarlyFusionDataset(data_root=".", debug=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in loader:
        print(f"Camera batch shape : {batch['camera'].shape}")
        print(f"Radar  batch shape : {batch['radar'].shape}")
        break
