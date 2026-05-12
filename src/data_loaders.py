import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# Note: nuScenes usually requires 'nuscenes-devkit'
# pip install nuscenes-devkit

class KITTIDataset(Dataset):
    """
    Loader for KITTI Object Detection Dataset.
    Structure:
    - image_2/*.png
    - velodyne/*.bin
    - label_2/*.txt
    """
    def __init__(self, root_dir, split='training'):
        self.root_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root_dir, "image_2")
        self.lidar_dir = os.path.join(self.root_dir, "velodyne")
        self.label_dir = os.path.join(self.root_dir, "label_2")
        
        self.sample_ids = [f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.image_dir, f"{sample_id}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 360))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 2. Load LiDAR (KITTI velodyne is .bin)
        # We treat LiDAR as Radar-like for early fusion tests
        lidar_path = os.path.join(self.lidar_dir, f"{sample_id}.bin")
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4) # [x, y, z, intensity]
        
        # Pad/Truncate
        max_points = 1000
        if pc.shape[0] < max_points:
            padding = np.zeros((max_points - pc.shape[0], 4), dtype=np.float32)
            pc = np.vstack((pc, padding))
        else:
            pc = pc[:max_points, :]
        pc_tensor = torch.from_numpy(pc)

        return image, pc_tensor

class NuScenesDataset(Dataset):
    """
    Placeholder for nuScenes. 
    In practice, use: from nuscenes.nuscenes import NuScenes
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Logic to use nuscenes-devkit would go here

    def __len__(self):
        return 0 # Requires real nuScenes setup

class ADASDataset(Dataset):
    """
    The original Generic Dataset (kept for mock data compatibility)
    """
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
        
        return image, radar_tensor

def get_dataloader(dataset_type, root_dir, batch_size=4):
    if dataset_type == 'KITTI':
        dataset = KITTIDataset(root_dir)
    elif dataset_type == 'nuScenes':
        dataset = NuScenesDataset(root_dir)
    else:
        dataset = ADASDataset(root_dir)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
