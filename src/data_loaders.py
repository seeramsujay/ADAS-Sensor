import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class ADASDataset(Dataset):
    """
    Generic PyTorch Dataset for loading Camera + Radar data.
    Works for the mock data structure generated for Waymo, KITTI, nuScenes, etc.
    """
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(dataset_path, "images", "*.png")))
        self.radar_paths = sorted(glob.glob(os.path.join(dataset_path, "radar", "*.npy")))
        self.label_paths = sorted(glob.glob(os.path.join(dataset_path, "labels", "*.txt")))

        assert len(self.image_paths) == len(self.radar_paths) == len(self.label_paths), \
            "Mismatch in number of images, radar files, and labels."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to a standard size for the network
        image = cv2.resize(image, (640, 360))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 2. Load Radar Point Cloud (N, 5: x, y, z, v, rcs)
        radar_pc = np.load(self.radar_paths[idx])
        # Pad or truncate radar to a fixed number of points (e.g., 200) for batching
        max_points = 200
        if radar_pc.shape[0] < max_points:
            padding = np.zeros((max_points - radar_pc.shape[0], 5), dtype=np.float32)
            radar_pc = np.vstack((radar_pc, padding))
        else:
            radar_pc = radar_pc[:max_points, :]
        radar_tensor = torch.from_numpy(radar_pc)

        # 3. Load Labels (Bounding boxes)
        # Using a fixed size for labels to allow batching, format: [num_boxes, 5]
        max_boxes = 10
        labels_arr = np.zeros((max_boxes, 5), dtype=np.float32)
        try:
            loaded_labels = np.loadtxt(self.label_paths[idx])
            if len(loaded_labels.shape) == 1 and loaded_labels.shape[0] > 0:
                loaded_labels = loaded_labels.reshape(1, -1)
            num_boxes = min(max_boxes, loaded_labels.shape[0])
            if num_boxes > 0:
                labels_arr[:num_boxes, :] = loaded_labels[:num_boxes, :]
        except Exception:
            pass # Empty file or parsing error
            
        labels_tensor = torch.from_numpy(labels_arr)

        return image, radar_tensor, labels_tensor

def get_dataloader(dataset_path, batch_size=4, shuffle=True):
    dataset = ADASDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
