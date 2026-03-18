import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class EarlyFusionDataset(Dataset):
    """
    EarlyFusionDataset: Dataloader for aligned Camera and 4D Radar data.
    This serves as a placeholder/baseline loader for testing Phase 1.
    """
    def __init__(self, data_root, split='train', debug=False):
        self.data_root = data_root
        self.split = split
        self.debug = debug
        
        # Placeholder lists for matched paths
        self.camera_paths = []
        self.radar_paths = []
        self.load_metadata()

    def load_metadata(self):
        """Mock method for loading metadata. Replace with actual dataset parsing (e.g., nuScenes)."""
        # For now, generate dummy data paths
        if self.debug:
            self.camera_paths = [f"camera_{i}.png" for i in range(10)]
            self.radar_paths = [f"radar_{i}.npy" for i in range(10)]
    
    def __len__(self):
        return len(self.camera_paths)
    
    def __getitem__(self, idx):
        # MOCK IMPLEMENTATION
        # Return dummy camera frames (noisy) and 4D radar point clouds.
        
        # 1. Dummy noisy image (representing low-light / salt-n-pepper noise)
        camera_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # add some noise
        noise = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        camera_img = cv2.addWeighted(camera_img, 0.5, noise, 0.5, 0)
        
        # 2. Dummy 4D radar data (N points x 4 features: x, y, z, velocity)
        # 4 features: x, y, z, doppler_velocity
        N_points = np.random.randint(50, 150)
        radar_pc = np.random.randn(N_points, 4) * 10
        
        # Apply transforms if needed
        camera_tensor = torch.from_numpy(camera_img).permute(2, 0, 1).float() / 255.0
        radar_tensor = torch.from_numpy(radar_pc).float()
        
        return {
            'camera': camera_tensor,
            'radar': radar_tensor,
            'frame_id': self.camera_paths[idx]
        }

if __name__ == "__main__":
    test_ds = EarlyFusionDataset(data_root=".", debug=True)
    test_loader = DataLoader(test_ds, batch_size=2)
    for batch in test_loader:
        print(f"Loaded batch - Camera: {batch['camera'].shape}, Radar PC List Length: {len(batch['radar'])}")
        break
