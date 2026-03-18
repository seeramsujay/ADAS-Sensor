import torch
import torch.nn as nn
import numpy as np
import cv2

def construct_early_fusion_tensor(denoised_img, radar_pts_2d, radar_metadata):
    """
    Constructs a true signal-level early fusion tensor.
    Combines the cleaned camera image with dense/sparse radar feature channels.
    
    Args:
        denoised_img (np.ndarray): Denoised image from Phase 4 (H, W, C)
        radar_pts_2d (np.ndarray): Cleaned radar points (V, 2)
        radar_metadata (np.ndarray): Cleaned metadata [depth, velocity] (V, 2)
        
    Returns:
        fused_tensor (torch.Tensor): Shape (1, C+2, H, W) composite tensor
    """
    h, w = denoised_img.shape[:2]
    
    # 1. Image Channels
    if len(denoised_img.shape) == 2:
        img_tensor = torch.from_numpy(denoised_img).unsqueeze(0).float() / 255.0
    else:
        # BGR -> RGB and to Tensor
        rgb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 # (3, H, W)
        
    # 2. Radar Feature Channels (Depth and Velocity)
    depth_grid = np.zeros((h, w), dtype=np.float32)
    vel_grid = np.zeros((h, w), dtype=np.float32)
    
    # Map sparse radar data to dense image grid
    for (u, v), (z, vel) in zip(radar_pts_2d, radar_metadata):
        # Cast to Python int — OpenCV rejects numpy int types
        cv2.circle(depth_grid, (int(u), int(v)), 3, float(z), -1)
        cv2.circle(vel_grid, (int(u), int(v)), 3, float(vel), -1)
        
    depth_tensor = torch.from_numpy(depth_grid).unsqueeze(0) # (1, H, W)
    vel_tensor = torch.from_numpy(vel_grid).unsqueeze(0) # (1, H, W)
    
    # 3. Concatenate along channel dimension
    # Resulting tensor has (C + 2) channels
    fused_tensor = torch.cat([img_tensor, depth_tensor, vel_tensor], dim=0)
    
    # Add batch dimension for inference
    return fused_tensor.unsqueeze(0)

class SimpleEarlyFusionHead(nn.Module):
    """
    A baseline Neural Network frontend that could be placed before a standard detector (e.g., YOLO)
    to process the multi-modal input tensor (e.g., 5 channels instead of 3).
    """
    def __init__(self, in_channels=5, out_channels=3):
        super(SimpleEarlyFusionHead, self).__init__()
        # Condense 5 modalities back to 3 to easily plug into off-the-shelf pre-trained models
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=1) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    # Mock data
    mock_denoised = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_radar_pts = np.array([[640, 360], [100, 100]])
    mock_metadata = np.array([[50.0, 10.0], [20.0, -5.0]])
    
    fused_tensor = construct_early_fusion_tensor(mock_denoised, mock_radar_pts, mock_metadata)
    print(f"Constructed Early Fusion Tensor Shape: {fused_tensor.shape}")
    
    # Pass through fusion head
    fusion_head = SimpleEarlyFusionHead(in_channels=fused_tensor.shape[1], out_channels=3)
    out = fusion_head(fused_tensor)
    print(f"Shape after Simple Fusion Head (ready for standard detector): {out.shape}")
