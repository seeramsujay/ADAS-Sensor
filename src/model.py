import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.mlp1 = nn.Linear(5, 32)
        self.mlp2 = nn.Linear(32, 64)
        self.mlp3 = nn.Linear(64, 128)

    def forward(self, x):
        # x shape: [Batch, N, 5]
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
