import numpy as np

def get_mock_calibration_matrices():
    """
    Returns mock intrinsic and extrinsic matrices for testing projection logic.
    In real usage, load from dataset calibration files.
    """
    # Camera Intrinsics (pinhole model: fx, fy, cx, cy) for a 1280x720 image
    K = np.array([
        [1000.0, 0.0,    640.0],
        [0.0,    1000.0, 360.0],
        [0.0,    0.0,    1.0]
    ])
    
    # Radar-to-Camera Extrinsics (Rotation and Translation)
    # Assume radar is slightly below and forward of the camera
    R = np.eye(3) # Identity for simple alignment
    T = np.array([[0.0], [-0.5], [1.0]]) # Translating y down 0.5m, z forward 1.0m
    
    return K, R, T

def project_radar_to_camera(radar_pc, K, R, T, img_shape=(720, 1280)):
    """
    Projects 4D Radar Point Cloud to 2D Camera image plane.
    Preserves Doppler velocity as metadata.
    
    Args:
        radar_pc (np.ndarray): Shape (N, 4) - contains [x, y, z, doppler_vel]
        K (np.ndarray): Camera intrinsic matrix (3, 3)
        R (np.ndarray): Rotation matrix from Radar to Camera (3, 3)
        T (np.ndarray): Translation vector from Radar to Camera (3, 1)
        img_shape (tuple): (height, width) of the target image
        
    Returns:
        valid_points_2d (np.ndarray): Shape (V, 2) - [u, v] pixel coordinates
        valid_metadata (np.ndarray): Shape (V, 2) - [depth_z, doppler_vel]
    """
    h, w = img_shape
    
    # Guard: empty point cloud
    if radar_pc.size == 0 or radar_pc.shape[0] == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=np.float64)
    
    # 1. Extract XYZ and velocity
    xyz = radar_pc[:, :3] # Shape (N, 3)
    velocity = radar_pc[:, 3] # Shape (N,)
    
    # 2. Transform from Radar Coordinate System to Camera Coordinate System
    # X_cam = R * X_rad + T
    xyz_cam = (R @ xyz.T) + T # Shape (3, N)
    
    # Ensure points are in front of the camera (Z > 0)
    front_mask = xyz_cam[2, :] > 0.0
    xyz_cam = xyz_cam[:, front_mask]
    velocity = velocity[front_mask]
    
    # 3. Project to Image Plane using Intrinsic Matrix
    # [u * z, v * z, z]^T = K * X_cam
    uvz = K @ xyz_cam # Shape (3, V)
    
    # Normalize by Depth (Z)
    u = uvz[0, :] / uvz[2, :]
    v = uvz[1, :] / uvz[2, :]
    depth = uvz[2, :]
    
    # 4. Filter out points outside the image dimensions
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    
    img_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    
    # 5. Extract Final Valid Data
    u_valid = u[img_mask]
    v_valid = v[img_mask]
    depth_valid = depth[img_mask]
    vel_valid = velocity[img_mask]
    
    if len(u_valid) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=np.float64)
    
    valid_points_2d = np.stack((u_valid, v_valid), axis=1)
    valid_metadata = np.stack((depth_valid, vel_valid), axis=1)
    
    return valid_points_2d, valid_metadata

if __name__ == "__main__":
    K, R, T = get_mock_calibration_matrices()
    
    # Mock Radar Data: 100 points, 4 features [x, y, z, velocity]
    N = 100
    mock_radar = np.random.randn(N, 4) * 5.0
    # Force z to be entirely positive and far enough ahead
    mock_radar[:, 2] = np.abs(mock_radar[:, 2]) + 5.0
    
    pts_2d, metadata = project_radar_to_camera(mock_radar, K, R, T)
    print(f"Projected {len(pts_2d)} out of {N} points onto the image plane.")
    if len(pts_2d) > 0:
        print(f"Sample 2D Point: {pts_2d[0]}")
        print(f"Sample Metadata (Z, Vel): {metadata[0]}")
