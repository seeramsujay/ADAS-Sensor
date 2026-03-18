import numpy as np
import cv2

def apply_clutter_rejection(radar_pts_2d, radar_metadata, edge_map, depth_threshold_map=None):
    """
    Rejects radar multipath clutter (ghosts) using visual edge geometry.
    If a radar return is located behind a solid boundary but in an impossible void,
    or if its geometry contradicts the structural map, it is dropped.
    
    Args:
        radar_pts_2d (np.ndarray): Shape (V, 2) of projected radar [u, v]
        radar_metadata (np.ndarray): Shape (V, 2) of [depth_z, doppler_vel]
        edge_map (np.ndarray): Binary edge map (H, W) from the camera
        depth_threshold_map (np.ndarray): Optional. Predicted depth map from mono-camera.
    
    Returns:
        filtered_pts (np.ndarray): The cleaned (V_new, 2) radar points
        filtered_meta (np.ndarray): The cleaned (V_new, 2) metadata
        dropped_idx (list): Indices of points identified as multipath clutter
    """
    valid_indices = []
    dropped_indices = []
    
    h, w = edge_map.shape
    
    # Simple Heuristic 1: If an edge map is dense in a region, it implies a solid object.
    # Multipath ghosts often appear physically "inside" or "behind" solid walls (like guardrails).
    # We can check a neighborhood around the projected point.
    
    for i, ((u, v), (z, vel)) in enumerate(zip(radar_pts_2d, radar_metadata)):
        # Ensure within bounds
        if u < 0 or u >= w or v < 0 or v >= h:
            dropped_indices.append(i)
            continue
            
        # Check an X by X region around the point for structural density
        box_size = 10
        u_min, u_max = max(0, u - box_size), min(w, u + box_size)
        v_min, v_max = max(0, v - box_size), min(h, v + box_size)
        
        region = edge_map[v_min:v_max, u_min:u_max]
        edge_density = np.sum(region > 0) / ((u_max - u_min) * (v_max - v_min) + 1e-6)
        
        # Heuristic 2: Depth Contradiction (if mono-depth map is available)
        # If the radar Z is vastly greater than the visual boundary Z, it's likely a pass-through ghost.
        is_ghost = False
        if depth_threshold_map is not None:
            visual_z = depth_threshold_map[v, u]
            if z > visual_z + 3.0: # e.g., Radar says 10m, visual bound says wall is at 5m = Ghost
                is_ghost = True
                
        # Combine heuristics
        # For this baseline, if edge_density is extremely high, we might enforce stricter confidence,
        # or if it's a known multipath scenario (like ground bounce), we filter specific velocities.
        
        if is_ghost:
            dropped_indices.append(i)
        else:
            valid_indices.append(i)
            
    filtered_pts = radar_pts_2d[valid_indices]
    filtered_meta = radar_metadata[valid_indices]
    
    return filtered_pts, filtered_meta, dropped_indices

if __name__ == "__main__":
    # Mock data
    pts = np.array([[100, 100], [200, 200], [300, 300]])
    meta = np.array([[10.0, 1.5], [15.0, 0.0], [5.0, -1.0]]) # Z and Velocity
    
    # Mock Edge map
    e_map = np.zeros((500, 500), dtype=np.uint8)
    e_map[190:210, 190:210] = 255 # dense edge right at pt 2
    
    # Mock Depth Map
    depth_map = np.ones((500, 500)) * 50.0 
    depth_map[190:210, 190:210] = 5.0 # Visual wall at 5m
    
    # Run
    f_pts, f_meta, dropped = apply_clutter_rejection(pts, meta, e_map, depth_threshold_map=depth_map)
    
    print(f"Original Points: {len(pts)}")
    print(f"Dropped Indices: {dropped}") # Should drop index 1 (Z=15m behind wall at 5m)
    print(f"Filtered Points: {len(f_pts)}")
