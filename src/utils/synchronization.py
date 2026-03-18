import numpy as np

def match_timestamps(camera_timestamps, radar_timestamps, max_delay_ms=10.0):
    """
    Match camera frames with the closest radar sweeps based on timestamps.
    This handles different sampling rates (e.g., Camera 30Hz, Radar 10Hz).
    
    Args:
        camera_timestamps (np.ndarray): Array of camera frame timestamps in ms.
        radar_timestamps (np.ndarray): Array of radar sweep timestamps in ms.
        max_delay_ms (float): Maximum acceptable time difference for a valid match.
        
    Returns:
        matches (list of tuple): List of (camera_idx, radar_idx) for valid matches.
        offsets (list of float): Time differences (camera_t - radar_t) for matches.
    """
    matches = []
    offsets = []
    
    for c_idx, c_time in enumerate(camera_timestamps):
        # Find the absolute time differences between this camera frame and all radar sweeps
        time_diffs = np.abs(radar_timestamps - c_time)
        
        # Find the index of the closest radar sweep
        best_r_idx = np.argmin(time_diffs)
        best_diff = time_diffs[best_r_idx]
        
        # Check against hard real-time threshold
        if best_diff <= max_delay_ms:
            matches.append((c_idx, best_r_idx))
            # Actual offset (positive if camera is ahead)
            offsets.append(c_time - radar_timestamps[best_r_idx])
            
    return matches, offsets

def validate_synchronization(offsets, max_delay_ms=10.0):
    """
    Validate that the synchronization meets real-time constraints.
    
    Args:
        offsets (list of float): Time differences from matches.
        max_delay_ms (float): The threshold for validation.
        
    Returns:
        bool: True if all offsets are within threshold, False otherwise.
        dict: Statistics including max delay, mean delay, and valid match count.
    """
    if not offsets:
        return False, {"max_delay": None, "mean_delay": None, "valid_count": 0}
        
    abs_offsets = np.abs(offsets)
    max_offset = np.max(abs_offsets)
    mean_offset = np.mean(abs_offsets)
    valid = bool(max_offset <= max_delay_ms)
    
    stats = {
        "max_delay_ms": max_offset,
        "mean_delay_ms": mean_offset,
        "valid_count": len(offsets)
    }
    
    return valid, stats

if __name__ == "__main__":
    # Test cases:
    # Camera at 30 FPS (~33ms apart), Radar at 10 FPS (100ms apart)
    cam_time = np.arange(0, 1000, 33.3)
    rad_time = np.arange(0, 1000, 100.0)
    
    # Intentionally add some jitter to radar
    rad_time += np.random.randn(len(rad_time)) * 2.0
    
    matches, offsets = match_timestamps(cam_time, rad_time, max_delay_ms=15.0)
    is_valid, stats = validate_synchronization(offsets, max_delay_ms=15.0)
    
    print(f"Total Camera Frames: {len(cam_time)}")
    print(f"Total Radar Sweeps: {len(rad_time)}")
    print(f"Matched Pairs: {len(matches)}")
    print(f"Synchronization Valid: {is_valid}")
    print(f"Delay Stats: {stats}")
