import numpy as np
import cv2

def generate_spatial_gate(img_shape, projected_radar_pts, gate_radius=15):
    """
    Creates a spatial gate (mask) based on radar confident points.
    
    Args:
        img_shape (tuple): (height, width) of the image
        projected_radar_pts (np.ndarray): Shape (V, 2) of (u, v) valid coordinates
        gate_radius (int): Radius of the trusted region around each radar point
        
    Returns:
        gate_mask (np.ndarray): Binary mask where 1 indicates trusted region
    """
    gate_mask = np.zeros(img_shape, dtype=np.float32)
    
    for (u, v) in projected_radar_pts:
        cv2.circle(gate_mask, (u, v), gate_radius, 1.0, -1)
        
    # Optional: Apply Gaussian Blur to create soft gates
    gate_mask = cv2.GaussianBlur(gate_mask, (15, 15), 5.0)
    return np.clip(gate_mask, 0.0, 1.0)

def apply_radar_gating(noisy_img, gate_mask, amplification_factor=1.5, suppression_factor=0.2):
    """
    Applies the radar-guided spatial gate to the camera image.
    Amplifies signal in the trusted radar regions, suppresses noise in voids.
    
    Args:
        noisy_img (np.ndarray): The raw noisy camera frame (H, W, C)
        gate_mask (np.ndarray): The 2D spatial gate mask (H, W)
        amplification_factor (float): Multiplier for trusted regions
        suppression_factor (float): Multiplier for untrusted void regions
        
    Returns:
        denoised_img (np.ndarray): The gated and filtered image
    """
    # Expand mask to match image channels if it's RGB
    if len(noisy_img.shape) == 3:
        gate_mask = np.expand_dims(gate_mask, axis=-1)
        
    img_float = noisy_img.astype(np.float32)
    
    # 1. Amplify trusted regions
    amplified = img_float * amplification_factor
    
    # 2. Suppress background regions
    suppressed = img_float * suppression_factor
    
    # 3. Blend using the soft gate mask
    fused_img = (gate_mask * amplified) + ((1.0 - gate_mask) * suppressed)
    
    return np.clip(fused_img, 0, 255).astype(np.uint8)

def calculate_snr(image, signal_mask):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in dB.
    Assuming the 'signal' is in the masked region and 'noise' is outside.
    """
    image_float = image.astype(np.float32)
    
    if len(image.shape) == 3:
        signal_mask = np.expand_dims(signal_mask, axis=-1)
        
    # Extract signal pixels
    signal_pixels = image_float[signal_mask > 0.5]
    if len(signal_pixels) == 0:
        return 0.0
        
    noise_pixels = image_float[signal_mask <= 0.5]
    if len(noise_pixels) == 0:
        return float('inf')
        
    signal_power = np.mean(signal_pixels ** 2)
    noise_power = np.mean(noise_pixels ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

if __name__ == "__main__":
    # Mock parameters
    h, w = 720, 1280
    mock_noisy_img = np.random.randint(0, 100, (h, w, 3), dtype=np.uint8)
    
    # Inject "signal" into a specific spot
    u, v = 640, 360
    cv2.circle(mock_noisy_img, (u, v), 30, (200, 200, 200), -1)
    
    # Mock radar points hitting the signal
    pts = np.array([[u, v], [u+5, v-5]])
    
    gate = generate_spatial_gate((h, w), pts, gate_radius=40)
    denoised = apply_radar_gating(mock_noisy_img, gate)
    
    snr_before = calculate_snr(mock_noisy_img, gate)
    snr_after = calculate_snr(denoised, gate)
    
    print(f"SNR Before Gating: {snr_before:.2f} dB")
    print(f"SNR After Gating: {snr_after:.2f} dB")
    print(f"Improvement: {(snr_after - snr_before):.2f} dB")
