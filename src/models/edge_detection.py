import cv2
import numpy as np

def extract_structural_boundaries(image, low_threshold=50, high_threshold=150):
    """
    Extracts structural boundaries from a low-light / noisy camera image.
    Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) and Canny.
    
    Args:
        image (np.ndarray): Input camera image (H, W, C) or grayscale
        low_threshold (int): Lower bound for Canny hysteresis
        high_threshold (int): Upper bound for Canny hysteresis
        
    Returns:
        edge_map (np.ndarray): Binary edge map (H, W) where 255 is an edge
    """
    # 1. Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Enhance contrast using CLAHE (critical for low-light)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Apply Gaussian Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 4. Extract edges using Canny
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # 5. Dilation to make boundaries slightly thicker for geometric matching
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    
    return thick_edges

if __name__ == "__main__":
    # Create a mock dark image with a faint square "structure"
    h, w = 720, 1280
    dark_img = np.ones((h, w), dtype=np.uint8) * 10 
    
    # Add a slightly brighter square
    cv2.rectangle(dark_img, (400, 200), (800, 500), 30, 2)
    
    # Add salt and pepper noise
    noise = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
    dark_img[noise == 255] = 255
    dark_img[np.random.randint(0, 2, (h, w), dtype=np.uint8) == 1] = 0
    
    # Run extraction
    edge_map = extract_structural_boundaries(dark_img)
    
    # Count edge pixels
    edge_pixels = np.count_nonzero(edge_map)
    print(f"Detected {edge_pixels} edge pixels in the structure map.")
