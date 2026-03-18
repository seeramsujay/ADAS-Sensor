import time
import numpy as np

from src.data.loader import EarlyFusionDataset
from src.utils.calibration import get_mock_calibration_matrices, project_radar_to_camera
from src.models.spatial_filtering import generate_spatial_gate, apply_radar_gating, calculate_snr
from src.models.edge_detection import extract_structural_boundaries
from src.models.clutter_rejection import apply_clutter_rejection
from src.models.early_fusion import construct_early_fusion_tensor, SimpleEarlyFusionHead


def run_pipeline():
    print("=== Initializing Cross-Modal Denoising ADAS Pipeline ===")
    
    # 1. Setup & Loaders (Phase 1)
    dataset = EarlyFusionDataset(data_root=".", debug=True)
    K, R, T = get_mock_calibration_matrices()
    fusion_head = SimpleEarlyFusionHead(in_channels=5, out_channels=3)
    
    total_time = 0.0
    
    for i in range(1): # Run only 1 frame for demo
        print(f"\n--- Processing Frame {i} ---")
        start_t = time.time()
        
        # Simulated raw inputs
        data = dataset[i]
        camera_frame = data['camera'].permute(1, 2, 0).numpy() * 255.0 # (H, W, 3)
        camera_frame = camera_frame.astype(np.uint8)
        radar_pc = data['radar'].numpy() # (N, 4)
        
        # 2. Sensor Synchronization
        # Handled by dataloader pairing in this mock, but we assume timestamps are matched
        
        # 3. Spatial Mapping (Phase 3)
        pts_2d, metadata = project_radar_to_camera(radar_pc, K, R, T, img_shape=(720, 1280))
        print(f"Projected Radar Points: {len(pts_2d)}")
        
        if len(pts_2d) == 0:
            print("No valid radar points in view. Skipping advanced fusion.")
            continue
            
        # 4. Radar-Gated Spatial Filtering (Phase 4)
        gate_mask = generate_spatial_gate((720, 1280), pts_2d, gate_radius=20)
        denoised_img = apply_radar_gating(camera_frame, gate_mask, amplification_factor=1.5, suppression_factor=0.2)
        
        snr_before = calculate_snr(camera_frame, gate_mask)
        snr_after = calculate_snr(denoised_img, gate_mask)
        print(f"SNR Denoising: Before={snr_before:.2f}dB | After={snr_after:.2f}dB | Gain={snr_after-snr_before:.2f}dB")
        
        # 5. Visual Edge Detection (Phase 5)
        edge_map = extract_structural_boundaries(camera_frame)
        print(f"Extracted {np.count_nonzero(edge_map)} structural edge pixels.")
        
        # 6. Clutter Rejection via Reciprocal Loop (Phase 6)
        clean_pts, clean_meta, dropped = apply_clutter_rejection(pts_2d, metadata, edge_map)
        print(f"Multipath Clutter Filtered: {len(dropped)} ghost points dropped.")
        
        # 7. Raw Signal-Level Early Fusion (Phase 7)
        early_fused_tensor = construct_early_fusion_tensor(denoised_img, clean_pts, clean_meta)
        print(f"Constructed Fused Tensor (Batch, Channels, H, W): {early_fused_tensor.shape}")
        
        # 8. Downstream Perception Entry (Phase 8)
        # Process through the minimal fusion head before sending to a generic detector
        detector_input = fusion_head(early_fused_tensor)
        print(f"Processed detector input tensor shape: {detector_input.shape} (ready for YOLO/PointPillars)")
        
        end_t = time.time()
        frame_time = (end_t - start_t) * 1000.0 # ms
        total_time += frame_time
        print(f"Pipeline Latency for Frame {i}: {frame_time:.2f} ms")
        

if __name__ == "__main__":
    run_pipeline()
