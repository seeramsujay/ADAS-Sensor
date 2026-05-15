# Project Roadmap: Cross-Modal Denoising in Early Sensor Fusion

This roadmap outlines the step-by-step implementation of the early-fusion architecture, leveraging reciprocal radar-camera signal filtering for zero-lux ADAS perception.

## Phase 1: Project Initialization and Setup
**Objective:** Establish the development environment, baseline tools, and data infrastructure.
- [x] **Task 1.1 - Workspace Setup:** Initialize version control, define project structure, and determine the software stack (e.g., Python, PyTorch/TensorFlow, ROS 2, OpenCV).
- **Task 1.2 - Dataset Acquisition:** Acquire or simulate matched datasets containing both high-ISO/noisy Camera (RGB/NIR) and 4D Radar data in low light/adverse weather (e.g., nuScenes, RADIATE, or CARLA simulations).
- **Task 1.3 - Data Loaders:** Implement baseline data loading pipelines and visualize the raw streams to formally profile the noise characteristics (e.g., salt-and-pepper visual artifacts, radar multipath).

## Phase 2: Sensor Synchronization (Architecture Step 01)
**Objective:** Time-align data streams from the 4D radar and the camera.
- **Task 2.1 - Temporal Alignment:** Implement timestamp-matching algorithms to pair the closest radar sweeps with camera frames.
- **Task 2.2 - Rate Handling:** Handle different sampling rates (e.g., Radar at 10Hz vs. Camera at 30Hz) using interpolation or nearest-neighbor logic.
- **Task 2.3 - Synchronization Validation:** Plot the temporal offsets to ensure they remain within an acceptable hard-real-time threshold (e.g., `<10ms`). Create a unified data structure for synchronized `[Camera_Frame, Radar_Point_Cloud]` pairs.

## Phase 3: Spatial Mapping and Calibration (Architecture Step 02)
**Objective:** Accurately project the 4D radar point cloud onto the 2D camera pixel array.
- **Task 3.1 - Matrix Calibration:** Obtain and implement intrinsic and extrinsic calibration matrices for both sensors.
- **Task 3.2 - Projection Logic:** Implement the mathematical projection of 3D radar coordinates $(x, y, z)$ into 2D pixel coordinates $(u, v)$. 
- **Task 3.3 - 4D Metadata Attachment:** Ensure radar's 4th dimension (Doppler velocity) is preserved and attached as metadata to the projected pixel locations.
- **Task 3.4 - Visual Verification:** Visualize the projection by overlaying radar points onto the camera frame during daylight/ideal conditions to confirm sub-degree spatial alignment.

## Phase 4: Radar-Gated Spatial Filtering (Architecture Step 03)
**Objective:** Use sparse 4D radar data to dynamically "gate" and clean the noisy visual matrix.
- **Task 4.1 - Region Trust Modeling:** Define trusted pixel regions (spatial gates) around the projected radar points based on radar confidence.
- **Task 4.2 - Signal Amplification & Gating:** Apply spatial masking to the camera's image matrix: amplify and preserve pixels aligned with radar returns while aggressively suppressing/denoising unverified void regions.
- **Task 4.3 - SNR Benchmarking:** Measure the fused Signal-to-Noise Ratio (SNR) enhancement. Optimize the cross-modal intersection mathematical formula to minimize noise power. Verify the goal of **10× SNR enhancement** in total darkness.

## Phase 5: Visual Edge Detection (Architecture Step 04)
**Objective:** Extract structural boundaries from the visual data to understand dynamic geometry.
- **Task 5.1 - Low-Light Edge Extraction:** Implement edge-detection algorithms (e.g., Canny, Sobel, or ML-based extractors) optimized for low-contrast/low-light visual feeds.
- **Task 5.2 - Boundary Identification:** Focus on isolating static environmental geometries like lane lines, curbs, and structural boundaries. Output a visual edge-map or structural confidence map.

## Phase 6: Clutter Rejection via Reciprocal Loop (Architecture Step 05)
**Objective:** Remove radar multipath ghosts using visual edge geometric constraints.
- **Task 6.1 - Geometric Overlay:** Overlay the raw radar point cloud with the generated visual edge-map.
- **Task 6.2 - Impossible Geometry Heuristics:** Define mathematical logic to identify false positives. For instance, if a radar return suggests an object located physically *behind* a solid structural boundary or in an impossible void region, lower its confidence.
- **Task 6.3 - Multipath Filtering:** Mathematically reject the identified radar multipath clutter.
- **Task 6.4 - FPR Validation:** Test specifically in environments prone to multipath reflections (wet surfaces, metal barriers) to verify the targeted **20% reduction in false positive rates**.

## Phase 7: Raw Signal-Level Early Fusion 
**Objective:** Integrate the denoised data streams at the signal level rather than the decision level.
- **Task 7.1 - Early Fusion Architecture:** Develop a neural network frontend or mathematical fusion layer that concatenates the physically strengthened camera and radar streams.
- **Task 7.2 - Tensor Construction:** Create a unified, high-SNR composite tensor (combining amplified image features with validated radar depth/velocity).
- **Task 7.3 - Prevent Late Fusion Errors:** Ensure the pipeline maintains sub-degree spatial accuracy and avoids cumulative errors found in traditional object-level voting modules.

## Phase 8: Downstream Perception and Object Detection (Architecture Step 06)
**Objective:** Feed the high-SNR fused data into standard detection models.
- **Task 8.1 - Detector Integration:** Pass the early-fused tensor to an object detection network (e.g., YOLO, PointPillars, or a custom 3D bounding box detector).
- **Task 8.2 - Model Fine-Tuning:** Train or fine-tune the detector on this specific early-fused representation, optimizing for mean Average Precision (mAP).
- **Task 8.3 - Zero-Lux Evaluation:** Evaluate the end-to-end maintainance of mAP when the camera operates in complete darkness.

## Phase 9: System Evaluation & Deployment Optimization
**Objective:** Verify the system as a robust, power-efficient, zero-lux ADAS alternative.
- **Task 9.1 - Latency Profiling:** Conduct real-time latency testing to ensure the reciprocal gating loop is fast enough for autonomous navigation.
- **Task 9.2 - Power Efficiency Verification:** Benchmark power consumption to validate the elimination of active IR illuminators.
- **Task 9.3 - Final Reporting:** Produce a comprehensive evaluation report demonstrating the safety enhancement, cost reduction, and edge-case resilience.
