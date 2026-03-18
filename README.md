# Cross-Modal Denoising in Early Sensor Fusion

> Enhancing Low-Light ADAS Reliability via Radar-Gated Spatial Filtering

## Overview

This project implements a novel **early-fusion architecture** for ADAS that addresses sensor degradation in **zero-lux, fog, and adverse weather** conditions. Instead of filtering sensors independently (late fusion), this system performs **reciprocal cross-modal denoising** at the raw signal level:

- **Radar → Camera**: Sparse 4D radar data dynamically "gates" and cleans noisy visual matrices in total darkness.
- **Camera → Radar**: Visual edge-detection identifies and rejects radar multipath clutter (ghost reflections).

The result is a high-SNR fused representation that feeds directly into downstream object detection.

## Architecture

```
Camera (noisy) ──┐                              ┌── Object Detection
                  ├─ Sync ─ Project ─ Gate ─ Fuse ┤
4D Radar ─────────┘     ↑               ↑        └── Bounding Boxes
                        │               │
                  Edge Detection   Clutter Rejection
                  (Camera→Radar)   (Radar→Camera)
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Data Loading | `src/data/loader.py` | Loads paired camera frames and 4D radar point clouds |
| 2. Synchronization | `src/utils/synchronization.py` | Timestamp matching across different sensor rates |
| 3. Spatial Mapping | `src/utils/calibration.py` | Projects 3D radar (x,y,z) → 2D pixels (u,v) with Doppler metadata |
| 4. Spatial Filtering | `src/models/spatial_filtering.py` | Radar-gated masking: amplify trusted regions, suppress noise |
| 5. Edge Detection | `src/models/edge_detection.py` | CLAHE + Canny extraction of structural boundaries in low-light |
| 6. Clutter Rejection | `src/models/clutter_rejection.py` | Geometric heuristics to reject multipath radar ghosts |
| 7. Early Fusion | `src/models/early_fusion.py` | Constructs 5-channel tensor (RGB + Depth + Velocity) |
| 8. Detection Head | `src/models/early_fusion.py` | Lightweight conv head to adapt fused input for standard detectors |

## Project Structure

```
ADAS-Sensor/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py              # Dataset and DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spatial_filtering.py   # Radar-gated denoising + SNR
│   │   ├── edge_detection.py      # Low-light boundary extraction
│   │   ├── clutter_rejection.py   # Multipath ghost filtering
│   │   └── early_fusion.py        # Tensor construction + fusion head
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── synchronization.py     # Temporal alignment
│   │   └── calibration.py         # Projection matrices
│   ├── __init__.py
│   └── main.py                    # End-to-end pipeline entry point
├── configs/
├── tests/
│   └── __init__.py
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd ADAS-Sensor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Pipeline (Mock Data)

```bash
PYTHONPATH=. python src/main.py
```

This runs the full pipeline on synthetic noisy camera frames and random radar point clouds, printing SNR gains, clutter rejection stats, and per-frame latency.

### 3. Run Individual Modules

Each module has a `__main__` block for standalone testing:

```bash
PYTHONPATH=. python src/utils/synchronization.py   # Test timestamp matching
PYTHONPATH=. python src/utils/calibration.py        # Test radar→camera projection
PYTHONPATH=. python src/models/spatial_filtering.py  # Test gating + SNR
PYTHONPATH=. python src/models/edge_detection.py     # Test edge extraction
PYTHONPATH=. python src/models/clutter_rejection.py  # Test ghost rejection
PYTHONPATH=. python src/models/early_fusion.py       # Test tensor construction
```

## Key Metrics

| Metric | Target |
|--------|--------|
| SNR Enhancement | 10× improvement through reciprocal filtering |
| False Positive Reduction | 20% lower FPR vs. late fusion |
| Operational Range | Maintains mAP at 0-lux illumination |

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- NumPy, SciPy, Matplotlib, Pandas

## License

This project is for academic and research purposes.
