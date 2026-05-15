# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-05

### Added
- **Early-Fusion Architecture**: Reciprocal radar-camera signal filtering pipeline.
- **Radar-Gated Spatial Filtering**: Dynamic masking and SNR enhancement for low-light camera frames.
- **Visual Edge-Detection**: Canny-based structure extraction for multipath rejection.
- **Clutter Rejection Heuristics**: Geometric logic for eliminating radar ghosts.
- **Simulation Pipeline**: `src/main.py` for end-to-end testing with synthetic data.
- **GPL-3.0 License**: Project is now open-source.

### Changed
- Refactored repository structure for professional release.
- Moved internal research documents and roadmaps to `Archives/`.
- Updated documentation with professional README and CHANGELOG.
- Enhanced `.gitignore` for Python environments and artifact isolation.

### Fixed
- Stabilized temporal synchronization between 10Hz radar and 30Hz camera streams.
- Improved calibration logic for sub-degree spatial alignment.
