# ADAS-Sensor: Advanced Driver Assistance Systems Simulation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **A High-Fidelity Sensor Simulation Framework for Autonomous Vehicle Research.**

ADAS-Sensor is a specialized Python framework designed to simulate the complex data streams produced by automotive sensors (Lidar, Radar, Ultrasonic). It provides a robust platform for testing ADAS algorithms (lane keeping, adaptive cruise control, collision avoidance) in a controlled virtual environment, reducing the need for costly and risky real-world trial runs.

## 🚗 The Vision: Safety Through Simulation
The development of autonomous systems requires millions of miles of testing. ADAS-Sensor enables rapid iteration of sensor-fusion and perception logic by providing precise control over environmental variables, edge-case scenarios, and sensor noise profiles.

## 🛠️ Key Features
- **Multi-Modal Simulation**: Support for various sensor types with customizable field-of-view, resolution, and range.
- **Dynamic Scenarios**: Scriptable traffic environments and obstacle behaviors for rigorous testing.
- **Data Export**: Seamless integration with perception pipelines via standard data formats (JSON, CSV, NumPy).
- **Config-Driven**: Define your vehicle's sensor suite entirely via YAML/JSON configuration files.

---

## 🏗️ Technical Architecture

- **Src/**: The core simulation engine, including sensor physics models and coordinate transformation logic.
- **Configs/**: Centralized machine and sensor definitions for reproducible experiments.
- **Automation Pipeline**: Designed to be integrated into CI/CD workflows for automated safety verification.

---

## 📂 Project Structure

- `src/`: Core Python source code for sensor models.
- `configs/`: YAML/JSON configuration templates.
- `Archives/`: Historical changelogs, legacy unit tests, and research documentation.
- `requirements.txt`: Minimal dependencies for local execution.

## 🚦 Quick Start

### Installation
```bash
git clone https://github.com/user/ADAS-Sensor
cd ADAS-Sensor
pip install -r requirements.txt
```

### Running a Simulation
```bash
# Start a basic radar simulation
python src/main.py --config configs/radar_default.yaml
```

## 📜 License
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
