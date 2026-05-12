#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "====================================="
echo " ADAS Early Sensor Fusion Project "
echo "====================================="

echo "[1/3] Installing minimal dependencies..."
pip install -r requirements.txt

echo "[2/3] Generating mock data for simulation..."
# Generating small mock dataset to verify pipeline
python src/generate_mock_data.py

echo "[3/3] Running training and evaluation..."
# Train the dummy model and plot metrics
python src/train.py

echo "====================================="
echo " Run complete! Check the 'results/' folder for output graphs."
echo "====================================="
