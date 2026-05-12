#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================================="
echo " ADAS Early Sensor Fusion: Dataset Acquisition & Training "
echo "========================================================="

echo "[1/4] Installing dependencies (AWS CLI, Kaggle, etc.)..."
pip install -r requirements.txt
pip install awscli nuscenes-devkit kaggle

echo "[2/4] Downloading Datasets (Using Public Mirrors where possible)..."

# Create data directory
mkdir -p data
cd data

# --- KITTI Dataset (AWS Open Data Mirror - No Registration Required) ---
echo "--- KITTI Dataset ---"
# aws s3 cp --no-sign-request s3://avg-kitti/data_object_image_2.zip .
# aws s3 cp --no-sign-request s3://avg-kitti/data_object_velodyne.zip .
echo "Command to run: aws s3 cp --no-sign-request s3://avg-kitti/ [file] ."

# --- nuScenes (AWS Open Data Mirror - No Registration Required) ---
echo "--- nuScenes Dataset ---"
# aws s3 ls --no-sign-request s3://motional-nuscenes/
echo "Command to run: aws s3 cp --no-sign-request s3://motional-nuscenes/ [file] ."

# --- Waymo Open Dataset (Requires gcloud auth) ---
echo "--- Waymo Open Dataset ---"
echo "Note: You must run 'gcloud auth login' and have Waymo access."
# gsutil -m cp -r gs://waymo_open_dataset_v_2_0_1/training/camera_image .

# --- Astyx (via Kaggle) ---
echo "--- Astyx Dataset ---"
echo "Note: Requires kaggle.json in /root/.kaggle/"
# kaggle datasets download -d theoviel/astyx-hires2019-dataset

# --- Oxford Radar RobotCar ---
echo "--- Oxford Radar RobotCar ---"
echo "Manual registration is required for this dataset."
echo "Visit: https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets"

cd ..

echo ""
echo "========================================================="
echo " [!] IMPORTANT NOTE ON DISK SPACE [!]"
echo " These datasets total over 2TB. Google Colab provides ~70GB."
echo " I recommend downloading only small subsets or using Google Drive."
echo "========================================================="
echo ""

echo "[3/4] Preparing scripts..."
# python src/train.py --dataset_path ./data

echo "========================================================="
echo " Setup script finished. Check the echo commands above."
echo "========================================================="
