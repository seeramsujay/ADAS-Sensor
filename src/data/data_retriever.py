import os
import subprocess
import argparse
import sys
from typing import List

class DataRetriever:
    """
    Automates the retrieval of autonomous driving datasets as described in Report.md.
    Targeted for use in headless Linux environments like Google Colab.
    """

    DATASETS = {
        "nuscenes-mini": {
            "type": "aws",
            "bucket": "s3://motional-nuscenes/public/v1.0/",
            "include": "v1.0-mini*",
            "description": "nuScenes v1.0-mini subset (~4GB)"
        },
        "radial": {
            "type": "gdrive",
            "id": "1OfqXXgoFg6xRYZkRqPJye4cQ29Fomh3l",
            "output": "RADIal_Raw_ADC.zip",
            "description": "RADIal Raw ADC Radar Dataset (~25GB)"
        },
        "k-radar-aux": {
            "type": "gdrive",
            "id": "1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_",
            "is_folder": True,
            "description": "K-Radar Auxiliary Data (Camera, LiDAR, GPS) - Folder"
        }
    }

    def __init__(self, target_dir: str):
        self.target_dir = os.path.abspath(target_dir)
        os.makedirs(self.target_dir, exist_ok=True)

    def _run_command(self, cmd: List[str]):
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}", file=sys.stderr)
            raise

    def retrieve_nuscenes_mini(self):
        print("Retrieving nuScenes-mini via AWS CLI...")
        # Ensure aws cli is installed and --no-sign-request is used
        cmd = [
            "aws", "s3", "sync",
            self.DATASETS["nuscenes-mini"]["bucket"],
            os.path.join(self.target_dir, "nuscenes"),
            "--no-sign-request",
            "--exclude", "*",
            "--include", self.DATASETS["nuscenes-mini"]["include"]
        ]
        self._run_command(cmd)

    def retrieve_radial(self):
        print("Retrieving RADIal via gdown...")
        output_path = os.path.join(self.target_dir, self.DATASETS["radial"]["output"])
        cmd = [
            "gdown",
            "--id", self.DATASETS["radial"]["id"],
            "-O", output_path
        ]
        self._run_command(cmd)

    def retrieve_k_radar_aux(self):
        print("Retrieving K-Radar auxiliary data via gdown...")
        # gdown --folder for directory links
        cmd = [
            "gdown",
            "--folder",
            f"https://drive.google.com/drive/folders/{self.DATASETS['k-radar-aux']['id']}",
            "-O", os.path.join(self.target_dir, "k-radar-aux")
        ]
        self._run_command(cmd)

def main():
    parser = argparse.ArgumentParser(description="ADAS Sensor Data Retriever")
    parser.add_argument("--dataset", choices=["nuscenes-mini", "radial", "k-radar-aux", "all"], 
                        required=True, help="Dataset to retrieve")
    parser.add_argument("--target_dir", default="./data", help="Directory to save data")
    
    args = parser.parse_args()
    retriever = DataRetriever(args.target_dir)

    if args.dataset == "nuscenes-mini" or args.dataset == "all":
        retriever.retrieve_nuscenes_mini()
    if args.dataset == "radial" or args.dataset == "all":
        retriever.retrieve_radial()
    if args.dataset == "k-radar-aux" or args.dataset == "all":
        retriever.retrieve_k_radar_aux()

if __name__ == "__main__":
    main()
