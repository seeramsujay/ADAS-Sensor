import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.data_retriever import DataRetriever

class TestDataRetriever(unittest.TestCase):

    def setUp(self):
        self.target_dir = "./test_data"
        self.retriever = DataRetriever(self.target_dir)

    @patch("subprocess.run")
    def test_retrieve_nuscenes_mini_calls_aws_cli(self, mock_run):
        self.retriever.retrieve_nuscenes_mini()
        
        # Check if aws s3 sync was called with correct arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "aws")
        self.assertEqual(cmd[1], "s3")
        self.assertEqual(cmd[2], "sync")
        self.assertIn("--no-sign-request", cmd)
        self.assertIn("v1.0-mini*", cmd)

    @patch("subprocess.run")
    def test_retrieve_radial_calls_gdown(self, mock_run):
        self.retriever.retrieve_radial()
        
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gdown")
        self.assertEqual(cmd[2], "1OfqXXgoFg6xRYZkRqPJye4cQ29Fomh3l")
        self.assertIn("RADIal_Raw_ADC.zip", cmd[4])

    @patch("subprocess.run")
    def test_retrieve_k_radar_aux_calls_gdown_folder(self, mock_run):
        self.retriever.retrieve_k_radar_aux()
        
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gdown")
        self.assertEqual(cmd[1], "--folder")
        self.assertIn("1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_", cmd[2])

    def test_target_dir_creation(self):
        # Verify that the target directory is created
        if os.path.exists(self.target_dir):
            os.rmdir(self.target_dir)
        
        DataRetriever(self.target_dir)
        self.assertTrue(os.path.exists(self.target_dir))
        
        # Clean up
        os.rmdir(self.target_dir)

if __name__ == "__main__":
    unittest.main()
