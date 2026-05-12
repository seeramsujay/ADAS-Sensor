import os
import numpy as np
import cv2

def generate_mock_dataset(base_path, dataset_name, num_samples=100):
    dataset_path = os.path.join(base_path, dataset_name)
    images_dir = os.path.join(dataset_path, "images")
    radar_dir = os.path.join(dataset_path, "radar")
    labels_dir = os.path.join(dataset_path, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Generating {num_samples} mock samples for {dataset_name} at {dataset_path}...")
    for i in range(num_samples):
        # Generate dummy image (1280x720, 3 channels)
        image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        # Add some "noise" simulating low light
        noise = np.random.normal(0, 50, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        cv2.imwrite(os.path.join(images_dir, f"{i:06d}.png"), image)

        # Generate dummy radar point cloud (N points, [x, y, z, velocity, RCS])
        num_points = np.random.randint(50, 300)
        radar_data = np.random.rand(num_points, 5).astype(np.float32)
        # Scale values to look realistic
        radar_data[:, 0] *= 100 # x distance
        radar_data[:, 1] = (radar_data[:, 1] - 0.5) * 40 # y distance
        radar_data[:, 2] = (radar_data[:, 2] - 0.5) * 10 # z height
        radar_data[:, 3] = (radar_data[:, 3] - 0.5) * 30 # doppler velocity
        radar_data[:, 4] *= 20 # RCS
        np.save(os.path.join(radar_dir, f"{i:06d}.npy"), radar_data)

        # Generate dummy labels (Bounding boxes: [x_center, y_center, width, height, class_id])
        num_boxes = np.random.randint(0, 5)
        labels = np.random.rand(num_boxes, 5)
        labels[:, 4] = np.random.randint(0, 3, num_boxes) # 3 classes: Car, Pedestrian, Cyclist
        np.savetxt(os.path.join(labels_dir, f"{i:06d}.txt"), labels)
    
    print(f"Finished generating {dataset_name}.")

if __name__ == "__main__":
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(base_data_dir, exist_ok=True)
    
    datasets = ["Waymo", "KITTI", "nuScenes", "RobotCar", "Astyx"]
    for ds in datasets:
        generate_mock_dataset(base_data_dir, ds, num_samples=20)
    
    print("All mock datasets generated successfully in", base_data_dir)
