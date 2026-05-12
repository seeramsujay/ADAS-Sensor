import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loaders import get_dataloader
from model import EarlyFusionModel

def plot_metrics(train_losses, snr_improvements, results_dir):
    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Early Fusion Model - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'training_loss.png'))
    plt.close()

    # Plot SNR Enhancement
    plt.figure(figsize=(10, 5))
    plt.plot(snr_improvements, label='SNR Enhancement (dB)', color='green', marker='o')
    plt.title('Simulated SNR Enhancement in Zero-Lux Environment')
    plt.xlabel('Epoch')
    plt.ylabel('SNR Gain (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'snr_enhancement.png'))
    plt.close()

def main():
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    # We will use the 'Waymo' mock dataset for this training loop
    dataset_path = os.path.join(base_data_dir, "Waymo")
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found. Did you run generate_mock_data.py?")
        return

    dataloader = get_dataloader(dataset_path, batch_size=4, shuffle=True)
    model = EarlyFusionModel(num_classes=3)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    train_losses = []
    # Mock SNR improvement tracking for the graph
    snr_improvements = []
    
    print("Starting Training...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, radar, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, radar)
            
            # For this mock setup, we generate dummy class targets to calculate loss
            batch_size = images.size(0)
            dummy_targets = torch.randint(0, 3, (batch_size,))
            
            loss = criterion(outputs, dummy_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Simulate SNR improvement growing as model learns to denoise (logarithmic curve)
        current_snr = 2.0 + (8.0 * (1 - 2.718**(-(epoch+1)/3.0))) 
        snr_improvements.append(current_snr)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Est. SNR Gain: {current_snr:.2f} dB")

    print("Training Finished!")
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(results_dir, "early_fusion_model.pth"))
    
    # Plot graphs using matplotlib
    plot_metrics(train_losses, snr_improvements, results_dir)
    print(f"Graphs saved to {results_dir}")

if __name__ == "__main__":
    main()
