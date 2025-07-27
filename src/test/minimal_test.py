import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MinimalDEM:
    def __init__(self, epsilon=0.05, device='cpu'):
        self.epsilon = epsilon
        self.device = device
        self.client_access_history = {}
        
    def generate_unlearnable_examples(self, images, targets, model):
        logger.info("Generate unlearnable examples...")
        
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.01)
        
        model.eval()
        for step in range(10):
            optimizer.zero_grad()
            
            perturbed = images + delta
            perturbed = torch.clamp(perturbed, 0, 1)
            
            outputs = model(perturbed)
            loss = -torch.nn.functional.cross_entropy(outputs, targets)  # Maximize losses
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        unlearnable_images = images + delta.detach()
        unlearnable_images = torch.clamp(unlearnable_images, 0, 1)
        
        return unlearnable_images, delta.detach()
    
    def save_client_access_history(self, filepath):
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.client_access_history, f, indent=2)
            logger.info(f"The visit history has been saved: {filepath}")
        except Exception as e:
            logger.error(f"fail to save: {e}")

def main():
    """Main test function"""

    
    # Create test data
    batch_size = 16
    images = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size,))
    
    # create models
    model = SimpleModel()
    
    # Create a DEM generator
    dem = MinimalDEM()
    
    # Generate unlearnable examples
    start_time = time.time()
    unlearnable_images, perturbations = dem.generate_unlearnable_examples(images, targets, model)
    end_time = time.time()
    
    # evaluation effect
    model.eval()
    with torch.no_grad():
        clean_outputs = model(images)
        unlearnable_outputs = model(unlearnable_images)
        
        clean_acc = (clean_outputs.argmax(1) == targets).float().mean()
        unlearnable_acc = (unlearnable_outputs.argmax(1) == targets).float().mean()
    
    dem.save_client_access_history('./test_output/client_history.json')
    
    print(f"processing time: {end_time - start_time:.2f}秒")
    print(f"Clean accuracy: {clean_acc:.3f}")
    print(f"accuracy: {unlearnable_acc:.3f}")
    print(f"accuracy rate has declined: {clean_acc - unlearnable_acc:.3f}")
    print(f"disturbance: {torch.norm(perturbations).item():.4f}")

if __name__ == "__main__":
    import os
    os.makedirs('./test_output', exist_ok=True)
    
    success = main()
    print(f"\nMinimize testing{'succeed' if success else 'finish'}")
