import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import time
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastDEM:
    
    def __init__(self, epsilon=0.1, device='cpu'):
        self.epsilon = epsilon
        self.device = device
        self.models = []
       
        for i in range(3):
            model = self._create_fast_model()
            model.to(device)
            self.models.append(model)
    
    def _create_fast_model(self):
     
        class FastNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                self.fc = nn.Sequential(
                    nn.Linear(64 * 4 * 4, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return FastNet()
    
    def fast_train(self, data_loader):
       
        accuracies = []
        
        for i, model in enumerate(self.models):
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 5:
                    break
                    
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total if total > 0 else 0
            accuracies.append(acc)
            logger.info(f"{i+1}, {acc:.2f}%")
        
        avg_acc = sum(accuracies) / len(accuracies)
        logger.info(f"{avg_acc:.2f}%")
        return avg_acc
    
    def generate_unlearnable(self, images, targets):

        logger.info(f"{images.size(0)}")
        
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.05)
       
        for step in range(10):
            optimizer.zero_grad()
            
            perturbed = images + delta
            perturbed = torch.clamp(perturbed, 0, 1)
           
            total_loss = 0
            for model in self.models:
                model.eval()
                outputs = model(perturbed)
                loss = nn.functional.cross_entropy(outputs, targets)
                total_loss += loss
           
            em_loss = -total_loss / len(self.models) * 3.0  
            
            wrong_targets = (targets + torch.randint(1, 10, targets.shape)) % 10
            wrong_loss = 0
            for model in self.models:
                outputs = model(perturbed)
                wrong_loss += -nn.functional.cross_entropy(outputs, wrong_targets)
            wrong_loss = wrong_loss / len(self.models)
            
            total_loss = em_loss + wrong_loss * 0.5
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        final_delta = torch.clamp(delta.detach(), -self.epsilon, self.epsilon)
        unlearnable_images = images + final_delta
        unlearnable_images = torch.clamp(unlearnable_images, 0, 1)
        
        effectiveness = self._check_effectiveness(images, unlearnable_images, targets)
        
        logger.info(f"The perturbation generation is completed. Effect: {effectiveness:.3f}")
        
        return unlearnable_images, final_delta, effectiveness
    
    def _check_effectiveness(self, clean_images, unlearnable_images, targets):
        """Check the disturbance effect"""
        with torch.no_grad():
            clean_correct = 0
            unlearnable_correct = 0
            total = targets.size(0)
            
            for model in self.models:
                model.eval()
                
                clean_outputs = model(clean_images)
                clean_correct += (clean_outputs.argmax(1) == targets).sum().item()
                
                unlearnable_outputs = model(unlearnable_images)
                unlearnable_correct += (unlearnable_outputs.argmax(1) == targets).sum().item()
            
            clean_acc = clean_correct / (total * len(self.models))
            unlearnable_acc = unlearnable_correct / (total * len(self.models))
            
            return clean_acc - unlearnable_acc

def load_cifar10_fast():
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
       
        subset_indices = torch.randperm(len(train_dataset))[:320]  
        subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
        
        train_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
        
        logger.info(f"{len(subset_dataset)} ")
        return train_loader
        
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        images = torch.randn(320, 3, 32, 32)
        targets = torch.randint(0, 10, (320,))
        dataset = TensorDataset(images, targets)
        return DataLoader(dataset, batch_size=32, shuffle=True)

def main():
    
    start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device}")
    
    try:
        train_loader = load_cifar10_fast()
        dem = FastDEM(epsilon=0.1, device=device)
       
        surrogate_acc = dem.fast_train(train_loader)
        
        test_images = None
        test_targets = None
        
        for images, targets in train_loader:
            test_images = images[:16].to(device)  
            test_targets = targets[:16].to(device)
            break
        
        unlearnable_images, perturbations, effectiveness = dem.generate_unlearnable(
            test_images, test_targets
        )
        
        total_time = time.time() - start_time
        
        print(f"time: {total_time:.2f}s")
        print(f"{surrogate_acc:.2f}%")
        print(f"{effectiveness:.3f}")
        print(f"{torch.norm(perturbations).item():.4f}")
        print(f"{torch.max(torch.abs(perturbations)).item():.4f}")
       
        os.makedirs('./speed_test_output', exist_ok=True)
        
        results = {
            'total_time': total_time,
            'surrogate_accuracy': surrogate_acc,
            'perturbation_effectiveness': effectiveness,
            'perturbation_l2_norm': torch.norm(perturbations).item(),
            'perturbation_max': torch.max(torch.abs(perturbations)).item(),
            'device': str(device),
            'samples_processed': test_images.size(0)
        }
        
        with open('./speed_test_output/results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        torch.save({
            'clean_images': test_images.cpu(),
            'unlearnable_images': unlearnable_images.cpu(),
            'perturbations': perturbations.cpu()
        }, './speed_test_output/samples.pth')
        
        print(f"\nThe result has been saved to: ./speed_test_output/")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
