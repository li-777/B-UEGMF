import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import logging
import json
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleConfig:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    EPSILON = 16/255
    OUTPUT_DIR = './simple_output'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleDEM:
    def __init__(self, epsilon=SimpleConfig.EPSILON, device=SimpleConfig.DEVICE):
        self.epsilon = epsilon
        self.device = device
        self.surrogate_models = []

        for i in range(3):
            model = SimpleResNet(num_classes=10).to(device)
            self.surrogate_models.append(model)
        
        logger.info(f"{device}")
    
    def train_surrogates(self, train_loader, epochs=3):
   
        
        for i, model in enumerate(self.surrogate_models):
            logger.info(f" {i+1}/3")
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    if batch_idx >= 20:  
                        break
                        
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                acc = 100. * correct / total
                logger.info(f" {i+1} Epoch {epoch+1}: Loss={total_loss/20:.4f}, Acc={acc:.2f}%")
        
    
    def generate_em_perturbation(self, images, targets):
        batch_size = images.size(0)
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        
        optimizer = optim.Adam([delta], lr=0.01)
        
        for step in range(20):  
            optimizer.zero_grad()
            
            perturbed_images = images + delta
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            
            total_loss = 0
            for model in self.surrogate_models:
                model.eval()
                outputs = model(perturbed_images)
                loss = nn.functional.cross_entropy(outputs, targets)
                total_loss += loss
            
            em_loss = -total_loss / len(self.surrogate_models)
            em_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        return delta.detach()
    
    def generate_text_perturbation(self, images):
        
        batch_size = images.size(0)
        
        img_mean = images.mean(dim=[2, 3], keepdim=True)
        img_std = images.std(dim=[2, 3], keepdim=True)
        
        noise = torch.randn_like(images) * 0.1
        text_perturbation = noise * img_std + 0.01 * (img_mean - 0.5)
        
        return text_perturbation
    
    def generate_unlearnable_examples(self, images, targets):
    
        em_perturbation = self.generate_em_perturbation(images, targets)
        
        text_perturbation = self.generate_text_perturbation(images)
        
        final_perturbation = 0.7 * em_perturbation + 0.3 * text_perturbation
        final_perturbation = torch.clamp(final_perturbation, -self.epsilon, self.epsilon)
        
        unlearnable_images = images + final_perturbation
        unlearnable_images = torch.clamp(unlearnable_images, 0, 1)
        
        return unlearnable_images, final_perturbation


def load_cifar10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=SimpleConfig.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=SimpleConfig.BATCH_SIZE, shuffle=False)
    
    logger.info(f" {len(train_dataset)}, {len(test_dataset)}")
    return train_loader, test_loader


def train_clean_model(model, train_loader, test_loader):
    
    optimizer = optim.Adam(model.parameters(), lr=SimpleConfig.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    best_acc = 0
    
    for epoch in range(SimpleConfig.EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            if batch_idx >= 50: 
                break
                
            data, targets = data.to(SimpleConfig.DEVICE), targets.to(SimpleConfig.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                if batch_idx >= 20: 
                    break
                data, targets = data.to(SimpleConfig.DEVICE), targets.to(SimpleConfig.DEVICE)
                outputs = model(data)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        model.train()
    
    logger.info(f"{best_acc:.2f}%")
    return best_acc


def evaluate_unlearnability(clean_model, dem, test_loader):
    
    clean_model.eval()
    
    clean_correct = 0
    unlearnable_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if batch_idx >= 10:  
                break
                
            images, targets = images.to(SimpleConfig.DEVICE), targets.to(SimpleConfig.DEVICE)
            
            clean_outputs = clean_model(images)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct += clean_predicted.eq(targets).sum().item()
            
            unlearnable_images, _ = dem.generate_unlearnable_examples(images, targets)
            
            unlearnable_outputs = clean_model(unlearnable_images)
            _, unlearnable_predicted = unlearnable_outputs.max(1)
            unlearnable_correct += unlearnable_predicted.eq(targets).sum().item()
            
            total += targets.size(0)
    
    clean_acc = 100. * clean_correct / total
    unlearnable_acc = 100. * unlearnable_correct / total
    protection_rate = (clean_acc - unlearnable_acc) / clean_acc if clean_acc > 0 else 0
    
    results = {
        'clean_accuracy': clean_acc,
        'unlearnable_accuracy': unlearnable_acc,
        'accuracy_drop': clean_acc - unlearnable_acc,
        'privacy_protection_rate': protection_rate
    }
    
    logger.info(f" {clean_acc:.2f}%")
    logger.info(f" {unlearnable_acc:.2f}%")
    logger.info(f" {clean_acc - unlearnable_acc:.2f}%")
    logger.info(f" {protection_rate:.3f}")
    
    return results


def main():

    logger.info(f"{SimpleConfig.DEVICE}")
    start_time = time.time()
    
    try:
        train_loader, test_loader = load_cifar10_data()
      
        clean_model = SimpleResNet(num_classes=10).to(SimpleConfig.DEVICE)
        clean_acc = train_clean_model(clean_model, train_loader, test_loader)
       
        dem = SimpleDEM()
        dem.train_surrogates(train_loader)
       
        results = evaluate_unlearnability(clean_model, dem, test_loader)
       
        experiment_results = {
            'experiment_info': {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': time.time() - start_time,
                'device': str(SimpleConfig.DEVICE),
                'epsilon': SimpleConfig.EPSILON
            },
            'clean_model_accuracy': clean_acc,
            'unlearnability_results': results
        }
        
        results_file = os.path.join(SimpleConfig.OUTPUT_DIR, 'simple_dem_results.json')
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        logger.info(f"{time.time() - start_time:.2f}s")
        logger.info(f"{results_file}")
        
        print(f"{clean_acc:.2f}%")
        print(f"{results['privacy_protection_rate']:.3f}")
        print(f"{results['accuracy_drop']:.2f}%")
        print(f"{time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())