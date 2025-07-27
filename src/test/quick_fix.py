"""
Quick Fix Script - Solve DEM algorithm operation issues
"""

import os
import sys
import shutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_original_files():

    backup_dir = f"./backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = ['dem_algorithm.py', 'config.py', 'trainer.py']
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(backup_dir, file))
            logger.info(f"{file} -> {backup_dir}")
    
    return backup_dir

def apply_quick_fixes():
    
    dem_fix = '''
    def save_client_access_history(self, filepath: str) -> None:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.client_access_history, f, indent=2)
            logger.info(f"{filepath}")
        except Exception as e:
            logger.error(e)
    
    def load_client_access_history(self, filepath: str) -> None:
        try:
            with open(filepath, 'r') as f:
                self.client_access_history = json.load(f)
            logger.info(f" {filepath}")
        except FileNotFoundError:
            logger.warning(f" {filepath}")
        except Exception as e:
            logger.error(e)
'''
   
    if os.path.exists('dem_algorithm.py'):
        with open('dem_algorithm.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'save_client_access_history' not in content:
        
            class_end = content.rfind('class DynamicErrorMinimizingNoise')
            if class_end != -1:
              
                next_class = content.find('\nclass ', class_end + 1)
                if next_class == -1:
                    next_class = content.find('\nif __name__', class_end + 1)
                if next_class == -1:
                    next_class = len(content)
                
                new_content = content[:next_class] + dem_fix + '\n' + content[next_class:]
                
                with open('dem_algorithm.py', 'w', encoding='utf-8') as f:
                    f.write(new_content)
                

def create_optimized_run_script():
    script_content = '''#!/usr/bin/env python3

import subprocess
import sys
import os

def run_optimized_experiment():
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", "cifar10",
        "--model", "resnet18", 
        "--epochs", "3",  
        "--batch-size", "32",  
        "--epsilon", "0.0627",
        "--surrogate-epochs", "5",  
        "--experiment-name", "dem_quick_test",
        "--no-blockchain"  
    ]
    
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"{e}")
        return False

if __name__ == "__main__":
    success = run_optimized_experiment()
    sys.exit(0 if success else 1)
'''
    
    with open('optimized_run.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    try:
        os.chmod('optimized_run.py', 0o755)
    except:
        pass
    

def create_minimal_test():
    test_content = '''"""
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
        
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.01)
        
        model.eval()
        for step in range(10): 
            optimizer.zero_grad()
            
            perturbed = images + delta
            perturbed = torch.clamp(perturbed, 0, 1)
            
            outputs = model(perturbed)
            loss = -torch.nn.functional.cross_entropy(outputs, targets)  
            
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
            logger.info(f"{filepath}")
        except Exception as e:
            logger.error(f" {e}")

def main():
    
    batch_size = 16
    images = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size,))
    
    model = SimpleModel()
    
    dem = MinimalDEM()
    
    start_time = time.time()
    unlearnable_images, perturbations = dem.generate_unlearnable_examples(images, targets, model)
    end_time = time.time()
    
    model.eval()
    with torch.no_grad():
        clean_outputs = model(images)
        unlearnable_outputs = model(unlearnable_images)
        
        clean_acc = (clean_outputs.argmax(1) == targets).float().mean()
        unlearnable_acc = (unlearnable_outputs.argmax(1) == targets).float().mean()

if __name__ == "__main__":
    import os
    os.makedirs('./test_output', exist_ok=True)
    
    success = main()
    print(f"\\nMinimize testing{'succeed' if success else 'finish'}")
'''
    
    with open('minimal_test.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    logger.info("The minimum test script has been created: minimal_test.py")

def main():
    
    try:
        backup_dir = backup_original_files()
        print(f"The original file has been backed up to: {backup_dir}")
        
        apply_quick_fixes()
        
        create_optimized_run_script()
        create_minimal_test()
        
    except Exception as e:
        logger.error(f"error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)