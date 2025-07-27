import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models import create_model
from trainer import ModelTrainer

def quick_test_dem_fix():
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_samples = 500  
    
    clean_images = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (num_samples,))
   
    model = create_model('resnet18', num_classes=10).to(device)
  
    def generate_simple_unlearnable(images, targets, model, epsilon=16/255):
        
        images_copy = images.clone().detach().requires_grad_(True).to(device)
        targets = targets.to(device)
        
        model.eval()
        outputs = model(images_copy)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        grad = torch.autograd.grad(loss, images_copy)[0]
        
        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
        grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
        
        perturbation = epsilon * grad_normalized
        unlearnable_images = images_copy + perturbation
        unlearnable_images = torch.clamp(unlearnable_images, 0, 1)
        
        return unlearnable_images.detach().cpu(), perturbation.detach().cpu()
    
    unlearnable_images, perturbations = generate_simple_unlearnable(
        clean_images, targets, model
    )
   
    with torch.no_grad():
        model.eval()
        clean_outputs = model(clean_images.to(device))
        unlearnable_outputs = model(unlearnable_images.to(device))
        
        clean_acc = (clean_outputs.argmax(1) == targets.to(device)).float().mean()
        unlearnable_acc = (unlearnable_outputs.argmax(1) == targets.to(device)).float().mean()
        
        print(f" {clean_acc:.3f}")
        print(f" {unlearnable_acc:.3f}")
        print(f" {clean_acc - unlearnable_acc:.3f}")
        
    print(f" L2: {torch.norm(perturbations).item():.4f}")
    print(f" Linf: {torch.max(torch.abs(perturbations)).item():.4f}")
    print(f" AMDF: {torch.mean(torch.abs(perturbations)).item():.4f}")
    
    clean_model = create_model('resnet18', num_classes=10)
    clean_trainer = ModelTrainer(clean_model, device='cpu')
    clean_trainer.setup_training(learning_rate=0.01)
    
    clean_dataset = TensorDataset(clean_images[:300], targets[:300])
    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=True)
    
    clean_results = clean_trainer.train(
        train_loader=clean_loader,
        epochs=5,
        save_best=False,
        save_last=False,
        experiment_name="test_clean"
    )
    
    unlearnable_model = create_model('resnet18', num_classes=10)
    unlearnable_trainer = ModelTrainer(unlearnable_model, device='cpu')
    unlearnable_trainer.setup_training(learning_rate=0.01)
    
    unlearnable_dataset = TensorDataset(unlearnable_images[:300], targets[:300])
    unlearnable_loader = DataLoader(unlearnable_dataset, batch_size=32, shuffle=True)
    
    unlearnable_results = unlearnable_trainer.train(
        train_loader=unlearnable_loader,
        epochs=5,
        save_best=False,
        save_last=False,
        experiment_name="test_unlearnable"
    )
    
    clean_final_acc = clean_results['best_accuracy']
    unlearnable_final_acc = unlearnable_results['best_accuracy']
    privacy_protection_rate = (clean_final_acc - unlearnable_final_acc) / clean_final_acc
    
    print(f" {clean_final_acc:.1f}%")
    print(f" {unlearnable_final_acc:.1f}%")
    print(f" {privacy_protection_rate:.3f}")
    
if __name__ == "__main__":
    success = quick_test_dem_fix()