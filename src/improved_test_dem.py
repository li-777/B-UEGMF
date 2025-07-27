import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os

def create_simple_model(num_classes=10):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_real_cifar10_subset(num_samples=500):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    try:
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        indices = torch.randperm(len(dataset))[:num_samples]
        
        images = []
        targets = []
        
        for i in indices:
            img, target = dataset[i]
            images.append(img)
            targets.append(target)
        
        images = torch.stack(images)
        targets = torch.tensor(targets)
        
        return images, targets
        
    except Exception as e:
        return generate_structured_data(num_samples)

def generate_structured_data(num_samples=500):
    
    images = []
    targets = []
    
    for i in range(num_samples):
        target = i % 10
        img = torch.randn(3, 32, 32) * 0.1
        
        if target == 0: 
            img[:, :16, :16] += 0.5
        elif target == 1:  
            img[:, :16, 16:] += 0.5
        elif target == 2:  
            img[:, 16:, :16] += 0.5
        elif target == 3:  
            img[:, 16:, 16:] += 0.5
        elif target == 4:  
            img[:, 12:20, 12:20] += 0.8
        elif target == 5:  
            img[:, ::2, :] += 0.3
        elif target == 6:  
            img[:, :, ::2] += 0.3
        elif target == 7: 
            for j in range(32):
                img[:, j, j] += 0.5
        elif target == 8:  
            img[:, [0, -1], :] += 0.4
            img[:, :, [0, -1]] += 0.4
        else:  
            mask = torch.rand(32, 32) > 0.7
            img[:, mask] += 0.6
        
        img = torch.clamp(img, 0, 1)
        
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images)
    targets = torch.tensor(targets)
    
    print(f" {len(images)}")
    return images, targets

def simple_train(model, dataloader, epochs=5, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    best_acc = 0
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}, Accuracy: {epoch_acc:.2f}%')
        best_acc = max(best_acc, epoch_acc)
    
    return best_acc

def generate_stronger_unlearnable(images, targets, model, epsilon=32/255, steps=50):
    print(f"（epsilon={epsilon:.4f}, steps={steps}）...")
    
    device = next(model.parameters()).device
    images = images.to(device)
    targets = targets.to(device)
    
    images_var = images.clone().detach()
    
    for step in range(steps):
        images_var.requires_grad_(True)
        
        model.eval()
        outputs = model(images_var)
        
        ce_loss = F.cross_entropy(outputs, targets)
        
        probs = F.softmax(outputs, dim=1)
        correct_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
        confidence_loss = correct_probs.mean()
        
        wrong_targets = (targets + torch.randint(1, 10, targets.shape, device=device)) % 10
        wrong_loss = -F.cross_entropy(outputs, wrong_targets)
        
        total_loss = 2.0 * ce_loss - 1.0 * confidence_loss + 0.5 * wrong_loss
        
        grad = torch.autograd.grad(total_loss, images_var, create_graph=False)[0]
        
        with torch.no_grad():
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            
            step_size = epsilon / steps * 2.0 
            images_var = images_var + step_size * grad_normalized
            
            perturbation = images_var - images
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            images_var = images + perturbation
            images_var = torch.clamp(images_var, 0, 1)
        
        if step % 10 == 0:
            with torch.no_grad():
                test_outputs = model(images_var)
                test_acc = (test_outputs.argmax(1) == targets).float().mean()
                print(f"   Step {step}: accuracy {test_acc:.3f}")
     
                if test_acc < 0.1:
                    print(f"  accuracy {test_acc:.3f}")
                    break
    
    unlearnable_images = images_var.detach()
    perturbation = unlearnable_images - images

    with torch.no_grad():
        model.eval()
        clean_outputs = model(images)
        unlearnable_outputs = model(unlearnable_images)
        
        clean_acc = (clean_outputs.argmax(1) == targets).float().mean()
        unlearnable_acc = (unlearnable_outputs.argmax(1) == targets).float().mean()
        
        print(f"{clean_acc:.3f}")
        print(f"{unlearnable_acc:.3f}")
        print(f"{clean_acc - unlearnable_acc:.3f}")
    
    return unlearnable_images.cpu(), perturbation.cpu()

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device}")
    
    num_samples = 500
    batch_size = 32
    
    clean_images, targets = load_real_cifar10_subset(num_samples)
   
    model = create_simple_model(num_classes=10)
    
    pretrain_dataset = TensorDataset(clean_images[:400], targets[:400])
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
    
    pretrain_acc = simple_train(model, pretrain_loader, epochs=10, lr=0.001)
    print(f"accuracy: {pretrain_acc:.1f}%")
    
    unlearnable_images, perturbations = generate_stronger_unlearnable(
        clean_images, targets, model, epsilon=32/255, steps=50
    )
    
    print(f"   L2: {torch.norm(perturbations).item():.4f}")
    print(f"   Linf: {torch.max(torch.abs(perturbations)).item():.4f}")
    print(f"   AMDF: {torch.mean(torch.abs(perturbations)).item():.6f}")
    print(f"   standard deviation: {torch.std(torch.abs(perturbations)).item():.6f}")
    
    train_samples = 300
    
    clean_train = clean_images[:train_samples]
    unlearnable_train = unlearnable_images[:train_samples]
    train_targets = targets[:train_samples]
    
    clean_model = create_simple_model(num_classes=10)
    clean_dataset = TensorDataset(clean_train, train_targets)
    clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
    
    clean_acc = simple_train(clean_model, clean_loader, epochs=8, lr=0.001)
    
    unlearnable_model = create_simple_model(num_classes=10)
    unlearnable_dataset = TensorDataset(unlearnable_train, train_targets)
    unlearnable_loader = DataLoader(unlearnable_dataset, batch_size=batch_size, shuffle=True)
    
    unlearnable_acc = simple_train(unlearnable_model, unlearnable_loader, epochs=8, lr=0.001)
    
    if clean_acc > 0:
        privacy_protection_rate = (clean_acc - unlearnable_acc) / clean_acc
    else:
        privacy_protection_rate = 0.0
    
    print(f"{clean_acc:.1f}%")
    print(f"{unlearnable_acc:.1f}%")
    print(f"{clean_acc - unlearnable_acc:.1f}%")
    print(f"{privacy_protection_rate:.3f}")
    

if __name__ == "__main__":
    success = main()