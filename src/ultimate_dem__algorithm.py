import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

class UltimateDEM:
    
    def __init__(self, epsilon=64/255, device='cpu'):
        self.epsilon = epsilon
        self.device = device
        print(f"epsilon={epsilon:.4f}")
    
    def generate_unlearnable_samples(self, images, targets, model, steps=200):
      
        print(f"（epsilon={self.epsilon:.4f}, steps={steps}）...")
        
        device = next(model.parameters()).device
        images = images.to(device)
        targets = targets.to(device)
        
        # Radical Gradient Pollution Attack (40% steps)
        stage1_images = self._aggressive_gradient_poisoning(
            images, targets, model, int(steps * 0.4)
        )
        
        # Feature space destruction attack (30% steps)
        stage2_images = self._feature_space_destruction(
            stage1_images, targets, model, int(steps * 0.3)
        )
        
        # Decision boundary pollution attack (30% steps)
        final_images = self._decision_boundary_pollution(
            stage2_images, targets, model, int(steps * 0.3)
        )
        
        # Final projection constraint
        perturbation = final_images - images
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        final_unlearnable = images + perturbation
        final_unlearnable = torch.clamp(final_unlearnable, 0, 1)
        
        return final_unlearnable.cpu(), perturbation.cpu()
    
    def _aggressive_gradient_poisoning(self, images, targets, model, steps):
        """Radical Gradient Pollution Attack"""
        images_var = images.clone().detach()
        
        for step in range(steps):
            images_var.requires_grad_(True)
            
            # Calculate the gradients of multiple loss functions
            outputs = model(images_var)
            
            # Classification loss gradient
            ce_loss = F.cross_entropy(outputs, targets)
            ce_grad = torch.autograd.grad(ce_loss, images_var, create_graph=True)[0]
            
            # Error category loss gradien
            wrong_targets = (targets + torch.randint(1, 10, targets.shape, device=targets.device)) % 10
            wrong_loss = F.cross_entropy(outputs, wrong_targets)
            wrong_grad = torch.autograd.grad(wrong_loss, images_var, create_graph=True)[0]
            
            # Confidence loss gradient
            probs = F.softmax(outputs, dim=1)
            confidence = probs.max(dim=1)[0]
            conf_loss = confidence.mean()
            conf_grad = torch.autograd.grad(conf_loss, images_var, create_graph=True)[0]
            
            # Feature variance loss gradient
            features = self._extract_features(model, images_var)
            var_loss = torch.var(features, dim=0).mean()
            var_grad = torch.autograd.grad(var_loss, images_var, create_graph=True)[0]
            
            # Combined gradient: Maximize classification loss, minimize error loss, maximize confidence, minimize feature variance
            combined_grad = (
                2.0 * ce_grad +           
                -1.0 * wrong_grad +       
                1.0 * conf_grad +        
                -0.5 * var_grad    
            )
            
            with torch.no_grad():
                # More radical gradient normalization
                grad_norm = torch.norm(combined_grad.view(combined_grad.size(0), -1), dim=1, keepdim=True)
                grad_normalized = combined_grad / (grad_norm.view(-1, 1, 1, 1) + 1e-12)
                
                # Adaptive step size: Adjusted according to the loss value
                step_multiplier = 1.0 + (step / steps) * 2.0  
                adaptive_step_size = (self.epsilon / steps) * step_multiplier
                
                images_var = images_var + adaptive_step_size * grad_normalized
                
                # Projection constraint
                perturbation = images_var - images
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                images_var = images + perturbation
                images_var = torch.clamp(images_var, 0, 1)
            
            if step % 20 == 0:
                with torch.no_grad():
                    test_outputs = model(images_var)
                    test_acc = (test_outputs.argmax(1) == targets).float().mean()
                    print(f"Step 1 {step}: accuracy {test_acc:.3f}")
        
        return images_var.detach()
    
    def _feature_space_destruction(self, images, targets, model, steps):
        """Feature space destruction attack"""
        images_var = images.clone().detach()
        
        all_features = []
        model.eval()
        with torch.no_grad():
            for class_id in range(10):
                class_mask = (targets == class_id)
                if class_mask.sum() > 0:
                    class_images = images[class_mask]
                    class_features = self._extract_features(model, class_images)
                    avg_features = class_features.mean(dim=0, keepdim=True)
                    all_features.append(avg_features)
                else:
                    # If there is no sample for a certain category, use random features
                    dummy_features = torch.randn(1, 512, device=images.device)  
                    all_features.append(dummy_features)
        
        for step in range(steps):
            images_var.requires_grad_(True)
            
            current_features = self._extract_features(model, images_var)
            
            # Bring the features closer to those of the wrong category
            batch_size = images.size(0)
            wrong_class_features = []
            
            for i in range(batch_size):
                current_class = targets[i].item()
                wrong_class = (current_class + 1 + np.random.randint(0, 8)) % 10  
                wrong_class_features.append(all_features[wrong_class])
            
            wrong_features_batch = torch.cat(wrong_class_features, dim=0)
            
            feature_confusion_loss = F.mse_loss(current_features, wrong_features_batch)
            
            # Minimize the variance of features within a class
            intra_class_var_loss = 0
            for class_id in range(10):
                class_mask = (targets == class_id)
                if class_mask.sum() > 1:
                    class_features = current_features[class_mask]
                    class_var = torch.var(class_features, dim=0).mean()
                    intra_class_var_loss += class_var
            
            # Maximize the similarity of features between classes
            inter_class_sim_loss = 0
            unique_classes = torch.unique(targets)
            for i, class_a in enumerate(unique_classes):
                for class_b in unique_classes[i+1:]:
                    features_a = current_features[targets == class_a].mean(dim=0)
                    features_b = current_features[targets == class_b].mean(dim=0)
                    similarity = F.cosine_similarity(features_a.unsqueeze(0), features_b.unsqueeze(0))
                    inter_class_sim_loss += (1.0 - similarity)  
            
            total_loss = (
                1.0 * feature_confusion_loss +
                -0.5 * intra_class_var_loss +    
                -0.5 * inter_class_sim_loss      
            )
            
            grad = torch.autograd.grad(total_loss, images_var)[0]
            
            with torch.no_grad():
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
                grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-12)
                
                step_size = self.epsilon / steps * 1.5  
                images_var = images_var + step_size * grad_normalized
                
                perturbation = images_var - images
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                images_var = images + perturbation
                images_var = torch.clamp(images_var, 0, 1)
            
            if step % 20 == 0:
                with torch.no_grad():
                    test_outputs = model(images_var)
                    test_acc = (test_outputs.argmax(1) == targets).float().mean()
                    print(f"Step {step}: accuracy {test_acc:.3f}")
        
        return images_var.detach()
    
    def _decision_boundary_pollution(self, images, targets, model, steps):
        """Decision boundary pollution attack"""
        images_var = images.clone().detach()
        
        for step in range(steps):
            images_var.requires_grad_(True)
            
            outputs = model(images_var)
            
            probs = F.softmax(outputs, dim=1)
            correct_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            outputs_masked = outputs.clone()
            outputs_masked.scatter_(1, targets.unsqueeze(1), -float('inf'))
            wrong_class_logits = outputs_masked.max(dim=1)[0]
            
            margin_loss = F.relu(outputs.gather(1, targets.unsqueeze(1)).squeeze() - wrong_class_logits + 1.0)
            
            entropy_loss = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
            uncertainty_loss = -entropy_loss  
            
            noise = torch.randn_like(images_var) * 0.01
            noisy_outputs = model(images_var + noise)
            sensitivity_loss = F.mse_loss(outputs, noisy_outputs)
            
            total_loss = (
                1.0 * margin_loss.mean() +
                0.5 * uncertainty_loss +
                0.3 * sensitivity_loss
            )
            
            grad = torch.autograd.grad(total_loss, images_var)[0]
            
            with torch.no_grad():
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
                grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-12)
                
                step_multiplier = 1.0 + (step / steps) * 3.0
                step_size = (self.epsilon / steps) * step_multiplier
                
                images_var = images_var + step_size * grad_normalized
                
                perturbation = images_var - images
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                images_var = images + perturbation
                images_var = torch.clamp(images_var, 0, 1)
            
            if step % 20 == 0:
                with torch.no_grad():
                    test_outputs = model(images_var)
                    test_acc = (test_outputs.argmax(1) == targets).float().mean()
                    print(f" Step 3 {step}: accuracy {test_acc:.3f}")
        
        return images_var.detach()
    
    def _extract_features(self, model, x):
        """Extract model features"""
        if hasattr(model, 'features'):
            features = model.features(x)
            return features.view(features.size(0), -1)
        else:
            features = x
            for name, module in model.named_children():
                if 'fc' not in name and 'classifier' not in name:
                    features = module(features)
                else:
                    break
            return features.view(features.size(0), -1)

def test_ultimate_dem():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = 300
    batch_size = 32
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    
    indices = torch.randperm(len(dataset))[:num_samples]
    images = []
    targets = []
    
    for i in indices:
        img, target = dataset[i]
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images)
    targets = torch.tensor(targets)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(images[:250], targets[:250])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(8):  
        epoch_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            epoch_acc += pred.eq(target).sum().item()
        
        epoch_loss /= len(train_loader)
        epoch_acc = 100. * epoch_acc / len(train_dataset)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.1f}%")
    
    dem_generator = UltimateDEM(epsilon=64/255, device=device)  
    
    unlearnable_images, perturbations = dem_generator.generate_unlearnable_samples(
        images, targets, model, steps=150  
    )
    
    model.eval()
    with torch.no_grad():
        clean_outputs = model(images)
        unlearnable_outputs = model(unlearnable_images.to(device))
        
        clean_acc = (clean_outputs.argmax(1) == targets).float().mean()
        unlearnable_acc = (unlearnable_outputs.argmax(1) == targets).float().mean()
        
        print(f"   Accuracy of clean samples: {clean_acc:.3f}")
        print(f"   Accuracy OF UE : {unlearnable_acc:.3f}")
        print(f"   The accuracy rate has declined.: {clean_acc - unlearnable_acc:.3f}")
    
   
    print(f"   L2: {torch.norm(perturbations).item():.4f}")
    print(f"   Linf: {torch.max(torch.abs(perturbations)).item():.4f}")
    print(f"   AMDF: {torch.mean(torch.abs(perturbations)).item():.6f}")
    
    
    def train_test_model(data, labels, name, epochs=10):
        test_model = models.resnet18(weights=None)
        test_model.fc = nn.Linear(test_model.fc.in_features, 10)
        
        optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_acc = 0
        for epoch in range(epochs):
            test_model.train()
            for batch_idx, (batch_data, batch_target) in enumerate(loader):
                optimizer.zero_grad()
                output = test_model(batch_data)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
            
            test_model.eval()
            with torch.no_grad():
                test_output = test_model(data)
                test_acc = (test_output.argmax(1) == labels).float().mean() * 100
                best_acc = max(best_acc, test_acc)
                print(f"{name} Epoch {epoch+1}: accuracy {test_acc:.1f}%")
        
        return best_acc
    
    # 训练对比
    clean_acc = train_test_model(images[:200], targets[:200], "CLEAN", epochs=12)
    unlearnable_acc = train_test_model(unlearnable_images[:200], targets[:200], "UE", epochs=12)
    
    # 计算隐私保护率
    privacy_rate = (clean_acc - unlearnable_acc) / clean_acc if clean_acc > 0 else 0
    
    print(f"   The best training accuracy with clean data: {clean_acc:.1f}%")
    print(f"   The best training accuracy with UE: {unlearnable_acc:.1f}%")
    print(f"   The accuracy rate has declined: {clean_acc - unlearnable_acc:.1f}%")
    print(f"   Privacy protection rate: {privacy_rate:.3f}")

if __name__ == "__main__":
    success = test_ultimate_dem()