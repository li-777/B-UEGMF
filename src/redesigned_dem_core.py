import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrulyUnlearnableDEM:
    
    def __init__(self, epsilon=32/255, device='cpu'):
        self.epsilon = epsilon
        self.device = device
        print(f"epsilon={epsilon:.4f}")
    
    def generate_unlearnable_samples(self, images, targets, model, steps=100):
        
        print(f"（steps={steps}）...")
        
        device = next(model.parameters()).device
        images = images.to(device)
        targets = targets.to(device)
        
        unlearnable_v1 = self._wrong_feature_attack(images, targets, model, steps//3)
        
        unlearnable_v2 = self._label_leakage_attack(images, targets, model, steps//3)
        
        unlearnable_v3 = self._gradient_poisoning_attack(images, targets, model, steps//3)
        
        final_unlearnable = (unlearnable_v1 + unlearnable_v2 + unlearnable_v3) / 3.0
        
        perturbation = final_unlearnable - images
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        final_unlearnable = images + perturbation
        final_unlearnable = torch.clamp(final_unlearnable, 0, 1)
        
        return final_unlearnable.cpu(), perturbation.cpu()
    
    def _wrong_feature_attack(self, images, targets, model, steps):
        """False feature learning attack: Making the model learn incorrect features"""
        images_var = images.clone().detach()
        
        for step in range(steps):
            images_var.requires_grad_(True)
            
            features = self._extract_features(model, images_var)
            
            # Create incorrect target features (using features from other samples)
            batch_size = images.size(0)
            wrong_indices = torch.randperm(batch_size, device=images.device)
            
            for i in range(batch_size):
                if wrong_indices[i] == i:
                    wrong_indices[i] = (i + 1) % batch_size
            
            wrong_features = features[wrong_indices].detach()
            
            feature_loss = F.mse_loss(features, wrong_features)
            
            grad = torch.autograd.grad(feature_loss, images_var)[0]
            
            with torch.no_grad():
                # Normalized gradient
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
                grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                
                # Move in the direction of the wrong feature
                step_size = self.epsilon / steps
                images_var = images_var + step_size * grad_normalized
                
                # Projection constraint
                perturbation = images_var - images
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                images_var = images + perturbation
                images_var = torch.clamp(images_var, 0, 1)
        
        return images_var.detach()
    
    def _label_leakage_attack(self, images, targets, model, steps):
        """Label leakage attack: Making the model learn incorrect label associations"""
        images_var = images.clone().detach()
        
        # Create an incorrect label mapping
        num_classes = 10
        wrong_targets = (targets + torch.randint(1, num_classes, targets.shape, device=targets.device)) % num_classes
        
        for step in range(steps):
            images_var.requires_grad_(True)
            
            model.eval()
            outputs = model(images_var)
            
            correct_loss = F.cross_entropy(outputs, targets)           
            wrong_loss = F.cross_entropy(outputs, wrong_targets)    
            
            total_loss = correct_loss - 0.5 * wrong_loss
            
            grad = torch.autograd.grad(total_loss, images_var)[0]
            
            with torch.no_grad():
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
                grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                
                step_size = self.epsilon / steps
                images_var = images_var + step_size * grad_normalized
                
                perturbation = images_var - images
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                images_var = images + perturbation
                images_var = torch.clamp(images_var, 0, 1)
        
        return images_var.detach()
    
    def _gradient_poisoning_attack(self, images, targets, model, steps):
        """Gradient contamination attack: Contaminate the training gradient"""
        images_var = images.clone().detach()
        
        for step in range(steps):
            images_var.requires_grad_(True)
            
            model.eval()
            outputs = model(images_var)
            
            # Calculate the gradient of the original loss
            original_loss = F.cross_entropy(outputs, targets)
            model_grads = torch.autograd.grad(original_loss, model.parameters(), 
                                            create_graph=True, retain_graph=True)
            
            grad_norm_loss = sum(torch.norm(g)**2 for g in model_grads if g is not None)
        
            probs = F.softmax(outputs, dim=1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            total_loss = grad_norm_loss - entropy_loss
            
            grad = torch.autograd.grad(total_loss, images_var)[0]
            
            with torch.no_grad():
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
                grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                
                step_size = self.epsilon / steps
                images_var = images_var + step_size * grad_normalized
                
                perturbation = images_var - images
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                images_var = images + perturbation
                images_var = torch.clamp(images_var, 0, 1)
        
        return images_var.detach()
    
    def _extract_features(self, model, x):
        """Extract the intermediate features of the model"""
        if hasattr(model, 'features'):
            return model.features(x)
        else:
            # For models such as ResNet, obtain the features of the final convolutional layer
            features = x
            for name, module in model.named_children():
                if 'fc' not in name and 'classifier' not in name:
                    features = module(features)
                else:
                    break
            return features.view(features.size(0), -1)  # Flattening feature

def test_truly_unlearnable_dem():
   
    import torchvision.models as models
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, TensorDataset
    
    # config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = 300
    batch_size = 32
    
    # data loading
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
    
    # create models
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Pre-trained model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(images[:250], targets[:250])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_output = model(images[:250])
            test_acc = (test_output.argmax(1) == targets[:250]).float().mean()
            print(f"Epoch {epoch+1}: accurary {test_acc:.3f}")
        model.train()
    
    dem_generator = TrulyUnlearnableDEM(epsilon=64/255, device=device)
    
    unlearnable_images, perturbations = dem_generator.generate_unlearnable_samples(
        images, targets, model, steps=50
    )
    
    model.eval()
    with torch.no_grad():
        clean_outputs = model(images)
        unlearnable_outputs = model(unlearnable_images.to(device))
        
        clean_acc = (clean_outputs.argmax(1) == targets).float().mean()
        unlearnable_acc = (unlearnable_outputs.argmax(1) == targets).float().mean()
        
        print(f"   accuracy clean: {clean_acc:.3f}")
        print(f"   accuracy UE: {unlearnable_acc:.3f}")
        print(f"   reduce: {clean_acc - unlearnable_acc:.3f}")
    
    
    def train_test_model(data, labels, name, epochs=8):
        test_model = models.resnet18(pretrained=False)
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
    
    # Training comparison
    clean_acc = train_test_model(images[:200], targets[:200], "clean")
    unlearnable_acc = train_test_model(unlearnable_images[:200], targets[:200], "UE")
    
    # Calculate the privacy protection rate
    privacy_rate = (clean_acc - unlearnable_acc) / clean_acc if clean_acc > 0 else 0
    
    print(f"   The best training accuracy with clean data: {clean_acc:.1f}%")
    print(f"   The best training accuracy for UE: {unlearnable_acc:.1f}%")
    print(f"   The accuracy rate has declined.: {clean_acc - unlearnable_acc:.1f}%")
    print(f"   Privacy protection rate: {privacy_rate:.3f}")
    

if __name__ == "__main__":
    success = test_truly_unlearnable_dem()