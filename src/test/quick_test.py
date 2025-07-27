"""
Quick Test Script - Skip BERT model download and focus on testing core functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# Set up simple logs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDEMGenerator:
    """Simplified DEM generator, without using BERT"""
    
    def __init__(self, epsilon=8/255, alpha=0.5, lambda_reg=0.01, device='cpu'):
        self.epsilon = epsilon
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.device = device
        
    def generate_image_perturbation(self, images: torch.Tensor, 
                                  targets: torch.Tensor, 
                                  model: nn.Module,
                                  num_steps: int = 10) -> torch.Tensor:
        """Generate image perturbation"""
        batch_size = images.size(0)
        
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.SGD([delta], lr=0.01)
        
        model.eval()
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            perturbed_images = images + delta
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

            outputs = model(perturbed_images)
      
            loss = F.cross_entropy(outputs, targets)
       
            loss.backward()
            optimizer.step()
       
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        return delta.detach()
    
    def generate_simple_text_perturbation(self, batch_size: int, target_shape: tuple) -> torch.Tensor:
        
        channels, height, width = target_shape[1:]
        text_perturbation = torch.randn(batch_size, channels, height, width, device=self.device)
        text_perturbation = torch.tanh(text_perturbation) * self.epsilon * 0.1  
        return text_perturbation
    
    def simple_fusion(self, image_perturbation: torch.Tensor, 
                     text_perturbation: torch.Tensor) -> torch.Tensor:
        
        return self.alpha * image_perturbation + (1 - self.alpha) * text_perturbation
    
    def compute_simple_loss(self, perturbed_images: torch.Tensor,
                           targets: torch.Tensor,
                           perturbation: torch.Tensor,
                           model: nn.Module) -> tuple:
     
        outputs = model(perturbed_images)
        ce_loss = F.cross_entropy(outputs, targets)
        
        reg_loss = self.lambda_reg * torch.norm(perturbation, p=2) ** 2
        
        batch_size, channels, height, width = perturbation.shape
        smoothness_loss = torch.tensor(0.0, device=perturbation.device)
        
        if height > 1 and width > 1:
            try:
                vertical_diff = torch.abs(perturbation[:, :, 1:, :] - perturbation[:, :, :-1, :])
                horizontal_diff = torch.abs(perturbation[:, :, :, 1:] - perturbation[:, :, :, :-1])
                smoothness_loss = 0.001 * (torch.mean(vertical_diff) + torch.mean(horizontal_diff))
            except:
                smoothness_loss = torch.tensor(0.0, device=perturbation.device)
        
        total_loss = ce_loss + reg_loss + smoothness_loss
        
        loss_breakdown = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'reg_loss': reg_loss.item(),
            'smoothness_loss': smoothness_loss.item()
        }
        
        return total_loss, loss_breakdown
    
    def generate_unlearnable_examples(self, images: torch.Tensor,
                                    targets: torch.Tensor,
                                    model: nn.Module) -> tuple:

        batch_size = images.size(0)
    
        image_perturbation = self.generate_image_perturbation(images, targets, model)
        
        text_perturbation = self.generate_simple_text_perturbation(batch_size, images.shape)
        
        final_perturbation = self.simple_fusion(image_perturbation, text_perturbation)
        
        unlearnable_images = images + final_perturbation
        unlearnable_images = torch.clamp(unlearnable_images, 0, 1)
        
        total_loss, loss_breakdown = self.compute_simple_loss(
            unlearnable_images, targets, final_perturbation, model
        )
        
        metadata = {
            'client_ids': [f"client_{i}" for i in range(batch_size)],
            'loss_breakdown': loss_breakdown,
            'epsilon': self.epsilon,
            'alpha': self.alpha
        }
        
        return unlearnable_images, final_perturbation, metadata
    
    def evaluate_unlearnability(self, clean_images: torch.Tensor,
                               unlearnable_images: torch.Tensor,
                               targets: torch.Tensor,
                               model: nn.Module) -> dict:
      
        model.eval()
        
        with torch.no_grad():
            clean_outputs = model(clean_images)
            clean_acc = (clean_outputs.argmax(dim=1) == targets).float().mean().item()
            clean_loss = F.cross_entropy(clean_outputs, targets).item()
            
            unlearnable_outputs = model(unlearnable_images)
            unlearnable_acc = (unlearnable_outputs.argmax(dim=1) == targets).float().mean().item()
            unlearnable_loss = F.cross_entropy(unlearnable_outputs, targets).item()
            
            acc_drop = clean_acc - unlearnable_acc
            loss_increase = unlearnable_loss - clean_loss
            
            privacy_protection_rate = acc_drop / clean_acc if clean_acc > 0 else 0
        
        return {
            'clean_accuracy': clean_acc,
            'unlearnable_accuracy': unlearnable_acc,
            'accuracy_drop': acc_drop,
            'clean_loss': clean_loss,
            'unlearnable_loss': unlearnable_loss,
            'loss_increase': loss_increase,
            'privacy_protection_rate': privacy_protection_rate
        }


def quick_test():
    try:
 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"{device}")
        
        from data_loader import create_data_loader
        data_loader = create_data_loader('cifar10')
        train_loader, test_loader = data_loader.get_data_loaders(batch_size=8)
        
        from models import create_model
        model = create_model('resnet18', num_classes=10)
        model.to(device)
        
        dem_generator = SimpleDEMGenerator(device=device)
        
        for images, targets in train_loader:
            images = images[:4].to(device)  
            targets = targets[:4].to(device)
            break
        
        logger.info(f"{images.shape}")
        
        unlearnable_images, perturbations, metadata = dem_generator.generate_unlearnable_examples(
            images, targets, model
        )
        
        logger.info(f" {unlearnable_images.shape}")
        logger.info(f" {perturbations.shape}")
        logger.info(f" [{perturbations.min():.4f}, {perturbations.max():.4f}]")
        
        metrics = dem_generator.evaluate_unlearnability(
            images, unlearnable_images, targets, model
        )
        
        logger.info(f" {metrics['clean_accuracy']:.3f}")
        logger.info(f" {metrics['unlearnable_accuracy']:.3f}")
        logger.info(f" {metrics['privacy_protection_rate']:.3f}")
        
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = quick_test()
    sys.exit(0 if success else 1)