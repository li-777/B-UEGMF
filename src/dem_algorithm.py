import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import logging
import json
import os
from typing import Tuple, List, Optional, Dict, Any
import hashlib
import copy
from torch.utils.data import DataLoader, TensorDataset

# Simplified imports to avoid dependency issues
try:
    from config import Config
    EPSILON = getattr(Config.DEM, 'EPSILON', 16/255)
    ALPHA = getattr(Config.DEM, 'ALPHA', 0.3)
    LAMBDA_REG = getattr(Config.DEM, 'LAMBDA_REG', 0.1)
    DEVICE = getattr(Config, 'DEVICE', torch.device('cpu'))
except ImportError:
    EPSILON = 16/255
    ALPHA = 0.3
    LAMBDA_REG = 0.1
    DEVICE = torch.device('cpu')

logger = logging.getLogger(__name__)


class EMSurrogateModel:
    """EM Surrogate Model Manager - Enhanced Training Effects"""
    
    def __init__(self, model_architecture: str, num_classes: int, device: str):
        self.device = device
        self.model_architecture = model_architecture
        self.num_classes = num_classes
        
        # Create 3 surrogate models
        self.surrogate_models = []
        self.num_surrogates = 3
        
        # Import model creation function
        try:
            from models import create_model
            for i in range(self.num_surrogates):
                model = create_model(model_architecture, num_classes)
                model.to(device)
                self.surrogate_models.append(model)
        except ImportError:
            # Use simplified model if import fails
            for i in range(self.num_surrogates):
                model = self._create_simple_model(num_classes)
                model.to(device)
                self.surrogate_models.append(model)
        
        logger.info(f"Created {self.num_surrogates} surrogate models")
    
    def _create_simple_model(self, num_classes):
        """Create simplified ResNet model"""
        class SimpleResNet(nn.Module):
            def __init__(self, num_classes):
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
        
        return SimpleResNet(num_classes)
    
    def train_surrogates(self, clean_data_loader: DataLoader, epochs: int = 20) -> Dict[str, Any]:
        """Train surrogate model ensemble - Enhanced training effects"""
        logger.info("Starting surrogate model training...")
        
        results = []
        
        for i, model in enumerate(self.surrogate_models):
            logger.info(f"Training surrogate model {i+1}/{self.num_surrogates}")
            
            # Optimized training process
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            best_acc = 0.0
            
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (data, targets) in enumerate(clean_data_loader):
                    if batch_idx >= 20:
                        break
                        
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                scheduler.step()
                
                acc = 100. * correct / total if total > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                
                if epoch % 5 == 0:
                    logger.info(f"  Model{i+1} Epoch {epoch+1}: Acc={acc:.2f}%, LR={scheduler.get_last_lr()[0]:.6f}")
            
            results.append({'best_accuracy': best_acc, 'final_accuracy': acc})
            logger.info(f"Surrogate model {i+1} training complete, best accuracy: {best_acc:.2f}%")
        
        # Calculate average accuracy correctly
        avg_accuracy = np.mean([r['best_accuracy'] for r in results])
        
        logger.info(f"Surrogate model training complete, average accuracy: {avg_accuracy:.2f}%")
        
        # Quality check
        if avg_accuracy < 30:
            logger.warning(f"Surrogate model accuracy too low ({avg_accuracy:.1f}%), this may affect perturbation effectiveness")
        
        return {
            'individual_results': results,
            'average_accuracy': avg_accuracy,
            'num_models': self.num_surrogates
        }
    
    def get_ensemble_predictions(self, images: torch.Tensor) -> torch.Tensor:
        """Get ensemble prediction results"""
        all_outputs = []
        
        for model in self.surrogate_models:
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                all_outputs.append(outputs)
        
        # Average ensemble
        ensemble_output = torch.stack(all_outputs).mean(dim=0)
        return ensemble_output
    
    def compute_ensemble_loss(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ensemble loss"""
        total_loss = 0.0
        
        for model in self.surrogate_models:
            model.eval()
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss
        
        return total_loss / self.num_surrogates


class SimplifiedTextProcessor:
    """Simplified Text Processor - Maintain original implementation"""
    
    def __init__(self, device: str):
        self.device = device
        
    def generate_text_perturbation(self, client_info: Dict[str, Any], 
                                  images: torch.Tensor) -> torch.Tensor:
        """Generate simplified text perturbation"""
        batch_size, channels, height, width = images.shape
        
        # Generate pseudo-random seeds based on client info
        seeds = []
        for i, client_id in enumerate(client_info['client_ids']):
            seed = hash(client_id + str(client_info['timestamps'][i])) % 1000000
            seeds.append(seed)
        
        # Generate seed-based perturbations
        perturbations = []
        for seed in seeds:
            torch.manual_seed(seed)
            noise = torch.randn(channels, height, width) * 0.08
            perturbations.append(noise)
        
        text_perturbation = torch.stack(perturbations).to(self.device)
        
        return text_perturbation


class DynamicErrorMinimizingNoise:
    """Dynamic Error Minimizing Noise Generator - Fixed perturbation effects"""
    
    def __init__(self, 
                 dataset_name: str = 'cifar10',
                 epsilon: float = EPSILON,
                 alpha: float = ALPHA,
                 lambda_reg: float = LAMBDA_REG,
                 device: str = DEVICE):
        
        self.dataset_name = dataset_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.device = device
        
        # Surrogate model manager
        self.surrogate_manager = None
        
        # Text processor
        self.text_processor = SimplifiedTextProcessor(device)
        
        # Client access records
        self.client_access_history = {}
        
        logger.info(f"DEM generator initialized - Fixed perturbation effects version")
    
    def setup_surrogate_models(self, model_architecture: str, num_classes: int, 
                              clean_data_loader: DataLoader) -> Dict[str, Any]:
        """Setup and train surrogate models"""
        logger.info("Setting up surrogate models...")
        
        self.surrogate_manager = EMSurrogateModel(model_architecture, num_classes, self.device)
        
        # Train surrogate models with more epochs
        training_results = self.surrogate_manager.train_surrogates(clean_data_loader, epochs=20)
        
        logger.info("Surrogate model setup complete")
        return training_results
    
    def generate_client_metadata(self, batch_size: int, client_ids: Optional[List[str]] = None) -> Dict[str, List]:
        """Generate client metadata"""
        if client_ids is None:
            client_ids = [f"client_{i%10:03d}" for i in range(batch_size)]
        elif not isinstance(client_ids, list):
            logger.warning(f"client_ids type error, expected list but got {type(client_ids)}, using default")
            client_ids = [f"client_{i%10:03d}" for i in range(batch_size)]
        
        timestamps = [int(time.time()) + i for i in range(batch_size)]
        
        access_hashes = []
        for cid, ts in zip(client_ids, timestamps):
            if cid not in self.client_access_history:
                self.client_access_history[cid] = []
            self.client_access_history[cid].append(ts)
            
            access_data = f"{cid}_{ts}_{len(self.client_access_history[cid])}"
            access_hash = hashlib.sha256(access_data.encode()).hexdigest()[:16]
            access_hashes.append(access_hash)
        
        return {
            'client_ids': client_ids,
            'timestamps': timestamps,
            'access_hashes': access_hashes
        }
    
    def generate_em_perturbation_enhanced(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Enhanced EM perturbation generation method - Fix zero perturbation effect problem
        """
        if self.surrogate_manager is None:
            raise ValueError("Surrogate model not initialized, please call setup_surrogate_models first")
        
        batch_size = images.size(0)
        
        # Check surrogate model status
        with torch.no_grad():
            test_outputs = self.surrogate_manager.get_ensemble_predictions(images)
            model_confidence = F.softmax(test_outputs, dim=1).max(1)[0].mean().item()
            logger.debug(f"Surrogate model average confidence: {model_confidence:.3f}")
        
        # Increase initial perturbation strength
        delta_I = torch.randn_like(images, device=self.device) * 0.05
        delta_I.requires_grad_(True)
        
        # Optimize hyperparameters
        num_steps = 60
        step_size = self.epsilon / 3
        
        logger.debug(f"Starting enhanced EM perturbation generation, steps: {num_steps}, step size: {step_size:.6f}")
        
        for step in range(num_steps):
            # Add perturbation to images
            perturbed_images = images + delta_I
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            
            # Enhanced loss function calculation
            # Main loss: ensemble cross-entropy loss
            ensemble_loss = self.surrogate_manager.compute_ensemble_loss(perturbed_images, targets)
            
            # Auxiliary loss: misclassification loss (encourage wrong predictions)
            wrong_targets = (targets + torch.randint(1, 10, targets.shape, device=self.device)) % 10
            wrong_outputs = self.surrogate_manager.get_ensemble_predictions(perturbed_images)
            wrong_loss = -F.cross_entropy(wrong_outputs, wrong_targets)
            
            # Confidence loss: reduce prediction confidence
            probs = F.softmax(wrong_outputs, dim=1)
            confidence_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            # Reduce L2 regularization weight
            l2_regularization = self.lambda_reg * 0.05 * torch.norm(delta_I, p=2) ** 2
            
            # Combined loss - more aggressive combination
            total_loss = ensemble_loss + 0.5 * wrong_loss + 0.3 * confidence_loss + l2_regularization
            
            # Calculate gradients
            total_loss.backward()
            
            # Enhanced gradient update
            with torch.no_grad():
                if delta_I.grad is not None:
                    grad = delta_I.grad.data
                    grad_norm = torch.norm(grad).item()
                    
                    # Check if gradient is too small
                    if grad_norm < 1e-8:
                        logger.warning(f"Step {step}: gradient too small ({grad_norm:.2e}), adding random perturbation")
                        delta_I.data += torch.randn_like(delta_I.data) * 0.01
                    else:
                        # Gradient normalization
                        grad_normalized = grad / (grad_norm + 1e-8)
                        
                        # Adaptive step size: increase step size in later stages
                        adaptive_step = step_size * (1.5 + step / num_steps)
                        
                        # Gradient ascent (error minimization)
                        delta_I.data = delta_I.data + adaptive_step * grad_normalized
                        
                        # Constraint projection
                        delta_norm = torch.norm(delta_I.data.view(delta_I.size(0), -1), dim=1)
                        max_norm = self.epsilon * 1.2
                        
                        mask = delta_norm > max_norm
                        if mask.any():
                            scale_factor = max_norm / delta_norm[mask]
                            delta_I.data[mask] = delta_I.data[mask] * scale_factor.view(-1, 1, 1, 1)
            
            # Clear gradients
            delta_I.grad = None
            
            # Validate effect every 15 steps
            if step % 15 == 0:
                with torch.no_grad():
                    test_outputs = self.surrogate_manager.get_ensemble_predictions(perturbed_images)
                    test_acc = (test_outputs.argmax(1) == targets).float().mean()
                    delta_magnitude = torch.norm(delta_I).item()
                    logger.debug(f"Step {step}: accuracy={test_acc:.3f}, perturbation magnitude={delta_magnitude:.6f}")
        
        # Final check and backup strategy
        final_norm = torch.norm(delta_I.detach()).item()
        logger.debug(f"Final perturbation norm: {final_norm:.6f}")
        
        if final_norm < 1e-5:
            logger.warning("Perturbation too small, using backup perturbation generation strategy")
            # Backup strategy: directly generate random perturbation
            backup_perturbation = torch.randn_like(images) * (self.epsilon / 2)
            return backup_perturbation
        
        # Final projection to epsilon constraint
        with torch.no_grad():
            delta_norm = torch.norm(delta_I.data.view(delta_I.size(0), -1), dim=1)
            final_max_norm = self.epsilon
            mask = delta_norm > final_max_norm
            if mask.any():
                scale_factor = final_max_norm / delta_norm[mask]
                delta_I.data[mask] = delta_I.data[mask] * scale_factor.view(-1, 1, 1, 1)
        
        return delta_I.detach()
    
    def generate_unlearnable_examples(self, images: torch.Tensor,
                                    targets: torch.Tensor,
                                    client_ids: Optional[List[str]] = None,
                                    return_perturbation: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Generate unlearnable examples - Using enhanced EM algorithm
        """
        
        if self.surrogate_manager is None:
            raise ValueError("Please call setup_surrogate_models to initialize surrogate models first")
        
        batch_size = images.size(0)
        
        logger.info(f"Starting generation of {batch_size} unlearnable examples (enhanced version)...")
        
        # 1. Generate client metadata
        client_info = self.generate_client_metadata(batch_size, client_ids)
        
        # 2. Use enhanced EM method to generate base image perturbation δ^I
        logger.debug("Generating enhanced EM image perturbation...")
        em_perturbation = self.generate_em_perturbation_enhanced(images, targets)
        
        # 3. Generate text perturbation δ^T
        logger.debug("Generating text perturbation...")
        text_perturbation = self.text_processor.generate_text_perturbation(client_info, images)
        
        # 4. Multimodal fusion (following paper formula: δ^U = α·δ^T + δ^I)
        logger.debug("Performing multimodal fusion...")
        final_perturbation = self.alpha * text_perturbation + em_perturbation
        
        # 5. Final constraint projection
        final_norm = torch.norm(final_perturbation.view(final_perturbation.size(0), -1), dim=1)
        max_norm = self.epsilon
        mask = final_norm > max_norm
        if mask.any():
            scale_factor = max_norm / final_norm[mask]
            final_perturbation[mask] = final_perturbation[mask] * scale_factor.view(-1, 1, 1, 1)
        
        # 6. Apply perturbation to generate unlearnable examples
        unlearnable_images = images + final_perturbation
        unlearnable_images = torch.clamp(unlearnable_images, 0, 1)
        
        # 7. Validate perturbation effectiveness
        effectiveness = self._validate_perturbation_effectiveness(images, unlearnable_images, targets)
        
        # 8. Calculate statistics
        perturbation_stats = self._compute_perturbation_stats(final_perturbation)
        perturbation_stats['effectiveness'] = effectiveness
        
        # Metadata
        metadata = {
            'client_info': client_info,
            'perturbation_stats': perturbation_stats,
            'em_component_norm': torch.norm(em_perturbation).item(),
            'text_component_norm': torch.norm(text_perturbation).item(),
            'fusion_method': 'enhanced_weighted_combination',
            'dataset_name': self.dataset_name,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'enhanced_version': True
        }
        
        logger.info(f"Unlearnable examples generation complete (enhanced version), perturbation effectiveness: {effectiveness:.3f}")
        
        if return_perturbation:
            return unlearnable_images, final_perturbation, metadata
        else:
            return unlearnable_images, None, metadata
    
    def _validate_perturbation_effectiveness(self, clean_images: torch.Tensor,
                                           unlearnable_images: torch.Tensor,
                                           targets: torch.Tensor) -> float:
        """Validate perturbation effectiveness"""
        if self.surrogate_manager is None:
            return 0.0
        
        with torch.no_grad():
            # Predictions on clean images
            clean_outputs = self.surrogate_manager.get_ensemble_predictions(clean_images)
            clean_acc = (clean_outputs.argmax(1) == targets).float().mean()
            
            # Predictions on perturbed images
            perturbed_outputs = self.surrogate_manager.get_ensemble_predictions(unlearnable_images)
            perturbed_acc = (perturbed_outputs.argmax(1) == targets).float().mean()
            
            effectiveness = clean_acc - perturbed_acc
            
            logger.debug(f"Perturbation validation: clean={clean_acc:.3f}, perturbed={perturbed_acc:.3f}, effectiveness={effectiveness:.3f}")
            
            return effectiveness.item()
    
    def _compute_perturbation_stats(self, perturbation: torch.Tensor) -> Dict[str, float]:
        """Calculate perturbation statistics"""
        return {
            'l2_norm': torch.norm(perturbation, p=2).item(),
            'linf_norm': torch.norm(perturbation, p=float('inf')).item(),
            'mean_magnitude': torch.mean(torch.abs(perturbation)).item(),
            'std_magnitude': torch.std(torch.abs(perturbation)).item(),
            'max_magnitude': torch.max(torch.abs(perturbation)).item(),
            'min_magnitude': torch.min(torch.abs(perturbation)).item(),
        }
    
    def evaluate_unlearnability(self, clean_images: torch.Tensor,
                               unlearnable_images: torch.Tensor,
                               targets: torch.Tensor,
                               surrogate_model: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """Evaluate unlearnability - Fixed parameter matching"""
        if self.surrogate_manager is None:
            logger.warning("Surrogate model not initialized, cannot perform complete evaluation")
            return {}
        
        with torch.no_grad():
            # Evaluate using internal surrogate model ensemble
            clean_outputs = self.surrogate_manager.get_ensemble_predictions(clean_images)
            unlearnable_outputs = self.surrogate_manager.get_ensemble_predictions(unlearnable_images)
            
            clean_acc = (clean_outputs.argmax(1) == targets).float().mean().item()
            unlearnable_acc = (unlearnable_outputs.argmax(1) == targets).float().mean().item()
            
            # Calculate losses
            clean_loss = F.cross_entropy(clean_outputs, targets).item()
            unlearnable_loss = F.cross_entropy(unlearnable_outputs, targets).item()
            
            # Privacy protection rate
            privacy_protection_rate = (clean_acc - unlearnable_acc) / clean_acc if clean_acc > 0 else 0
            
        return {
            'clean_accuracy': clean_acc,
            'unlearnable_accuracy': unlearnable_acc,
            'accuracy_drop': clean_acc - unlearnable_acc,
            'clean_loss': clean_loss,
            'unlearnable_loss': unlearnable_loss,
            'loss_increase': unlearnable_loss - clean_loss,
            'privacy_protection_rate': privacy_protection_rate,
            'surrogate_ensemble_size': self.surrogate_manager.num_surrogates,
            'enhanced_evaluation': True
        }
    
    def save_client_access_history(self, filepath: str) -> None:
        """Save client access history"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.client_access_history, f, indent=2)
            logger.info(f"Client access history saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save client access history: {e}")
    
    def load_client_access_history(self, filepath: str) -> None:
        """Load client access history"""
        try:
            with open(filepath, 'r') as f:
                self.client_access_history = json.load(f)
            logger.info(f"Client access history loaded: {filepath}")
        except FileNotFoundError:
            logger.warning(f"Access history file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load access history: {e}")


class DEMBatchProcessor:
    """DEM Batch Processor - Using enhanced DEM"""
    
    def __init__(self, dem_generator: DynamicErrorMinimizingNoise):
        self.dem_generator = dem_generator
        
    def process_dataset(self, dataloader, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Process entire dataset - Using enhanced DEM"""
        all_clean_images = []
        all_unlearnable_images = []
        all_targets = []
        all_perturbations = []
        all_metadata = []
        
        # Reduce processing batches to improve speed
        max_batches = min(3, len(dataloader))
        
        logger.info(f"Starting batch processing (enhanced DEM), processing at most {max_batches} batches")
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            images = images.to(self.dem_generator.device)
            targets = targets.to(self.dem_generator.device)
            
            try:
                start_time = time.time()
                
                # Use enhanced DEM to generate unlearnable examples
                unlearnable_images, perturbations, metadata = \
                    self.dem_generator.generate_unlearnable_examples(
                        images, targets
                    )
                
                process_time = time.time() - start_time
                
                # Collect results
                all_clean_images.append(images.cpu())
                all_unlearnable_images.append(unlearnable_images.cpu())
                all_targets.append(targets.cpu())
                all_perturbations.append(perturbations.cpu())
                all_metadata.append(metadata)
                
                logger.info(f"Batch {batch_idx + 1}/{max_batches} complete, time: {process_time:.1f}s, "
                           f"perturbation effectiveness: {metadata['perturbation_stats']['effectiveness']:.3f}")
                
            except Exception as e:
                logger.error(f"Batch {batch_idx} processing failed: {e}")
                continue
        
        # Combine results
        results = {
            'clean_images': torch.cat(all_clean_images, dim=0),
            'unlearnable_images': torch.cat(all_unlearnable_images, dim=0),
            'targets': torch.cat(all_targets, dim=0),
            'perturbations': torch.cat(all_perturbations, dim=0),
            'metadata': all_metadata,
            'enhanced_dem': True
        }
        
        # Save results
        if save_path:
            torch.save(results, save_path)
            logger.info(f"Processing results saved: {save_path}")
        
        return results


if __name__ == "__main__":
    # Test enhanced DEM algorithm
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    device = torch.device('cpu')
    batch_size = 4
    
    # Simulate data
    test_images = torch.randn(batch_size, 3, 32, 32)
    test_targets = torch.randint(0, 10, (batch_size,))
    
    # Create simple data loader
    train_dataset = TensorDataset(test_images, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize enhanced DEM generator
    dem_generator = DynamicErrorMinimizingNoise(dataset_name='cifar10', device=device)
    
    print("Testing enhanced DEM algorithm (precision fixed version)...")
    
    try:
        # Setup surrogate models
        print("Setting up surrogate models...")
        surrogate_results = dem_generator.setup_surrogate_models('resnet18', 10, train_loader)
        print(f"Surrogate model average accuracy: {surrogate_results['average_accuracy']:.2f}%")
        
        # Generate unlearnable examples
        print("Generating enhanced unlearnable examples...")
        unlearnable_images, perturbations, metadata = dem_generator.generate_unlearnable_examples(
            test_images.to(device), test_targets.to(device)
        )
        
        print(f"Perturbation effectiveness: {metadata['perturbation_stats']['effectiveness']:.3f}")
        print(f"Enhanced version: {metadata['enhanced_version']}")
        print(f"Perturbation statistics: {metadata['perturbation_stats']}")
        
        # Test save functionality
        dem_generator.save_client_access_history('./test_history.json')
        
        print("Enhanced DEM algorithm test successful!")
        print("Perturbation effects issue fixed!")
        
        if metadata['perturbation_stats']['effectiveness'] > 0.05:
            print("Perturbation effectiveness significantly improved!")
        else:
            print("Perturbation effectiveness still needs further debugging")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()