import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Try to import config, use defaults if failed
try:
    from config import Config
    # Check if config is complete
    TRAINING_EPOCHS = getattr(Config.Training, 'EPOCHS', 10) if hasattr(Config, 'Training') else 10
    TRAINING_LR = getattr(Config.Training, 'LEARNING_RATE', 0.001) if hasattr(Config, 'Training') else 0.001
    TRAINING_WEIGHT_DECAY = getattr(Config.Training, 'WEIGHT_DECAY', 1e-4) if hasattr(Config, 'Training') else 1e-4
    TRAINING_MOMENTUM = getattr(Config.Training, 'MOMENTUM', 0.9) if hasattr(Config, 'Training') else 0.9
    TRAINING_PATIENCE = getattr(Config.Training, 'PATIENCE', 5) if hasattr(Config, 'Training') else 5
    TRAINING_MIN_DELTA = getattr(Config.Training, 'MIN_DELTA', 0.001) if hasattr(Config, 'Training') else 0.001
    TRAINING_LR_SCHEDULER = getattr(Config.Training, 'LR_SCHEDULER', 'cosine') if hasattr(Config, 'Training') else 'cosine'
    TRAINING_LR_STEP_SIZE = getattr(Config.Training, 'LR_STEP_SIZE', 30) if hasattr(Config, 'Training') else 30
    TRAINING_LR_GAMMA = getattr(Config.Training, 'LR_GAMMA', 0.1) if hasattr(Config, 'Training') else 0.1
    TRAINING_EARLY_STOPPING = getattr(Config.Training, 'EARLY_STOPPING', True) if hasattr(Config, 'Training') else True
    OUTPUT_ROOT = getattr(Config, 'OUTPUT_ROOT', './output')
    DEVICE = getattr(Config, 'DEVICE', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
except ImportError:
    # Use defaults if config cannot be imported
    TRAINING_EPOCHS = 10
    TRAINING_LR = 0.001
    TRAINING_WEIGHT_DECAY = 1e-4
    TRAINING_MOMENTUM = 0.9
    TRAINING_PATIENCE = 5
    TRAINING_MIN_DELTA = 0.001
    TRAINING_LR_SCHEDULER = 'cosine'
    TRAINING_LR_STEP_SIZE = 30
    TRAINING_LR_GAMMA = 0.1
    TRAINING_EARLY_STOPPING = True
    OUTPUT_ROOT = './output'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create basic functions if models module doesn't exist
try:
    from models import save_model, load_pretrained_model, get_model_info
except ImportError:
    def save_model(model, model_path, epoch, optimizer=None, loss=None, accuracy=None):
        """Basic model save function"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, model_path)
    
    def load_pretrained_model(model_path, model, device='cpu'):
        """Basic model load function"""
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model
    
    def get_model_info(model):
        """Basic model info function"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
        }

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping mechanism"""
    
    def __init__(self, patience: int = TRAINING_PATIENCE, 
                 min_delta: float = TRAINING_MIN_DELTA,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, val_loss: float, val_accuracy: float, 
                 model: nn.Module, epoch: int) -> bool:
        """Check if should early stop"""
        # Use validation loss as main metric
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
                
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
            
        return False
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics"""
        return {
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch
        }


class LearningRateScheduler:
    """Learning rate scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, 
                 scheduler_type: str = TRAINING_LR_SCHEDULER,
                 **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=kwargs.get('step_size', TRAINING_LR_STEP_SIZE),
                gamma=kwargs.get('gamma', TRAINING_LR_GAMMA)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', TRAINING_EPOCHS),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', TRAINING_LR_GAMMA)
            )
        else:
            self.scheduler = None
            
    def step(self, metrics: Dict[str, float] = None) -> None:
        """Update learning rate"""
        if self.scheduler is not None:
            if self.scheduler_type == 'plateau' and metrics:
                self.scheduler.step(metrics.get('val_loss', 0))
            else:
                self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate"""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]


class MetricsLogger:
    """Metrics logger"""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = defaultdict(list)
        
    def log(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics"""
        for key, value in metrics.items():
            self.history[key].append(value)
            
    def get_history(self) -> Dict[str, List[float]]:
        """Get history"""
        return dict(self.history)
    
    def plot_metrics(self, save_path: Optional[str] = None, 
                    figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot metric curves"""
        metrics = self.get_history()
        
        # Separate training and validation metrics
        train_metrics = {k: v for k, v in metrics.items() if not k.startswith('val_')}
        val_metrics = {k: v for k, v in metrics.items() if k.startswith('val_')}
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss curves
        if 'loss' in train_metrics:
            axes[0].plot(train_metrics['loss'], label='Training Loss', color='blue')
        if 'val_loss' in val_metrics:
            axes[0].plot(val_metrics['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        if 'accuracy' in train_metrics:
            axes[1].plot(train_metrics['accuracy'], label='Training Accuracy', color='blue')
        if 'val_accuracy' in val_metrics:
            axes[1].plot(val_metrics['val_accuracy'], label='Validation Accuracy', color='red')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics chart saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


class ModelTrainer:
    """Model trainer"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = DEVICE,
                 save_dir: str = OUTPUT_ROOT):
        """Initialize trainer"""
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training state
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping = None
        self.metrics_logger = MetricsLogger()
        
        # Training history
        self.current_epoch = 0
        self.training_time = 0.0
        self.best_model_path = None
        
        logger.info(f"Model trainer initialized, device: {device}")
        
        # Print model information
        try:
            model_info = get_model_info(model)
            logger.info(f"Model parameters: {model_info['total_parameters']:,}")
            logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
        except Exception as e:
            logger.warning(f"Cannot get model info: {e}")
    
    def setup_training(self, 
                      optimizer_name: str = 'adam',
                      learning_rate: float = TRAINING_LR,
                      weight_decay: float = TRAINING_WEIGHT_DECAY,
                      criterion_name: str = 'cross_entropy',
                      scheduler_config: Optional[Dict[str, Any]] = None,
                      early_stopping_config: Optional[Dict[str, Any]] = None) -> None:
        """Setup training configuration"""
        # Setup optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=TRAINING_MOMENTUM,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup loss function
        if criterion_name.lower() == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name.lower() == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {criterion_name}")
        
        # Setup learning rate scheduler
        if scheduler_config:
            self.scheduler = LearningRateScheduler(self.optimizer, **scheduler_config)
        
        # Setup early stopping
        if TRAINING_EARLY_STOPPING or early_stopping_config:
            config = early_stopping_config or {}
            self.early_stopping = EarlyStopping(**config)
        
        logger.info(f"Training config set - Optimizer: {optimizer_name}, LR: {learning_rate}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = TRAINING_EPOCHS,
              save_best: bool = True,
              save_last: bool = True,
              experiment_name: str = "experiment") -> Dict[str, Any]:
        """Train model"""
        if self.optimizer is None:
            self.setup_training()
        
        logger.info(f"Starting model training for {epochs} epochs")
        start_time = time.time()
        
        best_val_accuracy = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Check if best model
                if val_metrics['val_accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['val_accuracy']
                    best_epoch = epoch
                    
                    if save_best:
                        self.best_model_path = os.path.join(
                            self.save_dir, f"{experiment_name}_best_model.pth"
                        )
                        save_model(
                            self.model, self.best_model_path, epoch,
                            self.optimizer, val_metrics['val_loss'],
                            val_metrics['val_accuracy']
                        )
            else:
                epoch_metrics = train_metrics
                
                # If no validation set, use training accuracy
                if train_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = train_metrics['accuracy']
                    best_epoch = epoch
                    
                    if save_best:
                        self.best_model_path = os.path.join(
                            self.save_dir, f"{experiment_name}_best_model.pth"
                        )
                        save_model(
                            self.model, self.best_model_path, epoch,
                            self.optimizer, train_metrics['loss'],
                            train_metrics['accuracy']
                        )
            
            # Log metrics
            self.metrics_logger.log(epoch_metrics, epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch_metrics)
            
            # Print progress
            lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            
            if val_loader is not None:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.2f}%, "
                    f"LR: {lr:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"LR: {lr:.6f}"
                )
            
            # Early stopping check
            if self.early_stopping is not None and val_loader is not None:
                if self.early_stopping(
                    val_metrics['val_loss'], 
                    val_metrics['val_accuracy'],
                    self.model, 
                    epoch
                ):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Training finished
        end_time = time.time()
        self.training_time = end_time - start_time
        
        # Save last epoch model
        if save_last:
            last_model_path = os.path.join(
                self.save_dir, f"{experiment_name}_last_model.pth"
            )
            final_metrics = epoch_metrics
            save_model(
                self.model, last_model_path, self.current_epoch,
                self.optimizer, 
                final_metrics.get('val_loss', final_metrics['loss']),
                final_metrics.get('val_accuracy', final_metrics['accuracy'])
            )
        
        # Save training curves
        plot_path = os.path.join(self.save_dir, f"{experiment_name}_training_curves.png")
        try:
            self.metrics_logger.plot_metrics(plot_path)
        except Exception as e:
            logger.warning(f"Failed to save training curves: {e}")
        
        # Training results
        training_results = {
            'best_epoch': best_epoch,
            'best_accuracy': best_val_accuracy,
            'total_epochs': self.current_epoch + 1,
            'training_time': self.training_time,
            'metrics_history': self.metrics_logger.get_history(),
            'best_model_path': self.best_model_path,
        }
        
        if self.early_stopping is not None:
            training_results.update(self.early_stopping.get_best_metrics())
        
        logger.info(f"Training complete! Best accuracy: {best_val_accuracy:.2f}% (epoch {best_epoch+1})")
        logger.info(f"Training time: {self.training_time:.2f} seconds")
        
        return training_results
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Statistics by class
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i] == targets[i]:
                        class_correct[label] += 1
        
        # Calculate metrics
        test_loss = total_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for class_id in class_total.keys():
            if class_total[class_id] > 0:
                class_accuracies[class_id] = 100. * class_correct[class_id] / class_total[class_id]
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        logger.info(f"Test results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
        
        return results
    
    def load_model(self, model_path: str) -> None:
        """Load model"""
        self.model = load_pretrained_model(model_path, self.model, self.device)
        logger.info(f"Model loaded: {model_path}")


if __name__ == "__main__":
    # Test trainer
    print("Testing model trainer (compatibility mode)...")
    
    # Create simple test model
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.classifier = nn.Linear(32, num_classes)
            
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # Create test data
    batch_size = 16
    test_images = torch.randn(batch_size, 3, 32, 32)
    test_targets = torch.randint(0, 10, (batch_size,))
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(test_images, test_targets)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Create model and trainer
    model = SimpleModel(num_classes=10)
    trainer = ModelTrainer(model)
    trainer.setup_training(optimizer_name='adam', learning_rate=0.001)
    
    # Train model (1 epoch for testing)
    print("Starting test training...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=1,
        experiment_name="test_training"
    )
    
    print(f"Training results: {results}")
    print("Trainer test complete!")