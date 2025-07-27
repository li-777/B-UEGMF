import os
import json
import logging
import random
import numpy as np
import torch
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = Config.Logging.LEVEL,
                 log_file: Optional[str] = None,
                 log_to_console: bool = True) -> None:
    """
    Configure logging settings
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        log_to_console: Whether to output to console
    """
    # Create log directory
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    handlers = []
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(Config.Logging.FORMAT)
        )
        handlers.append(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(Config.Logging.FORMAT)
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format=Config.Logging.FORMAT,
        force=True
    )
    
    logger.info(f"Logging system configured - Level: {log_level}")


def set_random_seed(seed: int = Config.Experiment.RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set: {seed}")


def create_experiment_dir(base_dir: str = Config.OUTPUT_ROOT,
                         experiment_name: str = None) -> str:
    """
    Create experiment directory
    
    Args:
        base_dir: Base directory path
        experiment_name: Experiment name
        
    Returns:
        str: Path to experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'plots', 'logs', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def save_config(config_dict: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to file
    
    Args:
        config_dict: Configuration dictionary
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Configuration saved: {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Configuration loaded: {filepath}")
    return config


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict[str, float]: Model size information
    """
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024  # MB
    
    return {
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': all_size,
        'param_count': param_sum,
        'buffer_count': buffer_sum
    }


def format_time(seconds: float) -> str:
    """
    Format time duration
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def get_device_info() -> Dict[str, Any]:
    """
    Get device information
    
    Returns:
        Dict[str, Any]: Device information
    """
    info = {
        'device': str(Config.DEVICE),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_current_device': torch.cuda.current_device(),
            'cuda_device_name': torch.cuda.get_device_name(),
            'cuda_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cuda_memory_reserved': torch.cuda.memory_reserved() / 1024**3,   # GB
        })
    
    return info


def check_memory_usage() -> Dict[str, float]:
    """
    Check memory usage
    
    Returns:
        Dict[str, float]: Memory usage information
    """
    import psutil
    
    # System memory
    memory = psutil.virtual_memory()
    
    info = {
        'system_memory_total_gb': memory.total / 1024**3,
        'system_memory_used_gb': memory.used / 1024**3,
        'system_memory_percent': memory.percent,
    }
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        info.update({
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        })
    
    return info


def create_data_hash(data: Union[torch.Tensor, np.ndarray, str]) -> str:
    """
    Create data hash
    
    Args:
        data: Input data
        
    Returns:
        str: Data hash value
    """
    if isinstance(data, torch.Tensor):
        data_bytes = data.detach().cpu().numpy().tobytes()
    elif isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    elif isinstance(data, str):
        data_bytes = data.encode()
    else:
        data_bytes = str(data).encode()
    
    return hashlib.sha256(data_bytes).hexdigest()


def save_tensor(tensor: torch.Tensor, filepath: str, 
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save tensor to file
    
    Args:
        tensor: Tensor to save
        filepath: Path to save file
        metadata: Metadata dictionary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_data = {
        'tensor': tensor,
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': str(tensor.device),
        'save_time': datetime.now().isoformat(),
    }
    
    if metadata:
        save_data['metadata'] = metadata
    
    torch.save(save_data, filepath)
    logger.info(f"Tensor saved: {filepath}")


def load_tensor(filepath: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load tensor from file
    
    Args:
        filepath: Path to file
        
    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]: (Tensor, metadata)
    """
    data = torch.load(filepath, map_location='cpu')
    tensor = data['tensor']
    metadata = data.get('metadata', {})
    
    logger.info(f"Tensor loaded: {filepath}")
    return tensor, metadata


def plot_tensor_histogram(tensor: torch.Tensor, title: str = "Tensor Histogram",
                         bins: int = 50, save_path: Optional[str] = None) -> None:
    """
    Plot tensor histogram
    
    Args:
        tensor: Input tensor
        title: Plot title
        bins: Number of histogram bins
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    data = tensor.detach().cpu().numpy().flatten()
    plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.4f}')
    plt.axvline(np.median(data), color='green', linestyle='--', label=f'Median: {np.median(data):.4f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Histogram saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor,
                   names: List[str] = None) -> Dict[str, float]:
    """
    Compare two tensors
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        names: Tensor names
        
    Returns:
        Dict[str, float]: Comparison results
    """
    if names is None:
        names = ['Tensor1', 'Tensor2']
    
    # Ensure shapes match
    if tensor1.shape != tensor2.shape:
        logger.warning(f"Tensor shapes don't match: {tensor1.shape} vs {tensor2.shape}")
        return {}
    
    # Calculate various distance and similarity metrics
    mse = torch.mean((tensor1 - tensor2) ** 2).item()
    mae = torch.mean(torch.abs(tensor1 - tensor2)).item()
    
    # Cosine similarity
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    cos_sim = torch.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
    
    # Pearson correlation
    corr = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1].item()
    
    # Maximum difference
    max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'max_difference': max_diff,
        'cosine_similarity': cos_sim,
        'pearson_correlation': corr,
        'l2_norm_diff': torch.norm(tensor1 - tensor2, p=2).item(),
        'l1_norm_diff': torch.norm(tensor1 - tensor2, p=1).item(),
    }


def create_summary_table(results: Dict[str, Any], 
                        save_path: Optional[str] = None) -> str:
    """
    Create results summary table
    
    Args:
        results: Results dictionary
        save_path: Path to save table
        
    Returns:
        str: Table string
    """
    import pandas as pd
    
    # Flatten nested dictionary
    flat_results = {}
    
    def flatten_dict(d, parent_key='', sep='_'):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, new_key, sep=sep)
            else:
                flat_results[new_key] = v
    
    flatten_dict(results)
    
    # Create DataFrame
    df = pd.DataFrame(list(flat_results.items()), columns=['Metric', 'Value'])
    
    # Format values
    def format_value(val):
        if isinstance(val, float):
            if abs(val) < 0.001:
                return f"{val:.2e}"
            else:
                return f"{val:.4f}"
        return str(val)
    
    df['Value'] = df['Value'].apply(format_value)
    
    # Save table
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Summary table saved: {save_path}")
    
    # Return table string
    return df.to_string(index=False)


def validate_experiment_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate experiment configuration
    
    Args:
        config: Experiment configuration
        
    Returns:
        List[str]: List of error messages
    """
    errors = []
    
    # Check required keys
    required_keys = [
        'dataset_name', 'model_architecture', 'batch_size',
        'num_epochs', 'learning_rate'
    ]
    
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration: {key}")
    
    # Check value ranges
    if 'learning_rate' in config:
        lr = config['learning_rate']
        if not (0 < lr < 1):
            errors.append(f"Learning rate should be in (0,1) range: {lr}")
    
    if 'batch_size' in config:
        bs = config['batch_size']
        if not (1 <= bs <= 1024):
            errors.append(f"Batch size should be in [1,1024] range: {bs}")
    
    # Check dataset name
    if 'dataset_name' in config:
        if config['dataset_name'] not in Config.AVAILABLE_DATASETS:
            errors.append(f"Unsupported dataset: {config['dataset_name']}")
    
    return errors


def benchmark_function(func, *args, num_runs: int = 5, **kwargs) -> Dict[str, float]:
    """
    Benchmark function execution
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of runs
        **kwargs: Function keyword arguments
        
    Returns:
        Dict[str, float]: Benchmark results
    """
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
        'num_runs': num_runs
    }


class ProgressTracker:
    """Progress tracking utility"""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, step: int = 1) -> None:
        """Update progress"""
        self.current_step += step
        progress = self.current_step / self.total_steps
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            eta = elapsed * (self.total_steps - self.current_step) / self.current_step
            eta_str = format_time(eta)
        else:
            eta_str = "Unknown"
        
        logger.info(
            f"{self.description}: {self.current_step}/{self.total_steps} "
            f"({progress*100:.1f}%) - ETA: {eta_str}"
        )
    
    def finish(self) -> None:
        """Finish progress tracking"""
        total_time = time.time() - self.start_time
        logger.info(f"{self.description} completed! Total time: {format_time(total_time)}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Setup logging
    setup_logging()
    
    # Set random seed
    set_random_seed(42)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(experiment_name="test_utils")
    print(f"Experiment directory: {exp_dir}")
    
    # Get device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Test tensor operations
    tensor1 = torch.randn(100, 100)
    tensor2 = torch.randn(100, 100)
    
    # Compare tensors
    comparison = compare_tensors(tensor1, tensor2)
    print(f"Tensor comparison: {comparison}")
    
    # Test progress tracker
    tracker = ProgressTracker(10, "Test progress")
    for i in range(10):
        time.sleep(0.1)  # Simulate work
        tracker.update()
    tracker.finish()
    print("Utility function testing completed!")