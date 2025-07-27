import os
import torch

# Global path configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')
LOGS_ROOT = os.path.join(PROJECT_ROOT, 'logs')

# Create necessary folders
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)

class Config:
    # ========== Basic Configuration ==========
    PROJECT_ROOT = PROJECT_ROOT
    DATA_ROOT = DATA_ROOT
    OUTPUT_ROOT = OUTPUT_ROOT
    LOGS_ROOT = LOGS_ROOT
    
    # ========== Device Configuration ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # ========== Dataset Configuration ==========
    AVAILABLE_DATASETS = ['cifar10', 'cifar100', 'imagenet']
    DEFAULT_DATASET = 'cifar10'
    
    # Dataset download URLs and descriptions
    DATASET_URLS = {
        'cifar10': {
            'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'description': 'CIFAR-10 dataset (10 classes, 32x32 images)',
            'auto_download': True
        },
        'cifar100': {
            'url': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 
            'description': 'CIFAR-100 dataset (100 classes, 32x32 images)',
            'auto_download': True
        },
        'imagenet': {
            'url': 'http://www.image-net.org/download',
            'description': 'ImageNet dataset (requires account registration for download)',
            'auto_download': False,
            'manual_instructions': [
                '1. Visit http://www.image-net.org/download',
                '2. Register account and apply for download permission',
                '3. Download ILSVRC2012 training and validation sets',
                '4. Extract to data/imagenet/ folder'
            ]
        }
    }
    
    # Dataset parameters
    DATASET_PARAMS = {
        'cifar10': {
            'num_classes': 10,
            'image_size': (32, 32),
            'channels': 3,
            'batch_size': 128,
            'train_size': None
        },
        'cifar100': {
            'num_classes': 100,
            'image_size': (32, 32), 
            'channels': 3,
            'batch_size': 64,
            'train_size': None
        },
        'imagenet': {
            'num_classes': 1000,
            'image_size': (224, 224),
            'channels': 3,
            'batch_size': 32,
            'train_size': 10000
        }
    }
    
    # ========== Model Training Parameters ==========
    class Training:
        EPOCHS = 10
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 1e-4
        MOMENTUM = 0.9
        
        # Learning rate scheduling
        LR_SCHEDULER = 'cosine'  # 'step', 'cosine', 'exponential'
        LR_STEP_SIZE = 30
        LR_GAMMA = 0.1
        
        # Early stopping
        EARLY_STOPPING = True
        PATIENCE = 5
        MIN_DELTA = 0.001
    
    # ========== DEM Algorithm Parameters ==========
    class DEM:
        # Basic perturbation parameters
        EPSILON = 16/255  
        ALPHA = 0.3      
        LAMBDA_REG = 0.1  
        NUM_STEPS = 20   
        LEARNING_RATE = 0.05  
        
        # Text feature extractor parameters
        TEXT_HIDDEN_DIM = 256
        TEXT_NUM_LAYERS = 2
        BERT_MODEL = 'bert-base-uncased'
        MAX_TEXT_LENGTH = 512
        
        # EM surrogate model parameters
        NUM_SURROGATE_MODELS = 3
        SURROGATE_EPOCHS = 15
        SURROGATE_LEARNING_RATE = 0.01
        SURROGATE_WEIGHT_DECAY = 1e-4
        
        # EM perturbation generation parameters
        EM_ITERATIONS = 40
        EM_LEARNING_RATE = 0.01
        EM_REGULARIZATION = 0.1
        
        # Multimodal fusion parameters
        CROSS_ATTENTION_HIDDEN_DIM = 512
        CROSS_ATTENTION_HEADS = 8
        MULTIMODAL_FUSION_WEIGHT = 0.7
        TEXT_FUSION_WEIGHT = 0.3
        
        # BLIP model parameters
        BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
        MAX_CAPTION_LENGTH = 50
        
        # Batch processing parameters
        MAX_SURROGATE_BATCHES = 10
        MAX_PROCESSING_BATCHES = 5
        
        # Evaluation parameters
        VALIDATION_SAMPLES = 50
        EFFECTIVENESS_THRESHOLD = 0.1
        
        # Client simulation parameters
        NUM_CLIENTS = 10
        CLIENT_ACCESS_WINDOW = 3600
    
    # ========== Blockchain Configuration ==========
    class Blockchain:
        # Network configuration
        NETWORK_ID = 31337
        GAS_LIMIT = 8000000
        GAS_PRICE = 20000000000  # 20 Gwei
        
        # Account configuration
        NUM_ACCOUNTS = 10
        INITIAL_BALANCE = 10000  # ETH
        
        # Smart contract configuration
        CONTRACT_NAME = 'UEAccessControl'
        SOLIDITY_VERSION = '0.8.17'
        
        # IPFS configuration
        IPFS_API_URL = 'http://127.0.0.1:5001'
        IPFS_GATEWAY = 'http://127.0.0.1:8080'
        
        # NFT configuration
        NFT_NAME = 'UnlearnableExamplesAccess'
        NFT_SYMBOL = 'UEA'
        BASE_URI = 'ipfs://'
    
    # ========== Experiment Configuration ==========
    class Experiment:
        # Experiment name and version
        NAME = 'B-UEGMF-DEM-Experiment'
        VERSION = '2.0.0'
        
        # Random seed
        RANDOM_SEED = 42
        
        # DEM experiment configuration
        DEM_ENABLED = True
        MULTIMODAL_FUSION_ENABLED = True
        SURROGATE_MODEL_ENABLED = True
        
        # Save configuration
        SAVE_MODELS = True
        SAVE_RESULTS = True
        SAVE_PLOTS = True
        SAVE_SURROGATE_MODELS = True
        SAVE_PERTURBATIONS = True
        SAVE_CLIENT_HISTORY = True
        
        # Experiment modes
        MODES = [
            'train_clean', 
            'train_surrogates', 
            'generate_ue', 
            'train_on_ue', 
            'evaluate'
        ]
        DEFAULT_MODE = 'full'
        
        # Evaluation metrics
        METRICS = [
            'accuracy', 
            'loss', 
            'privacy_protection_rate',
            'surrogate_ensemble_accuracy',
            'multimodal_fusion_effectiveness',
            'perturbation_imperceptibility'
        ]
    
    # ========== Logging Configuration ==========
    class Logging:
        LEVEL = 'INFO'
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File configuration
        LOG_TO_FILE = True
        LOG_FILE = os.path.join(LOGS_ROOT, 'experiment.log')
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        BACKUP_COUNT = 5


# Helper functions for configuration validation
def validate_config():
    """Validate configuration parameter validity"""
    errors = []
    
    # Check dataset configuration
    if Config.DEFAULT_DATASET not in Config.AVAILABLE_DATASETS:
        errors.append(f"Default dataset {Config.DEFAULT_DATASET} not in available datasets list")
    
    # Check device availability
    if Config.DEVICE.type == 'cuda' and not torch.cuda.is_available():
        errors.append("CUDA device not available, but configuration specifies CUDA usage")
    
    # Check perturbation parameters
    if not (0 < Config.DEM.EPSILON <= 1):
        errors.append(f"Perturbation constraint epsilon={Config.DEM.EPSILON} should be in (0,1] range")
    
    if not (0 <= Config.DEM.ALPHA <= 1):
        errors.append(f"Attention parameter alpha={Config.DEM.ALPHA} should be in [0,1] range")
    
    # Check surrogate model parameters
    if not (1 <= Config.DEM.NUM_SURROGATE_MODELS <= 5):
        errors.append(f"Number of surrogate models should be in [1,5] range: {Config.DEM.NUM_SURROGATE_MODELS}")
    
    # Check multimodal fusion weights
    total_weight = Config.DEM.MULTIMODAL_FUSION_WEIGHT + Config.DEM.TEXT_FUSION_WEIGHT
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Multimodal fusion weights should sum to 1.0: {total_weight}")
    
    return errors


def print_config_summary():
    """Print configuration summary"""
    print("=" * 70)
    print("B-UEGMF Framework Configuration Summary")
    print("=" * 70)
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"Device: {Config.DEVICE}")
    print(f"Default Dataset: {Config.DEFAULT_DATASET}")
    print(f"DEM Epsilon: {Config.DEM.EPSILON}")
    print(f"DEM Surrogate Models: {Config.DEM.NUM_SURROGATE_MODELS}")
    print(f"Training Epochs: {Config.Training.EPOCHS}")
    print("=" * 70)


def adjust_config_for_device(device_type: str):
    """Adjust configuration based on device type"""
    if device_type == 'cpu':
        # Reduce resource consumption for CPU environment
        Config.DEM.NUM_SURROGATE_MODELS = 2
        Config.DEM.MAX_SURROGATE_BATCHES = 5
        Config.DEM.MAX_PROCESSING_BATCHES = 3
        Config.DEM.CROSS_ATTENTION_HIDDEN_DIM = 256
        Config.DEM.VALIDATION_SAMPLES = 20
        print("Configuration adjusted for CPU environment")
        
    elif device_type == 'cuda':
        # Use more resources for GPU environment
        Config.DEM.NUM_SURROGATE_MODELS = 3
        Config.DEM.MAX_SURROGATE_BATCHES = 10
        Config.DEM.MAX_PROCESSING_BATCHES = 5
        Config.DEM.CROSS_ATTENTION_HIDDEN_DIM = 512
        Config.DEM.VALIDATION_SAMPLES = 50
        print("Configuration optimized for GPU environment")


if __name__ == "__main__":
    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        print("Configuration validation failed:")
        for error in config_errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed")
        
    print_config_summary()
    
    # Adjust configuration based on current device
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    adjust_config_for_device(device_type)