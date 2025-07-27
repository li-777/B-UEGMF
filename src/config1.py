import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')
LOGS_ROOT = os.path.join(PROJECT_ROOT, 'logs')

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)

class Config:
    PROJECT_ROOT = PROJECT_ROOT
    DATA_ROOT = DATA_ROOT
    OUTPUT_ROOT = OUTPUT_ROOT
    LOGS_ROOT = LOGS_ROOT
  
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    AVAILABLE_DATASETS = ['cifar10', 'cifar100', 'imagenet']
    DEFAULT_DATASET = 'cifar10'
    
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
            'description': 'ImageNet dataset (需要注册账号下载)',
            'auto_download': False,
            'manual_instructions': [
                '1. 访问 http://www.image-net.org/download',
                '2. 注册账号并申请下载权限',
                '3. 下载 ILSVRC2012 训练集和验证集',
                '4. 解压到 data/imagenet/ 文件夹'
            ]
        }
    }
    
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
    
    class Training:
        EPOCHS = 10
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 1e-4
        MOMENTUM = 0.9
       
        LR_SCHEDULER = 'cosine'  # 'step', 'cosine', 'exponential'
        LR_STEP_SIZE = 30
        LR_GAMMA = 0.1
        
        EARLY_STOPPING = True
        PATIENCE = 5
        MIN_DELTA = 0.001
    
    class DEM:
        EPSILON = 16/255  
        ALPHA = 0.3      
        LAMBDA_REG = 0.1  
        NUM_STEPS = 20   
        LEARNING_RATE = 0.05  
       
        TEXT_HIDDEN_DIM = 256
        TEXT_NUM_LAYERS = 2
        BERT_MODEL = 'bert-base-uncased'
        MAX_TEXT_LENGTH = 512
        
        NUM_SURROGATE_MODELS = 3
        SURROGATE_EPOCHS = 15
        SURROGATE_LEARNING_RATE = 0.01
        SURROGATE_WEIGHT_DECAY = 1e-4
        
        EM_ITERATIONS = 40
        EM_LEARNING_RATE = 0.01
        EM_REGULARIZATION = 0.1
        
        CROSS_ATTENTION_HIDDEN_DIM = 512
        CROSS_ATTENTION_HEADS = 8
        MULTIMODAL_FUSION_WEIGHT = 0.7
        TEXT_FUSION_WEIGHT = 0.3
        
        BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
        MAX_CAPTION_LENGTH = 50
       
        MAX_SURROGATE_BATCHES = 10
        MAX_PROCESSING_BATCHES = 5
        
        VALIDATION_SAMPLES = 50
        EFFECTIVENESS_THRESHOLD = 0.1
        
        NUM_CLIENTS = 10
        CLIENT_ACCESS_WINDOW = 3600
    
    class Blockchain:
        NETWORK_ID = 31337
        GAS_LIMIT = 8000000
        GAS_PRICE = 20000000000  # 20 Gwei
       
        NUM_ACCOUNTS = 10
        INITIAL_BALANCE = 10000  # ETH
        
        CONTRACT_NAME = 'UEAccessControl'
        SOLIDITY_VERSION = '0.8.17'
       
        IPFS_API_URL = 'http://127.0.0.1:5001'
        IPFS_GATEWAY = 'http://127.0.0.1:8080'
       
        NFT_NAME = 'UnlearnableExamplesAccess'
        NFT_SYMBOL = 'UEA'
        BASE_URI = 'ipfs://'
    
    class Experiment:
        NAME = 'B-UEGMF-DEM-Experiment'
        VERSION = '2.0.0'
        
        RANDOM_SEED = 42
       
        DEM_ENABLED = True
        MULTIMODAL_FUSION_ENABLED = True
        SURROGATE_MODEL_ENABLED = True
        
        SAVE_MODELS = True
        SAVE_RESULTS = True
        SAVE_PLOTS = True
        SAVE_SURROGATE_MODELS = True
        SAVE_PERTURBATIONS = True
        SAVE_CLIENT_HISTORY = True
       
        MODES = [
            'train_clean', 
            'train_surrogates', 
            'generate_ue', 
            'train_on_ue', 
            'evaluate'
        ]
        DEFAULT_MODE = 'full'
        
        METRICS = [
            'accuracy', 
            'loss', 
            'privacy_protection_rate',
            'surrogate_ensemble_accuracy',
            'multimodal_fusion_effectiveness',
            'perturbation_imperceptibility'
        ]
    
    class Logging:
        LEVEL = 'INFO'
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
       
        LOG_TO_FILE = True
        LOG_FILE = os.path.join(LOGS_ROOT, 'experiment.log')
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        BACKUP_COUNT = 5

def validate_config():
    errors = []
    
    if Config.DEFAULT_DATASET not in Config.AVAILABLE_DATASETS:
        errors.append(f"默认数据集 {Config.DEFAULT_DATASET} 不在可用数据集列表中")
    
    if Config.DEVICE.type == 'cuda' and not torch.cuda.is_available():
        errors.append("CUDA设备不可用，但配置中指定使用CUDA")
    
    if not (0 < Config.DEM.EPSILON <= 1):
        errors.append(f"扰动约束参数 epsilon={Config.DEM.EPSILON} 应该在(0,1]范围内")
    
    if not (0 <= Config.DEM.ALPHA <= 1):
        errors.append(f"注意力参数 alpha={Config.DEM.ALPHA} 应该在[0,1]范围内")
  
    if not (1 <= Config.DEM.NUM_SURROGATE_MODELS <= 5):
        errors.append(f"代理模型数量应在[1,5]范围内: {Config.DEM.NUM_SURROGATE_MODELS}")
    
    total_weight = Config.DEM.MULTIMODAL_FUSION_WEIGHT + Config.DEM.TEXT_FUSION_WEIGHT
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"多模态融合权重之和应为1.0: {total_weight}")
    
    return errors


def print_config_summary():
    print("=" * 70)
    print("B-UEGMF Framework Configuration Summary")
    print("=" * 70)
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"Device: {Config.DEVICE}")
    print(f"Default Dataset: {Config.DEFAULT_DATASET}")
    print(f"Available Datasets: {', '.join(Config.AVAILABLE_DATASETS)}")
    print(f"DEM Epsilon: {Config.DEM.EPSILON}")
    print(f"DEM Surrogate Models: {Config.DEM.NUM_SURROGATE_MODELS}")
    print(f"Training Epochs: {Config.Training.EPOCHS}")
    print(f"Multimodal Fusion: {Config.Experiment.MULTIMODAL_FUSION_ENABLED}")
    print(f"Blockchain Network ID: {Config.Blockchain.NETWORK_ID}")
    print("=" * 70)


def adjust_config_for_device(device_type: str):
    if device_type == 'cpu':
        Config.DEM.NUM_SURROGATE_MODELS = 2
        Config.DEM.MAX_SURROGATE_BATCHES = 5
        Config.DEM.MAX_PROCESSING_BATCHES = 3
        Config.DEM.CROSS_ATTENTION_HIDDEN_DIM = 256
        Config.DEM.VALIDATION_SAMPLES = 20
        
    elif device_type == 'cuda':
        Config.DEM.NUM_SURROGATE_MODELS = 3
        Config.DEM.MAX_SURROGATE_BATCHES = 10
        Config.DEM.MAX_PROCESSING_BATCHES = 5
        Config.DEM.CROSS_ATTENTION_HIDDEN_DIM = 512
        Config.DEM.VALIDATION_SAMPLES = 50


if __name__ == "__main__":
    config_errors = validate_config()
    print_config_summary()
   
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    adjust_config_for_device(device_type)