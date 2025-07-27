import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_test():
    try:
        from config import Config
        logger.info(f"{Config.DEVICE}")
      
        from data_loader import create_data_loader
        data_loader = create_data_loader('cifar10')
        train_loader, test_loader = data_loader.get_data_loaders(batch_size=16)
        logger.info(f"{len(train_loader.dataset)} ")
        
        from models import create_model
        model = create_model('resnet18', num_classes=10)
        model.to(Config.DEVICE)
      
        from dem_algorithm import DynamicErrorMinimizingNoise
        dem_generator = DynamicErrorMinimizingNoise(dataset_name='cifar10', device=Config.DEVICE)
        
        for images, targets in train_loader:
            images = images[:4].to(Config.DEVICE) 
            targets = targets[:4].to(Config.DEVICE)
            break
        
        unlearnable_images, perturbations, metadata = dem_generator.generate_unlearnable_examples(
            images, targets, model
        )
        
        logger.info(f"[{perturbations.min():.4f}, {perturbations.max():.4f}]")
        logger.info(f"{len(metadata['client_ids'])}")
        
        metrics = dem_generator.evaluate_unlearnability(
            images, unlearnable_images, targets, model
        )
        
        logger.info(f"{metrics['clean_accuracy']:.3f}")
        logger.info(f"{metrics['unlearnable_accuracy']:.3f}")
        logger.info(f"{metrics['privacy_protection_rate']:.3f}")
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='DEM')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"{device}")
    
    from config import Config
    Config.DEVICE = device
    

if __name__ == "__main__":
    sys.exit(main())