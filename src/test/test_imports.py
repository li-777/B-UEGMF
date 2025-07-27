import sys
import traceback

def test_import(module_name, description):
    try:
        if module_name == 'config':
            from config import Config, validate_config, print_config_summary
            print(f" {description}")
            return True
        elif module_name == 'utils':
            from utils import setup_logging, set_random_seed
            print(f" {description}")
            return True
        elif module_name == 'data_loader':
            from data_loader import create_data_loader
            print(f"{description}")
            return True
        elif module_name == 'models':
            from models import create_model
            print(f" {description}")
            return True
        elif module_name == 'dem_algorithm':
            from dem_algorithm import DynamicErrorMinimizingNoise
            print(f" {description}")
            return True
        elif module_name == 'trainer':
            from trainer import ModelTrainer
            print(f" {description}")
            return True
        elif module_name == 'evaluator':
            from evaluator1 import ComprehensiveEvaluator
            print(f" {description}")
            return True
        elif module_name == 'blockchain_simulation':
            from blockchain_simulation import SimulatedBlockchain
            print(f" {description}")
            return True
        elif module_name == 'smart_contracts':
            from smart_contracts import deploy_access_control_contract
            print(f" {description}")
            return True
        else:
            exec(f"import {module_name}")
            print(f" {description}")
            return True
    except Exception as e:
        print(f" {description}: {e}")
        traceback.print_exc()
        return False

def main():
    
    basic_modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    for module, desc in basic_modules:
        test_import(module, desc)
   
    project_modules = [
        ('config', ''),
        ('utils', ''),
        ('data_loader', ''),
        ('models', ''),
        ('dem_algorithm', ''),
        ('trainer', ''),
        ('evaluator', ''),
        ('blockchain_simulation', ''),
        ('smart_contracts', ''),
    ]
    
    success_count = 0
    for module, desc in project_modules:
        if test_import(module, desc):
            success_count += 1
    
    print(f"{success_count}/{len(project_modules)}")
    

if __name__ == "__main__":
    sys.exit(main())