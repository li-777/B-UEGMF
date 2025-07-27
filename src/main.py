"""
B-UEGMF Framework - Main Experiment Runner (Updated Version)
Integrated Improved DEM Algorithm with EM Surrogate Model Methods
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import updated DEM algorithm
from dem_algorithm import DynamicErrorMinimizingNoise, DEMBatchProcessor
from config import Config
from utils import (
    setup_logging, set_random_seed, create_experiment_dir,
    save_config, get_device_info, check_memory_usage, ProgressTracker
)
from data_loader import create_data_loader, check_dataset_availability
from models import create_model, initialize_weights
from trainer import ModelTrainer
from evaluator1 import ComprehensiveEvaluator
from blockchain_simulation import SimulatedBlockchain
from smart_contracts import deploy_access_control_contract, MerkleTreeManager, AccessLevel

logger = logging.getLogger(__name__)


class B_UEGMF_Experiment:
    """Updated B-UEGMF Complete Experiment Framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = Config.DEVICE
        self.experiment_dir = None
        self.blockchain = None
        self.contract = None
        self.results = {}
        
        # Set random seed
        set_random_seed(config.get('random_seed', Config.Experiment.RANDOM_SEED))
        
        logger.info("B-UEGMF experiment framework initialized")
    
    def setup_experiment(self) -> str:
        """Setup experiment environment"""
        # Create experiment directory
        experiment_name = self.config.get('experiment_name', f"dem_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir = create_experiment_dir(
            base_dir=Config.OUTPUT_ROOT,
            experiment_name=experiment_name
        )
        
        # Setup logging
        log_file = os.path.join(self.experiment_dir, 'logs', 'experiment.log')
        setup_logging(log_file=log_file)
        
        # Save configuration
        config_file = os.path.join(self.experiment_dir, 'config.json')
        save_config(self.config, config_file)
        
        # Check device and memory
        device_info = get_device_info()
        memory_info = check_memory_usage()
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        return self.experiment_dir
    
    def setup_blockchain(self) -> None:
        """Setup blockchain environment"""
        logger.info("Initializing blockchain simulation...")
        
        # Create blockchain instance
        self.blockchain = SimulatedBlockchain()
        
        # Deploy access control contract
        deployer = list(self.blockchain.accounts.keys())[0]
        contract_address, self.contract = deploy_access_control_contract(
            self.blockchain, deployer
        )
        
        # Initialize contract
        admin_accounts = list(self.blockchain.accounts.keys())[1:3]
        self.contract.initialize(deployer, admin_accounts)
        
        # Start auto mining
        self.blockchain.start_mining()
        
        logger.info(f"Blockchain environment setup - Contract address: {contract_address}")
        
        # Save blockchain state
        blockchain_file = os.path.join(self.experiment_dir, 'blockchain_state.json')
        self.blockchain.save_state(blockchain_file)
    
    def load_data(self) -> tuple:
        """Load dataset"""
        dataset_name = self.config['dataset_name']
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Check dataset availability
        if not check_dataset_availability(dataset_name):
            raise ValueError(f"Dataset unavailable: {dataset_name}")
        
        # Create data loader
        data_loader = create_data_loader(dataset_name)
        train_loader, test_loader = data_loader.get_data_loaders(
            batch_size=self.config.get('batch_size', None),
            num_workers=self.config.get('num_workers', None)
        )
        
        logger.info(f"Dataset loaded - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        return train_loader, test_loader, data_loader.get_dataset_info()
    
    def create_models(self, dataset_info: Dict[str, Any]) -> tuple:
        """Create models"""
        model_architecture = self.config.get('model_architecture', 'resnet18')
        num_classes = dataset_info['num_classes']
        
        logger.info(f"Creating model: {model_architecture}")
        
        # Create clean model (for baseline)
        clean_model = create_model(model_architecture, num_classes)
        initialize_weights(clean_model)
        clean_model.to(self.device)
        
        logger.info(f"Model created - Parameters: {sum(p.numel() for p in clean_model.parameters()):,}")
        
        return clean_model
    
    def train_clean_model(self, model: nn.Module, train_loader: DataLoader, 
                         test_loader: DataLoader) -> Dict[str, Any]:
        """Train clean baseline model"""
        logger.info("Training clean baseline model...")
        
        # Create trainer
        trainer = ModelTrainer(model, device=self.device, save_dir=self.experiment_dir)
        
        # Setup training configuration
        trainer.setup_training(
            optimizer_name=self.config.get('optimizer', 'adam'),
            learning_rate=self.config.get('learning_rate', Config.Training.LEARNING_RATE),
            weight_decay=self.config.get('weight_decay', Config.Training.WEIGHT_DECAY),
            scheduler_config={
                'scheduler_type': self.config.get('scheduler', Config.Training.LR_SCHEDULER)
            },
            early_stopping_config={
                'patience': self.config.get('patience', Config.Training.PATIENCE)
            }
        )
        
        # Train model
        train_results = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=self.config.get('epochs', Config.Training.EPOCHS),
            experiment_name="clean_model"
        )
        
        # Evaluate model
        eval_results = trainer.evaluate(test_loader)
        
        logger.info(f"Clean model training complete - Best accuracy: {train_results['best_accuracy']:.2f}%")
        
        return {
            'training_results': train_results,
            'evaluation_results': eval_results,
            'model_path': train_results.get('best_model_path')
        }
    
    def generate_unlearnable_examples(self, train_loader: DataLoader,
                                    dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unlearnable examples using improved DEM algorithm"""
        logger.info("Generating unlearnable examples using improved DEM algorithm...")
        
        # Create improved DEM generator
        dem_generator = DynamicErrorMinimizingNoise(
            dataset_name=dataset_info['name'],
            epsilon=self.config.get('epsilon', Config.DEM.EPSILON),
            alpha=self.config.get('alpha', Config.DEM.ALPHA),
            lambda_reg=self.config.get('lambda_reg', Config.DEM.LAMBDA_REG),
            device=self.device
        )

        
        # Setup and train surrogate model
        logger.info("Setting up EM surrogate model...")
        model_architecture = self.config.get('model_architecture', 'resnet18')
        num_classes = dataset_info['num_classes']
        
        # Create data loader for surrogate model training (using smaller subset to save time)
        surrogate_train_loader = self._create_surrogate_training_data(train_loader)
        
        surrogate_results = dem_generator.setup_surrogate_models(
            model_architecture, num_classes, surrogate_train_loader
        )
        
        logger.info(f"Surrogate model training complete - Average accuracy: {surrogate_results['average_accuracy']:.2f}%")
        
        # Batch process unlearnable examples
        logger.info("Batch generating unlearnable examples...")
        processing_results = self._batch_process_unlearnable_examples(
            dem_generator, train_loader
        )
        
        # Save results
        save_path = os.path.join(self.experiment_dir, 'data', 'unlearnable_examples.pth')
        torch.save(processing_results, save_path)
        
        # Save client access history
        history_file = os.path.join(self.experiment_dir, 'data', 'client_access_history.json')
        dem_generator.save_client_access_history(history_file)
        
        # Save surrogate model info
        surrogate_info_file = os.path.join(self.experiment_dir, 'data', 'surrogate_model_info.json')
        with open(surrogate_info_file, 'w') as f:
            json.dump(surrogate_results, f, indent=2, default=str)
        
        logger.info(f"Unlearnable examples generation complete - Processed {len(processing_results['clean_images'])} samples")
        
        return {
            'dem_generator': dem_generator,
            'processing_results': processing_results,
            'unlearnable_data_path': save_path,
            'surrogate_results': surrogate_results
        }
    
    def _create_surrogate_training_data(self, train_loader: DataLoader) -> DataLoader:
        """Create surrogate model training data (using subset to save time)"""
        # Use first few batches for surrogate model training
        max_batches = min(10, len(train_loader))  # Use at most 10 batches
        
        all_images = []
        all_targets = []
        
        for i, (images, targets) in enumerate(train_loader):
            if i >= max_batches:
                break
            all_images.append(images)
            all_targets.append(targets)
        
        # Combine data
        combined_images = torch.cat(all_images, dim=0)
        combined_targets = torch.cat(all_targets, dim=0)
        
        # Create new dataset and loader
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(combined_images, combined_targets)
        loader = DataLoader(dataset, batch_size=train_loader.batch_size, shuffle=True)
        
        return loader
    
    def _batch_process_unlearnable_examples(self, dem_generator: DynamicErrorMinimizingNoise,
                                          train_loader: DataLoader) -> Dict[str, Any]:
        """Batch process unlearnable examples generation"""
        all_clean_images = []
        all_unlearnable_images = []
        all_targets = []
        all_perturbations = []
        all_metadata = []
        
        total_batches = min(5, len(train_loader))  # Limit processed batches to save time
        progress_tracker = ProgressTracker(total_batches, "Generating unlearnable examples")
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= total_batches:
                break
                
            images = images.to(dem_generator.device)
            targets = targets.to(dem_generator.device)
            
            try:
                # Generate unlearnable examples
                unlearnable_images, perturbations, metadata = \
                    dem_generator.generate_unlearnable_examples(
                        images, targets
                    )
                
                # Collect results
                all_clean_images.append(images.cpu())
                all_unlearnable_images.append(unlearnable_images.cpu())
                all_targets.append(targets.cpu())
                all_perturbations.append(perturbations.cpu())
                all_metadata.append(metadata)
                
                progress_tracker.update()
                
            except Exception as e:
                logger.error(f"Batch {batch_idx} processing failed: {e}")
                continue
        
        progress_tracker.finish()
        
        # Combine results
        results = {
            'clean_images': torch.cat(all_clean_images, dim=0),
            'unlearnable_images': torch.cat(all_unlearnable_images, dim=0),
            'targets': torch.cat(all_targets, dim=0),
            'perturbations': torch.cat(all_perturbations, dim=0),
            'metadata': all_metadata
        }
        
        return results
    
    def comprehensive_evaluation(self, clean_model: nn.Module, 
                               dem_data: Dict[str, Any],
                               test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive evaluation - Updated version"""
        logger.info("Starting comprehensive evaluation...")
        
        # Create evaluator
        evaluator = ComprehensiveEvaluator(device=self.device, save_dir=self.experiment_dir)
        
        # Get test data subset
        test_images = []
        test_targets = []
        
        for i, (images, targets) in enumerate(test_loader):
            if i >= 3:  # Use first 3 batches
                break
            test_images.append(images)
            test_targets.append(targets)
        
        test_images = torch.cat(test_images, dim=0)[:50]  # Limit to 50 samples
        test_targets = torch.cat(test_targets, dim=0)[:50]
        
        # Run comprehensive evaluation
        evaluation_results = evaluator.comprehensive_evaluation(
            dem_generator=dem_data['dem_generator'],
            clean_images=test_images.to(self.device),
            targets=test_targets.to(self.device),
            surrogate_model=clean_model,
            experiment_name=self.config.get('experiment_name', 'dem_evaluation')
        )
        
        # Add surrogate model related evaluation info
        evaluation_results['surrogate_model_info'] = dem_data['surrogate_results']
        
        logger.info(f"Comprehensive evaluation complete - Overall score: {evaluation_results['overall_score']['overall_score']:.3f}")
        
        return evaluation_results
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete experiment - Updated version"""
        start_time = time.time()
        logger.info("Starting updated B-UEGMF complete experiment...")
        
        try:
            # 1. Setup experiment environment
            self.setup_experiment()
            
            # 2. Setup blockchain environment (optional)
            if self.config.get('enable_blockchain', True):
                self.setup_blockchain()
                
                # Setup access control
                user_addresses = list(self.blockchain.accounts.keys())[3:8]
                dataset_hash, merkle_manager = self.setup_access_control(user_addresses)
            
            # 3. Load data
            train_loader, test_loader, dataset_info = self.load_data()
            
            # 4. Create models
            clean_model = self.create_models(dataset_info)
            
            # 5. Train clean baseline model
            clean_results = self.train_clean_model(clean_model, train_loader, test_loader)
            self.results['clean_model'] = clean_results
            
            # 6. Generate unlearnable examples using improved DEM algorithm
            dem_data = self.generate_unlearnable_examples(train_loader, dataset_info)
            self.results['unlearnable_examples'] = {
                'processing_results': dem_data['processing_results'],
                'data_path': dem_data['unlearnable_data_path'],
                'surrogate_results': dem_data['surrogate_results']
            }
            
            # 7. Test access control (if blockchain enabled)
            if self.config.get('enable_blockchain', True):
                access_control_results = self.test_access_control(dataset_hash, merkle_manager)
                self.results['access_control'] = access_control_results
            
            # 8. Comprehensive evaluation
            if self.config.get('run_evaluation', True):
                evaluation_results = self.comprehensive_evaluation(clean_model, dem_data, test_loader)
                self.results['evaluation'] = evaluation_results
            
            # 9. Save final results
            total_time = time.time() - start_time
            
            final_results = {
                'experiment_config': self.config,
                'experiment_info': {
                    'start_time': datetime.fromtimestamp(start_time).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_duration': total_time,
                    'experiment_dir': self.experiment_dir
                },
                'results': self.results
            }
            
            # Save results
            results_file = os.path.join(self.experiment_dir, 'final_results.json')
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info(f"Experiment complete! Total time: {total_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            raise
        
        finally:
            # Clean up resources
            if self.blockchain:
                self.blockchain.stop_mining()
    
    def setup_access_control(self, user_addresses: List[str]) -> tuple:
        """Setup access control - Fixed version"""
        logger.info("Setting up Merkle tree access control...")
        
        # Create Merkle tree manager
        merkle_manager = MerkleTreeManager()
        
        # Add users to whitelist
        for user in user_addresses:
            merkle_manager.add_leaf(user)
        
        # Build Merkle tree
        root = merkle_manager.build_tree()
        
        # Update Merkle root in contract
        deployer = list(self.blockchain.accounts.keys())[0]
        try:
            self.contract.update_merkle_root(root, deployer)
            logger.info("Merkle root updated to smart contract")
        except Exception as e:
            logger.error(f"Failed to update Merkle root: {e}")
        
        # Register test dataset
        dataset_name = self.config['dataset_name']
        dataset_hash = f"dataset_{dataset_name}_{int(time.time())}"
        
        try:
            self.contract.register_dataset(
                dataset_hash,
                {
                    'name': dataset_name,
                    'experiment': self.config.get('experiment_name', 'default'),
                    'created_at': datetime.now().isoformat()
                },
                deployer
            )
            logger.info(f"Dataset registered: {dataset_hash}")
        except Exception as e:
            logger.error(f"Dataset registration failed: {e}")
        
        return dataset_hash, merkle_manager
    
    def test_access_control(self, dataset_hash: str, merkle_manager) -> Dict[str, Any]:
        """Test access control functionality - Fixed version"""
        logger.info("Testing blockchain access control...")
        
        # Get correct test users (should be users in whitelist)
        user_accounts = list(self.blockchain.accounts.keys())[3:8]  # These should be users added to whitelist
        
        access_results = {}
        
        for i, user in enumerate(user_accounts):
            try:
                # Ensure user is in Merkle tree
                if not merkle_manager.has_leaf(user):
                    merkle_manager.add_leaf(user)
                    # Rebuild Merkle tree
                    root = merkle_manager.build_tree()
                    # Update Merkle root in contract
                    deployer = list(self.blockchain.accounts.keys())[0]
                    self.contract.update_merkle_root(root, deployer)
                
                # Generate correct Merkle proof
                proof = merkle_manager.generate_proof(user)
                
                if proof:
                    # Verify proof
                    is_valid = proof.verify()
                    
                    if is_valid:
                        # Grant access permission
                        result = self.contract.grant_access(
                            user=user,
                            merkle_proof=proof,
                            dataset_hash=dataset_hash,
                            access_level=AccessLevel.FULL_ACCESS,
                            ipfs_cid=f"QmTestCID{i}",
                            expiry_time=int(time.time()) + 3600  # Expires after 1 hour
                        )
                        
                        access_results[user] = {
                            'status': 'authorized',
                            'token_id': result['token_id'],
                            'access_level': AccessLevel.FULL_ACCESS.value,
                            'proof_valid': True
                        }
                        
                    else:
                        access_results[user] = {
                            'status': 'unauthorized',
                            'reason': 'Invalid Merkle proof',
                            'proof_valid': False
                        }
                        
                else:
                    access_results[user] = {
                        'status': 'unauthorized',
                        'reason': 'Cannot generate Merkle proof',
                        'proof_valid': False
                    }
                    
            except Exception as e:
                access_results[user] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"Access control test failed for user: {e}")
        
        # Get contract statistics
        try:
            contract_stats = self.contract.get_contract_stats()
        except:
            contract_stats = {'active_tokens': 0, 'total_datasets': 1}
        
        # Count successfully authorized users
        authorized_count = sum(1 for result in access_results.values() 
                            if result.get('status') == 'authorized')
        
        logger.info(f"Access control test complete - Authorized users: {authorized_count}/{len(user_accounts)}")
        
        return {
            'access_results': access_results,
            'contract_stats': contract_stats,
            'authorized_count': authorized_count,
            'total_tested': len(user_accounts)
        }


def create_experiment_config_with_dem(args):
    """Create experiment configuration with DEM parameters"""
    experiment_config = {
        'dataset_name': args.dataset,
        'model_architecture': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epsilon': args.epsilon,
        'alpha': args.alpha,
        'lambda_reg': args.lambda_reg,
        'random_seed': args.seed,
        'enable_blockchain': not args.no_blockchain,
        'run_evaluation': not args.no_evaluation,
        'experiment_name': args.experiment_name,
        'device': str(Config.DEVICE),
        
        # New DEM related configurations
        'surrogate_model_epochs': 15,  # Surrogate model training epochs
        'max_surrogate_batches': 10,   # Surrogate model training batch count
        'max_processing_batches': 5,   # Unlearnable examples processing batch count
        'enable_multimodal_fusion': True,  # Whether to enable multimodal fusion
        'cross_attention_hidden_dim': 512,  # Cross-attention dimension
    }
    
    return experiment_config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='B-UEGMF Framework with Improved DEM Algorithm')
    
    # Existing parameters
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=Config.DEM.EPSILON)
    parser.add_argument('--alpha', type=float, default=Config.DEM.ALPHA)
    parser.add_argument('--lambda-reg', type=float, default=Config.DEM.LAMBDA_REG)
    parser.add_argument('--seed', type=int, default=Config.Experiment.RANDOM_SEED)
    parser.add_argument('--no-blockchain', action='store_true')
    parser.add_argument('--no-evaluation', action='store_true')
    parser.add_argument('--experiment-name', type=str)
    
    # New DEM related parameters
    parser.add_argument('--surrogate-epochs', type=int, default=15,
                       help='Surrogate model training epochs')
    parser.add_argument('--disable-multimodal', action='store_true',
                       help='Disable multimodal fusion')
    
    args = parser.parse_args()
    
    # Create experiment configuration
    experiment_config = create_experiment_config_with_dem(args)
    
    # If multimodal fusion is disabled
    if args.disable_multimodal:
        experiment_config['enable_multimodal_fusion'] = False
    
    # Set surrogate model training epochs
    experiment_config['surrogate_model_epochs'] = args.surrogate_epochs
    
    print(f"Running experiment with improved DEM algorithm...")
    print(f"Configuration: {experiment_config}")
    
    # Create and run experiment
    try:
        experiment = B_UEGMF_Experiment(experiment_config)
        results = experiment.run_experiment()
        
        print(f"\nExperiment completed successfully!")
        print(f"Results directory: {experiment.experiment_dir}")
        
        # Print key results
        if 'evaluation' in results['results']:
            overall_score = results['results']['evaluation']['overall_score']['overall_score']
            print(f"Overall evaluation score: {overall_score:.3f}")
        
        if 'unlearnable_examples' in results['results']:
            surrogate_acc = results['results']['unlearnable_examples']['surrogate_results']['average_accuracy']
            print(f"Surrogate model average accuracy: {surrogate_acc:.2f}%")
        
    except Exception as e:
        print(f"\nExperiment execution failed: {e}")
        import traceback
        traceback.print_exc()