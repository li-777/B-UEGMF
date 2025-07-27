# B-UEGMF Framework

**Blockchain-based Unlearnable Examples Generation with Multimodal Fusion**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

B-UEGMF is a privacy-preserving machine learning framework that combines blockchain-based access control with advanced unlearnable example generation. It uses a Dynamic Error Minimizing (DEM) algorithm enhanced with multimodal fusion to create training data that prevents unauthorized model training while maintaining data utility.

## Features

- 🔒 **Blockchain Access Control**: Smart contract-based dataset permission management
- 🧠 **Enhanced DEM Algorithm**: Advanced perturbation generation with surrogate model ensemble
- 🔄 **Multimodal Fusion**: Text and image perturbation combination for robust protection
- 📊 **Comprehensive Evaluation**: Multi-dimensional privacy protection assessment
- 🎛️ **Flexible Configuration**: Support for CIFAR-10/100, ImageNet datasets and multiple architectures

## Installation

```bash
git clone https://github.com/your-username/b-uegmf-framework.git
cd b-uegmf-framework
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Run experiment with default settings
python main.py --dataset cifar10 --model resnet18 --epochs 10

# Custom configuration
python main.py --dataset cifar100 --model vgg16 --epsilon 0.1 --alpha 0.3
```

### Python API

```python
from main import B_UEGMF_Experiment

config = {
    'dataset_name': 'cifar10',
    'model_architecture': 'resnet18',
    'epochs': 10,
    'epsilon': 16/255,
    'alpha': 0.3
}

experiment = B_UEGMF_Experiment(config)
results = experiment.run_experiment()
```

## Project Structure

```
b-uegmf-framework/
├── main.py                    # Main experiment runner
├── config.py                  # Configuration settings
├── dem_algorithm.py           # DEM algorithm implementation
├── blockchain_simulation.py   # Blockchain simulation
├── smart_contracts.py        # Smart contract logic
├── models.py                  # Neural network models
├── data_loader.py            # Dataset handling
├── trainer.py                # Training pipeline
├── evaluator.py              # Evaluation metrics
└── utils.py                  # Utility functions
```

## Configuration

Key parameters in `config.py`:

```python
# DEM Algorithm
EPSILON = 16/255      # Perturbation budget
ALPHA = 0.3          # Multimodal fusion weight
NUM_SURROGATE_MODELS = 3  # Ensemble size

# Training
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
```

## Command Line Options

```bash
--dataset {cifar10,cifar100,imagenet}  # Dataset selection
--model {resnet18,vgg16,densenet121}   # Model architecture
--epochs INT                           # Training epochs
--epsilon FLOAT                        # Perturbation budget
--alpha FLOAT                         # Fusion weight
--no-blockchain                       # Skip blockchain simulation
--experiment-name STR                 # Custom experiment name
```

## Results

The framework generates:
- Unlearnable examples with privacy protection
- Comprehensive evaluation metrics
- Blockchain access control logs
- Training performance reports

Results are saved in the `output/` directory with experiment timestamps.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{buegmf2024,
  title={B-UEGMF: Blockchain-based Unlearnable Examples Generation with Multimodal Fusion},
  author={Your Name},
  year={2024}
}
```
