# B-UEGMF

**Blockchain-based Unlearnable Examples Generation with Multimodal Fusion**

## Overview

B-UEGMF is a privacy-preserving machine learning framework that combines blockchain-based access control with advanced unlearnable example generation. It uses a Dynamic Error Minimizing (DEM) algorithm enhanced with multimodal fusion to create training data that prevents unauthorized model training while maintaining data utility.

## Quick Start

### Basic Usage

```bash
# Basic experiment
python main.py --dataset <DATASET> --model <MODEL> --epochs <NUM>

# Full configuration
python main.py \
    --dataset <DATASET> \
    --model <MODEL> \
    --epochs <NUM> \
    --batch-size <SIZE> \
    --lr <RATE> \
    --epsilon <BUDGET> \
    --alpha <WEIGHT> \
    --experiment-name "<NAME>"

# Quick testing
python main.py --dataset <DATASET> --no-blockchain --no-evaluation
