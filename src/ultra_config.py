import torch

class UltraConfig:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ultra-optimized DEM parameters
    class DEM:
        EPSILON = 0.1  # Increase perturbation constraint
        ALPHA = 0.2
        LAMBDA_REG = 0.05  # Reduce regularization
        
        # Significantly reduce computation
        NUM_SURROGATE_MODELS = 2  # Reduced from 3 to 2
        SURROGATE_EPOCHS = 3      # Reduced from 5 to 3
        SURROGATE_BATCHES = 3     # Reduced from 10 to 3
        
        EM_ITERATIONS = 10        # Reduced from 20 to 10
        EM_LEARNING_RATE = 0.05   # Increased learning rate
        
        # Batch processing optimization
        MAX_PROCESSING_BATCHES = 2  # Reduced from 3 to 2
        BATCH_SIZE = 16             # Reduced from 32 to 16
        
        # Validation optimization
        VALIDATION_SAMPLES = 16     # Reduced from 20 to 16
        EFFECTIVENESS_THRESHOLD = 0.03  # Lowered threshold

    # Training optimization
    class Training:
        EPOCHS = 2              # Reduced from 3 to 2
        LEARNING_RATE = 0.002   # Increased learning rate
        BATCH_SIZE = 16         # Reduced batch size
        
    print("Ultra-optimized configuration loaded - Focused on speed and core functionality")

if __name__ == "__main__":
    print("UltraConfig - Built for speed")
    print(f"Device: {UltraConfig.DEVICE}")
    print(f"DEM Epsilon: {UltraConfig.DEM.EPSILON}")
    print(f"Number of surrogate models: {UltraConfig.DEM.NUM_SURROGATE_MODELS}")
    print(f"Batch size: {UltraConfig.DEM.BATCH_SIZE}")