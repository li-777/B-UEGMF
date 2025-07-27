import subprocess
import sys
import os

def run_optimized_experiment():
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", "cifar10",
        "--model", "resnet18", 
        "--epochs", "3",  
        "--batch-size", "32",  
        "--epsilon", "0.0627",
        "--surrogate-epochs", "5",  
        "--experiment-name", "dem_quick_test",
        "--no-blockchain" 
    ]
    
    print(f" {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"{e}")
        return False

if __name__ == "__main__":
    success = run_optimized_experiment()
    sys.exit(0 if success else 1)
