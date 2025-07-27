import subprocess
import sys
import time

def run_speed_test():
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "speed_test.py"
        ], capture_output=False, text=True, timeout=300)
        
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n{total_time:.1f}s")
            return True
        else:
            print(f"\n {total_time:.1f}s")
            return False
            
    except subprocess.TimeoutExpired:
        print("\ntest timeout, there may be a performance issue")
        return False
    except Exception as e:
        print(f"\n error: {e}")
        return False

def run_optimized_experiment():
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", "cifar10",
        "--model", "resnet18",
        "--epochs", "2",
        "--batch-size", "16",
        "--epsilon", "0.1",
        "--surrogate-epochs", "3",
        "--experiment-name", "ultra_fast_test",
        "--no-blockchain",
        "--no-evaluation"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=900)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False

def main():
   
    speed_success = run_speed_test()
    if speed_success:
        exp_success = run_optimized_experiment()
    

if __name__ == "__main__":
    main()
