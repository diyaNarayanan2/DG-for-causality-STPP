import subprocess
import argparse
import time
from datetime import datetime
import os

def run_trials(n_trials=10):
    print(f"Starting {n_trials} trials of the EM algorithm")
    print("=" * 50)
    
    total_start_time = time.time()
    
    for i in range(n_trials):
        trial_start_time = time.time()
        print(f"\nStarting Trial {i+1}/{n_trials}")
        print("-" * 30)
        
        # Run the EM algorithm
        try:
            subprocess.run(['python', 'test.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in trial {i+1}: {e}")
            continue
            
        trial_duration = time.time() - trial_start_time
        print(f"Trial {i+1} completed in {trial_duration:.2f} seconds")
    
    total_duration = time.time() - total_start_time
    print("\n" + "=" * 50)
    print(f"All trials completed in {total_duration:.2f} seconds")
    print(f"Average time per trial: {total_duration/n_trials:.2f} seconds")

def clear_summary():
    checkpoint_dir = 'em_checkpoints'
    if os.path.exists(checkpoint_dir):
        response = input("This will delete ALL trial data in em_checkpoints directory. Are you sure? (y/n): ").lower()
        if response != 'y':
            print("Operation cancelled.")
            return
            
        for file in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Cleared all previous trial data.")

if __name__ == "__main__":
    # Command to clear and run: python run_trials.py --clear --n_trials 10
    # Command to just clear: python run_trials.py --clear --n_trials 0
    # Command to run without clearing: python run_trials.py --n_trials 10
    
    parser = argparse.ArgumentParser(description='Run multiple trials of the EM algorithm')
    parser.add_argument('--n_trials', type=int, default=10,
                      help='Number of trials to run (default: 10)')
    parser.add_argument('--clear', action='store_true',
                      help='Clear previous trials summary before running')
    
    args = parser.parse_args()
    
    if args.clear:
        clear_summary()
    
    if args.n_trials > 0:
        run_trials(args.n_trials) 