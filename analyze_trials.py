import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_trials(checkpoint_dir='em_checkpoints'):
    summary_file = Path(checkpoint_dir) / 'trials_summary.json'
    
    if not summary_file.exists():
        print(f"No summary file found at {summary_file}")
        return None
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"\nAnalysis of {summary['aggregate_statistics']['total_trials']} trials:")
    print("\nAggregate MSE Statistics:")
    for metric, mean_value in summary['aggregate_statistics']['mean_mse'].items():
        std_value = summary['aggregate_statistics']['std_mse'][metric]
        print(f"{metric:15s}: mean = {mean_value:.4f}, std = {std_value:.4f}")
    
    # Create visualizations
    plot_mse_distributions(summary)
    plot_convergence_trends(summary)
    
    return summary

def plot_mse_distributions(summary):
    """Plot distribution of MSE values across trials"""
    mse_data = {metric: [] for metric in summary['trials'][0]['mse_metrics'].keys()}
    
    for trial in summary['trials']:
        for metric, value in trial['mse_metrics'].items():
            mse_data[metric].append(value)
    
    plt.figure(figsize=(10, 6))
    plt.title('MSE Distribution Across Trials')
    sns.boxplot(data=mse_data)
    plt.xticks(rotation=45)
    plt.ylabel('MSE Value')
    plt.tight_layout()
    plt.show()

def plot_convergence_trends(summary):
    """Plot final delta values across trials"""
    convergence_data = {
        'alpha': [],
        'beta': [],
        'mus': [],
        'R0': []
    }
    
    for trial in summary['trials']:
        for metric, value in trial['convergence_metrics']['final_deltas'].items():
            if value is not None:  # Handle cases where delta might be None
                convergence_data[metric].append(value)
    
    plt.figure(figsize=(10, 6))
    plt.title('Final Convergence Values Across Trials')
    sns.boxplot(data=convergence_data)
    plt.xticks(rotation=45)
    plt.ylabel('Final Delta Value')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    summary = analyze_trials() 