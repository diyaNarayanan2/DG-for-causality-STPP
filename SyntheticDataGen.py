'''
This fike contains a code that runs the Synthetic Env class, stores the true parameters of the synthetic data generated.
It also stores visualizations to describe the synehtic data (Conditional Intensity, daily events, cummulative events- across different environments)'''

import numpy as np
from SyntheticEnvs import SyntheticEnvs
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#first run this file, then Main.py
# each time this file is run, the old data gets overwritten

def generate_and_save_data(args):
    # Create data directory if it doesn't exist
    data_dir = "./synthetic_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Generate synthetic data
    mu = np.repeat(args.true_mu, args.n_cty)
    True_Data = SyntheticEnvs(
        args.n_cty, 
        args.n_day, 
        args.true_alpha, 
        args.true_beta, 
        mu, 
        args.env_list, 
        args.covariate_dim, 
        args.seed
    )
    
    # Save all data and parameters in one file
    data = {
        'case_count_all': True_Data.case_count_envs,
        'covariates_all': True_Data.covariate_envs,
        'true_weights': True_Data.trueSolution(),
        'true_lambdas': True_Data.lambda_envs,
        'true_params': {
            'true_alpha': args.true_alpha,
            'true_beta': args.true_beta,
            'true_mu': mu,
            'seed': args.seed,
            'env_list': args.env_list,
            'n_cty': args.n_cty,
            'n_day': args.n_day
        }
    }
    
    with open(os.path.join(data_dir, 'synthetic_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    # Create and save visualizations
    # Plot 1: Lambda heatmaps
    n_envs = len(data['true_lambdas'])
    n_cols = min(3, n_envs)
    n_rows = (n_envs + 2) // 3
    
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_envs > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, lambda_env in enumerate(data['true_lambdas']):
        sns.heatmap(lambda_env, ax=axes[i], cmap='YlOrRd')
        axes[i].set_title(f'Environment {i+1}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Location')
    
    # Add parameter values as text at the bottom of the figure
    param_text = f'μ = {args.true_mu:.2f}, α = {args.true_alpha:.2f}, β = {args.true_beta:.2f}'
    fig1.text(0.5, 0.02, param_text, ha='center', fontsize=10)
    
    # Adjust layout to make room for the parameter text
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    for j in range(i + 1, len(axes)):
        axes[j].remove()
    
    plt.savefig(os.path.join(data_dir, 'lambda_heatmaps.png'))
    plt.close()
    
    # Plot 2: Cumulative case counts
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_envs))
    for i, cases in enumerate(data['case_count_all']):
        cumulative_cases = np.cumsum(cases.sum(axis=0))
        ax.plot(cumulative_cases, color=colors[i], 
                label=f'Environment {i+1}', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Cases')
    ax.set_title('Cumulative Case Counts Across Environments')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(data_dir, 'cumulative_cases.png'))
    plt.close()
    
    # Plot 3: Daily new cases
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    for i, cases in enumerate(data['case_count_all']):
        daily_cases = cases.sum(axis=0)  # Sum across all locations for each day
        ax.plot(daily_cases, color=colors[i], 
                label=f'Environment {i+1}', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Daily New Cases')
    ax.set_title('Daily New Cases Across Environments')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(data_dir, 'daily_cases.png'))
    plt.close()
    
    print("Synthetic data, parameters, and visualizations saved successfully")
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_day", type=int, default=20)
    parser.add_argument("--true_alpha", type=float, default=1)
    parser.add_argument("--true_beta", type=float, default=1.81)
    parser.add_argument("--env_list", type=list, default=[1.2, 1.8, 3, 0.2, 5])
    parser.add_argument("--covariate_dim", type=int, default=10)
    parser.add_argument("--true_mu", type=float, default=0.8)
    parser.add_argument("--n_cty", type=int, default=20)
    parser.add_argument("--seed", type=int, default=138)
    
    args = parser.parse_args()
    generate_and_save_data(args) 