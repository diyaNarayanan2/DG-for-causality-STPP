import numpy as np 
from SyntheticEnvs import SyntheticEnvs
from EMAlgo import *
from poisson import *
import time
import argparse
import seaborn as sns
import pandas as pd
from scipy.stats import weibull_min
import pickle
from datetime import datetime
import os
import json
from collections import defaultdict
import random
import torch
import matplotlib.pyplot as plt

# Create directories for model checkpoints
checkpoint_dir = "./model_checkpoints"
converged_dir = "./converged_model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(converged_dir, exist_ok=True)

#SET SEEDS FOR REPRODUCIBILITY
seed = 138
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

#input parameters 
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=1)
parser.add_argument('--lr', default=0.1)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--method", type=str, default="IRM")
parser.add_argument("--covariate_dim", type=int, default=10)
parser.add_argument("--true_mu", type=float, default=0.8)
parser.add_argument("--emiter", type=int, default=100)
parser.add_argument("--kernel_type", type=str, default="gaussian")
parser.add_argument("--start_alpha", type=float, default=6.0)
parser.add_argument("--start_beta", type=float, default=4.0)
parser.add_argument("--start_mu", type=float, default=0.6)
args = parser.parse_args()

# Replace the synthetic data generation with data loading
data_dir = "./synthetic_data"
try:
    with open(os.path.join(data_dir, 'synthetic_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    # Load data and parameters
    case_count_all = data['case_count_all']
    covariates_all = data['covariates_all']
    true_weights = data['true_weights']
    true_lambdas = data['true_lambdas']
    
    # Load true parameters
    args.true_alpha = data['true_params']['true_alpha']
    args.true_beta = data['true_params']['true_beta']
    args.n_day = data['true_params']['n_day']
    args.n_cty = data['true_params']['n_cty']
    args.env_list = data['true_params']['env_list']
    mu = data['true_params']['true_mu']
    
    print("Synthetic data and parameters loaded successfully")
except FileNotFoundError:
    raise FileNotFoundError(
        "Synthetic data not found. Please run generate_synthetic_data.py first"
    )

# EM initialization 
n_cty = args.n_cty
n_day = args.n_day
emitr = args.emiter
verbose = args.verbose
n_env = len(args.env_list)
break_diff = 1e-2
    
#initial guesses for Hawkes process parameters
env_data = [
    {
        "R0": np.exp(covariates_all[e] @ np.random.rand(args.covariate_dim)).reshape(-1, 1),
        "mus": np.full((n_cty, 1), args.start_mu),
        "lambda_t": np.zeros((n_cty, n_day)),
        "p_c_ij": None,
        "p_c_ii": None,
        "Q": None,
        "alpha": args.start_alpha,
        "beta": args.start_beta
    } for e in range(n_env)
]

# tracking updates across iterations
mus_prev, theta_prev = [None] * n_env, [None] * n_env
alphas_prev = [None] * n_env
betas_prev = [None] * n_env
alpha_delta = [[] for _ in range(n_env)]
beta_delta = [[] for _ in range(n_env)]
mus_delta = [[] for _ in range(n_env)]
theta_delta = [[] for _ in range(n_env)]

# Combine all environments for standardization
covariates_full = np.vstack([covariates_all[e] for e in range(n_env)])
cov_mean = np.mean(covariates_full, axis=0)
cov_std = np.std(covariates_full, axis=0)

# Add small constant to avoid division by zero
cov_std = np.where(cov_std < 1e-10, 1e-10, cov_std)

# Standardize each environment using the global parameters
for e in range(n_env): 
    covariates_all[e] = (covariates_all[e] - cov_mean) / cov_std

for itr in range(emitr):
    start_time = time.time()
    print("----------------------------------------------------------")
    print(f"Starting EM Iteration: {itr+1}")
    
    for e in range(n_env): 
        # E-step
        T = (case_count_all[e].shape[1])
        R0_ext_j = np.repeat(env_data[e]["R0"], T, axis=0) # shape: n_cty*n_day, 1
        trig_comp = R0_ext_j * wblval(T, n_cty, env_data[e]["alpha"], 0, env_data[e]["beta"]) * (np.repeat(case_count_all[e], T, axis=0) > 0)
        mu_comp = np.tile(np.eye(T), (n_cty, 1)) * np.repeat(env_data[e]["mus"], T, axis=0)
        lambda_t = np.sum(mu_comp + trig_comp, axis=1, keepdims=True)
        env_data[e]["lambda_t"] = lambda_t
        env_data[e]["p_c_ij"] = np.divide(trig_comp, lambda_t, where= lambda_t != 0)
        env_data[e]["p_c_ii"] = np.divide(mu_comp, lambda_t, where= lambda_t != 0)
        
        P_c_j = env_data[e]["p_c_ij"].reshape(n_day, n_day*n_cty)
        P_c_j = np.reshape(np.sum(P_c_j, axis=0), (n_cty, n_day))
        Q = P_c_j #* case_count_all[e]
        env_data[e]["Q"] = Q
    
        print(f"for environment {e}: ")
        print(f"Background rate spans: {np.min(env_data[e]['mus']):.4f} to {np.max(env_data[e]['mus']):.4f}")
        print(f"Conditional intensity: {np.min(env_data[e]['lambda_t']):.4f} to {np.max(env_data[e]['lambda_t']):.4f}")
        print(f"Q- Average children: {np.min(env_data[e]['Q']):.4f} to {np.max(env_data[e]['Q']):.4f}")
        print(f"P_c(i,j) range: {np.min(env_data[e]['p_c_ij']):.4f} to {np.max(env_data[e]['p_c_ij']):.4f}")
        print(f"P_c(i,i) range: {np.min(env_data[e]['p_c_ii']):.4f} to {np.max(env_data[e]['p_c_ii']):.4f}")
                
    endog = [env["Q"].reshape(-1,1) for env in env_data]
    exog = [np.repeat(covariates_all[e], T, axis=0) for e in range(n_env)]
    event_freqs = [case_count_all[e].reshape(-1,1) for e in range(n_env)]
    # not using frequencies for now 
    if args.verbose: 
        print(f"Starting poisson model learning using Domain Generalization: {args.method} ")
    if args.method == "IRM":
        model = InvariantRiskMinimization(exog, endog, event_freqs, args)
 
    elif args.method == "ERM": 
        model = EmpiricalRiskMinimizer(exog, endog, event_freqs, args)
        
    elif args.method == "MMD":
        model = MaximumMeanDiscrepancy(exog, endog, event_freqs, args)    
        
    coef = model.solution()
    print(f"Estimated coeffecients across all environments in EM itr: {itr}:")
    print(coef)  
    
    for e in range(n_env):
        # Update R0 with predictions for M-step
        env_data[e]["R0"] = model.predict(covariates_all[e])
        env_data[e]["R0"] = np.reshape(env_data[e]["R0"], (-1,1))
        
        # M-step
        mus = np.sum(env_data[e]["p_c_ii"], axis=1, keepdims=True)
        mus = mus.reshape(n_cty, n_day)
        mus = np.sum(mus, axis=1, keepdims=True) / n_day
        mus = np.clip(mus, 0, 1.5)
        env_data[e]["mus"] = mus
        
        obs = np.tril(np.arange(1, n_day+1)[:, None] - np.arange(1, n_day+1))
        weights = np.sum(env_data[e]["p_c_ij"].reshape(n_cty, n_day, n_day), axis=0)
        obs_1d = np.arange(1, n_day + 1)
        weights_1d = np.array([np.sum(weights[np.where(obs == time_diff)]) for time_diff in obs_1d])
        normalized_weights = np.divide(weights_1d, np.sum(weights_1d), where=np.sum(weights_1d) != 0)
        sample_size = 1000  # Choose an appropriate sample size
        sampled_times = np.random.choice(obs_1d, size=sample_size, p=normalized_weights)
        shape, loc, scale = weibull_min.fit(sampled_times, floc=0)
        env_data[e]["beta"] = np.clip(scale, 0.5, 20)
        env_data[e]["alpha"] = np.clip(shape, 0.5, 25)    
        
        print(f"Environment {e+1}: Mus from {np.min(env_data[e]['mus']):.4f} to {np.max(env_data[e]['mus']):.4f}")
        print(f"Weibull scale: {env_data[e]['alpha']:.4f}, Shape: {env_data[e]['beta']:.4f}")
        
        if itr == 0:
            # Save the first iteration values
            mus_prev[e] = np.mean(env_data[e]["mus"], axis=0)
            theta_prev[e] = coef.flatten()
            alphas_prev[e] = env_data[e]["alpha"]
            betas_prev[e] = env_data[e]["beta"]
        else:
            # Calculate RMSR for convergence check
            mus_delta[e].append(np.sqrt(np.mean((mus_prev[e] - env_data[e]["mus"]) ** 2)))
            theta_delta[e].append(np.sqrt(np.mean((theta_prev[e] - coef) ** 2)))
            alpha_delta[e].append(np.sqrt((env_data[e]["alpha"] - alphas_prev[e]) ** 2))
            beta_delta[e].append(np.sqrt((env_data[e]["beta"] - betas_prev[e]) ** 2))
            
            # Save current values for next iteration
            mus_prev[e] = env_data[e]["mus"]
            theta_prev[e] = coef
            alphas_prev[e] = env_data[e]["alpha"]
            betas_prev[e] = env_data[e]["beta"]

    # Early stopping criteria
    if itr > 5:
        print("Commencing convergence check: ")
        converged = True
        for e in range(n_env):
            env_converged = (
                np.all(np.array(mus_delta[e][-5:]) < break_diff) and
                np.all(np.array(theta_delta[e][-5:]) < break_diff) and
                np.all(np.array(alpha_delta[e][-5:]) < break_diff) and
                np.all(np.array(beta_delta[e][-5:]) < break_diff)
            )
            if not env_converged:
                converged = False
                break
        
        if converged:
            print(f"Convergence criterion met at iteration {itr + 1}. Exiting EM loop.")
            
            # Save all plots to converged directory
            # Convergence plot
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(1, len(alpha_delta[0])+1), np.mean(alpha_delta, axis=0), label="Alpha Deltas", color="red")
            plt.plot(np.arange(1, len(beta_delta[0])+1), np.mean(beta_delta, axis=0), label="Beta Deltas", color="blue")
            plt.plot(np.arange(1, len(mus_delta[0])+1), np.mean(mus_delta, axis=0), label="Mus Deltas", color="green")
            plt.plot(np.arange(1, len(theta_delta[0])+1), np.mean(theta_delta, axis=0), label="0 Deltas", color="orange")
            plt.legend()
            plt.savefig(os.path.join(converged_dir, "convergence_plot.png"))
            plt.close()

            # Final comparison plots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            plt.tight_layout()
            plt.savefig(os.path.join(converged_dir, "final_comparison_plots.png"))
            plt.close()

            # Save model parameters
            avg_alpha = np.mean([env["alpha"] for env in env_data])
            avg_beta = np.mean([env["beta"] for env in env_data])
            avg_mus = np.mean([env["mus"] for env in env_data], axis=0)
            scaled_coef = coef.flatten() * cov_std
            
            model_params = {
                'method': args.method,
                'true_params': {
                    'alpha': args.true_alpha,
                    'beta': args.true_beta,
                    'mu': mu.flatten().tolist(),
                    'weights': true_weights.flatten().tolist()
                },
                'estimated_params': {
                    'alpha_per_env': [env["alpha"] for env in env_data],
                    'beta_per_env': [env["beta"] for env in env_data],
                    'mu_per_env': [env["mus"].flatten().tolist() for env in env_data],
                    'avg_alpha': avg_alpha,
                    'avg_beta': avg_beta,
                    'avg_mus': avg_mus.flatten().tolist(),
                    'scaled_coefficients': scaled_coef.tolist()
                },
                'convergence_metrics': {
                    'final_iteration': itr + 1,
                    'alpha_delta': np.mean(alpha_delta, axis=0).tolist(),
                    'beta_delta': np.mean(beta_delta, axis=0).tolist(),
                    'mus_delta': np.mean(mus_delta, axis=0).tolist(),
                    'theta_delta': np.mean(theta_delta, axis=0).tolist()
                },
                'mse_metrics': {
                    'alpha_mse': float((args.true_alpha - avg_alpha)**2),
                    'beta_mse': float((args.true_beta - avg_beta)**2),
                    'mus_mse': float(np.mean((mu.flatten() - avg_mus.flatten())**2)),
                    'weights_mse': float(np.mean((true_weights - scaled_coef)**2))
                }
            }

            # Save parameters to JSON file
            with open(os.path.join(converged_dir, 'model_parameters.json'), 'w') as f:
                json.dump(model_params, f, indent=4)

            break

    elapsed_time = time.time() - start_time
    print(f"Iteration {itr+1} completed in {elapsed_time:.2f} seconds.")
    print("----------------------------------------------------------")


print("EM Algorithm Completed.")
print("_______________________________________________________________________________________________________________________")

# Save final model parameters to checkpoint directory
avg_alpha = np.mean([env["alpha"] for env in env_data])
avg_beta = np.mean([env["beta"] for env in env_data])
avg_mus = np.mean([env["mus"] for env in env_data], axis=0)
scaled_coef = coef.flatten() * cov_std

checkpoint_params = {
    'method': args.method,
    'iterations_completed': itr + 1,
    'true_params': {
        'alpha': args.true_alpha,
        'beta': args.true_beta,
        'mu': mu.flatten().tolist(),
        'weights': true_weights.flatten().tolist()
    },
    'estimated_params': {
        'alpha_per_env': [float(env["alpha"]) for env in env_data],
        'beta_per_env': [float(env["beta"]) for env in env_data],
        'mu_per_env': [env["mus"].flatten().tolist() for env in env_data],
        'avg_alpha': float(avg_alpha),
        'avg_beta': float(avg_beta),
        'avg_mus': avg_mus.flatten().tolist(),
        'scaled_coefficients': scaled_coef.tolist()
    },
    'training_metrics': {
        'alpha_delta_history': np.mean(alpha_delta, axis=0).tolist() if alpha_delta else [],
        'beta_delta_history': np.mean(beta_delta, axis=0).tolist() if beta_delta else [],
        'mus_delta_history': np.mean(mus_delta, axis=0).tolist() if mus_delta else [],
        'theta_delta_history': np.mean(theta_delta, axis=0).tolist() if theta_delta else []
    },
    'final_mse': {
        'alpha_mse': float((args.true_alpha - avg_alpha)**2),
        'beta_mse': float((args.true_beta - avg_beta)**2),
        'mus_mse': float(np.mean((mu.flatten() - avg_mus.flatten())**2)),
        'weights_mse': float(np.mean((true_weights - scaled_coef)**2))
    },
    'hyperparameters': {
        'learning_rate': float(args.lr),
        'n_iterations': args.n_iterations,
        'n_day': args.n_day,
        'n_cty': args.n_cty,
        'covariate_dim': args.covariate_dim,
        'em_iterations': args.emiter,
        'kernel_type': args.kernel_type
    }
}

# Save parameters to JSON file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
params_filename = f'model_parameters_{timestamp}.json'
with open(os.path.join(checkpoint_dir, params_filename), 'w') as f:
    json.dump(checkpoint_params, f, indent=4)

print(f"EM Convergence: after {itr+1} iterations, method: {args.method}")
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(alpha_delta[0])+1), np.mean(alpha_delta, axis=0), label="Alpha Deltas", color="red")
plt.plot(np.arange(1, len(beta_delta[0])+1), np.mean(beta_delta, axis=0), label="Beta Deltas", color="blue")
plt.plot(np.arange(1, len(mus_delta[0])+1), np.mean(mus_delta, axis=0), label="Mus Deltas", color="green")
plt.plot(np.arange(1, len(theta_delta[0])+1), np.mean(theta_delta, axis=0), label="0 Deltas", color="orange")
plt.legend()
plt.savefig(os.path.join(checkpoint_dir, "final_convergence_plot.png"))
plt.close()

# Calculate averages across environments
avg_alpha = np.mean([env["alpha"] for env in env_data])
avg_beta = np.mean([env["beta"] for env in env_data])
avg_mus = np.mean([env["mus"] for env in env_data], axis=0)

# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Alpha and Beta comparison
params = ['Alpha', 'Beta']
true_vals = [args.true_alpha, args.true_beta]
est_vals = [avg_alpha, avg_beta]
x = np.arange(len(params))
width = 0.35

ax1.bar(x - width/2, true_vals, width, label='True Values', color='skyblue')
ax1.bar(x + width/2, est_vals, width, label='Estimated Values', color='lightcoral')
ax1.set_ylabel('Value')
ax1.set_title('True vs Estimated Alpha and Beta')
ax1.set_xticks(x)
ax1.set_xticklabels(params)
ax1.legend()

# Plot 2: True mu vs Estimated mu comparison across counties
x = np.arange(len(mu))
ax2.bar(x - width/2, mu.flatten(), width, label='True μ', color='skyblue')
ax2.bar(x + width/2, avg_mus.flatten(), width, label='Estimated μ', color='lightcoral')
ax2.set_xlabel('County Index')
ax2.set_ylabel('Background Rate (μ)')
ax2.set_title('True vs Estimated Background Rates')
ax2.set_xticks(x)
ax2.legend()

# Plot 3: True weights vs Estimated coefficients across dimensions
x = np.arange(len(true_weights))
scaled_coef = coef.flatten() * cov_std
ax3.bar(x - width/2, true_weights.flatten(), width, label='True Weights', alpha=0.7)
ax3.bar(x + width/2, scaled_coef, width, label='Estimated Coef', alpha=0.7)
ax3.set_xlabel('Coefficient Index')
ax3.set_ylabel('Value')
ax3.set_title('True vs Estimated Weights')
ax3.legend()

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "final_comparison_plots.png"))
plt.close()

# Print MSE values
print(f"MSE for Alpha: {(args.true_alpha - avg_alpha)**2:.4f}")
print(f"MSE for Beta: {(args.true_beta - avg_beta)**2:.4f}")
print(f"MSE for Mus: {np.mean((mu.flatten() - avg_mus.flatten())**2):.4f}")
print(f"MSE for Weights: {np.mean((true_weights - scaled_coef)**2):.4f}")

print("\nFinal Model Parameters:")
print("-----------------------")
print(f"True Parameters:")
print(f"- Alpha: {args.true_alpha:.4f}")
print(f"- Beta: {args.true_beta:.4f}")
print(f"- Mu: {mu.flatten()}")
print(f"- Weights: {true_weights.flatten()}")
print("\nEstimated Parameters:")
print(f"- Alpha (avg across envs): {avg_alpha:.4f}")
print(f"- Beta (avg across envs): {avg_beta:.4f}")
print(f"- Mu (avg across envs): {avg_mus.flatten()}")
print(f"- Weights (scaled): {scaled_coef}")

# Print per-environment parameters
print("\nPer-Environment Parameters:")
for e in range(len(env_data)):
    print(f"\nEnvironment {e+1}:")
    print(f"- Alpha: {env_data[e]['alpha']:.4f}")
    print(f"- Beta: {env_data[e]['beta']:.4f}")
    print(f"- Mu range: [{np.min(env_data[e]['mus']):.4f}, {np.max(env_data[e]['mus']):.4f}]")