"""
Temporary implementation of EM Algo 
for synthetically generated causal and acausal 
covariate and case count data 
"""

import numpy as np 
from SyntheticEnvs import SyntheticEnvs
from EMAlgo import ExpStep, MaxStep
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

COVARIATES = ["Grocery_pharmacy", "Parks", "Residential", "Retail_recreation", "Transit_stations", "Workplace", "Population Density", 
              "Population_est", "#ICUbeds", "Median_Age", "Smokers%", "Diabetes%", "HeartDiseaseMosrtality", "#Hospitals"]

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument("--n_iterations", type=int, default=5000)
parser.add_argument("--n_cty", type=int, default=20)
parser.add_argument("--n_day", type=int, default=100)
parser.add_argument("--n_demo_cov", type=int, default=8)
parser.add_argument("--n_mob_cov", type=int, default=6)
parser.add_argument("--true_alpha", type=float, default=2.5)
parser.add_argument("--true_beta", type=float, default=3)
parser.add_argument("--loc", type=float, default=0)
parser.add_argument("--emitr", type=int, default=30)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=2.5)
parser.add_argument("--env_list", type=float, nargs='+', default=[1, 2.3, 3.2, 0.2, 1.2])
parser.add_argument("--covariate_dim", type=int, default=8)
parser.add_argument("--method", type=str, default="IRM")

# Parse arguments
args = parser.parse_args([])

# Use parsed arguments
n_cty = args.n_cty
T = n_day = args.n_day
n_demo_cov = args.n_demo_cov
n_mob_cov = args.n_mob_cov
true_alpha = args.true_alpha
true_beta = args.true_beta
loc = args.loc
emitr = args.emitr
alpha = args.alpha
beta = args.beta
env_list = args.env_list
covariate_dim = args.covariate_dim

break_diff = 0.01
n_env = len(env_list)
true_mu = np.concatenate((0.4 * np.ones((n_cty//2, 1)), 0.6 * np.ones((n_cty//2, 1))))
syn = SyntheticEnvs(n_cty, n_day, n_demo_cov, n_mob_cov, alpha, beta, true_mu, env_list, covariate_dim)
case_count_all = syn.case_count_envs
covariates_all = syn.covariate_envs
true_weights = syn.trueSolution()
# access diff env data as covariates_all[e]
#normalise covariate data 
for e in range(n_env): 
    cov_mean = np.mean(covariates_all[e], axis=0)
    cov_std = np.std(covariates_all[e], axis=0)
    
    # Add small constant to avoid division by zero
    cov_std = np.where(cov_std < 1e-10, 1e-10, cov_std)
    covariates_all[e] = (covariates_all[e] - cov_mean) / cov_std


#initialise dictionary to store variables across diff envs 
env_data = [
    {
        "R0": np.ones((n_cty, n_day)) * (0.8 + 0.4 * np.random.random()),
        "mus": np.random.uniform(0.3, 0.7, (n_cty, 1)),
        "lam": np.zeros((n_cty, n_day)),
        "pc_ij": None,
        "pc_ii": None,
        "Q": None,
        "alpha": 1.5 + np.random.random(),
        "beta": 1.5 + np.random.random()
    } for _ in range(n_env)
]

# tracking updates across iterations
mus_prev = [None] * n_env
R0_prev = [None] * n_env
alphas_prev, betas_prev = [2] * n_env, [2] * n_env
alpha_delta = [[] for _ in range(n_env)]
beta_delta = [[] for _ in range(n_env)]
mus_delta = [[] for _ in range(n_env)]
R0_delta = [[] for _ in range(n_env)]

# Create checkpoint directory
checkpoint_dir = "em_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a unique trial ID using timestamp
trial_id = datetime.now().strftime("%Y%m%d_%H%M%S")
trial_dir = os.path.join(checkpoint_dir, trial_id)
os.makedirs(trial_dir, exist_ok=True)

# Save initial parameters
initial_params = {
    'true_params': {
        'mu': true_mu,
        'alpha': true_alpha,
        'beta': true_beta,
        'weights': true_weights
    },
    'initial_guesses': {
        'mus': [env["mus"] for env in env_data],
        'alphas': [env["alpha"] for env in env_data],
        'betas': [env["beta"] for env in env_data]
    }
}

checkpoint_path = os.path.join(trial_dir, 'initial_params.pkl')
with open(checkpoint_path, 'wb') as f:
    pickle.dump(initial_params, f)

for itr in range(emitr):
    start_time = time.time()
    print("----------------------------------------------------------")
    print(f"Starting EM Iteration: {itr+1}")
    
    for e in range(n_env): 
        # E-step
        env_data[e]["lam"], env_data[e]["pc_ij"], env_data[e]["pc_ii"], env_data[e]["Q"] = ExpStep(
            case_count_all[e], env_data[e]["mus"], env_data[e]["R0"], env_data[e]["alpha"], loc,  env_data[e]["beta"], args.verbose
        )
        if args.verbose: 
            plt.figure(figsize=(10,6))
            sns.heatmap(env_data[e]["lam"])
            plt.title(f'Conditional intensity at itr:{itr} for env{e+1}')
            plt.xlabel('Days')
            plt.ylabel('Counties')
            plt.savefig(os.path.join(trial_dir, f'conditional_intensity_itr_{itr}_env_{e+1}.png'))
            plt.close()
    
        print(f"for environment {e}: ")
        print(f"Background rate spans: {np.min(env_data[e]['mus']):.4f} to {np.max(env_data[e]['mus']):.4f} with avg: {np.mean(env_data[e]['mus']):.4f}")
        print(f"Conditional intensity: {np.min(env_data[e]['lam']):.4f} to {np.max(env_data[e]['lam']):.4f} with avg: {np.mean(env_data[e]['lam']):.4f}")
        print(f"Q- Average children: {np.min(env_data[e]['Q']):.4f} to {np.max(env_data[e]['Q']):.4f} with avg: {np.mean(env_data[e]['Q']):.4f}")
        print(f"P_c(i,j) range: {np.min(env_data[e]['pc_ij']):.4f} to {np.max(env_data[e]['pc_ij']):.4f}")
        print(f"P_c(i,i) range: {np.min(env_data[e]['pc_ii']):.4f} to {np.max(env_data[e]['pc_ii']):.4f}")
                
    all_endog = [env["Q"].reshape(-1,1) for env in env_data]
    all_event_freqs = case_count_all
    # not using frequencies for now 
    if args.verbose: 
        print(f"Starting poisson model learning using Domain Generalization: ")
    if args.method == "IRM":
        model = InvariantRiskMinimization(covariates_all, all_endog, all_event_freqs, args)
 
    elif args.method == "ERM": 
        model = EmpiricalRiskMinimizer(covariates_all, all_endog, all_event_freqs, args)
        
    elif args.method == "MMD":
        model = MaximumMeanDiscrepancy(covariates_all, all_endog, all_event_freqs, args)    
        
    coef = model.solution()
    print(f"Estimated coeffecients across all environments in EM itr: {itr}:")
    print(coef)  
    
    for e in range(n_env):
        # Update R0 with predictions for M-step
        env_data[e]["R0"] = model.predict(covariates_all[e])
        env_data[e]["R0"] = np.reshape(env_data[e]["R0"], (n_cty, n_day))
        
        # M-step
        env_data[e]["mus"], env_data[e]["alpha"], _ ,  env_data[e]["beta"] = MaxStep(
            env_data[e]["R0"], env_data[e]["pc_ii"], env_data[e]["pc_ij"], case_count_all[e], args.verbose
        )
        print(f"Environment {e+1}: Mus from {np.min(env_data[e]['mus']):.4f} to {np.max(env_data[e]['mus']):.4f} with avg: {np.mean(env_data[e]['mus']):.4f}")
        print(f"Weibull scale: {env_data[e]['alpha']:.4f}, Shape: {env_data[e]['beta']:.4f}")
        
        if itr == 0:
            # Save the first iteration values
            mus_prev[e] = env_data[e]["mus"]
            R0_prev[e] = env_data[e]["R0"]
        else:
            # Calculate RMSR for convergence check
            mus_delta[e].append(np.sqrt(np.mean((mus_prev[e] - env_data[e]["mus"]) ** 2)))
            R0_delta[e].append(np.sqrt(np.mean((R0_prev[e] - env_data[e]["R0"]) ** 2)))
            alpha_delta[e].append(np.sqrt((env_data[e]["alpha"] - alphas_prev[e]) ** 2))
            beta_delta[e].append(np.sqrt((env_data[e]["beta"] - betas_prev[e]) ** 2))
            

            # Save current values for next iteration
            mus_prev[e] = env_data[e]["mus"]
            R0_prev[e] = env_data[e]["R0"]

    # Early stopping criteria
    if itr > 5:
        print("Commencing convergence check: ")
        converged = True
        for e in range(n_env):
            env_converged = (
                np.all(np.array(mus_delta[e][-5:]) < break_diff) and
                np.all(np.array(R0_delta[e][-5:]) < break_diff) and
                np.all(np.array(alpha_delta[e][-5:]) < break_diff) and
                np.all(np.array(beta_delta[e][-5:]) < break_diff)
            )
            if not env_converged:
                converged = False
                break
        
        if converged:
            print(f"Convergence criterion met at iteration {itr + 1}. Exiting EM loop.")
            break


    elapsed_time = time.time() - start_time
    print(f"Iteration {itr+1} completed in {elapsed_time:.2f} seconds.")
    print("----------------------------------------------------------")


print("EM Algorithm Completed.")
print("_______________________________________________________________________________________________________________________")

print("EM Convergence: ")
plt.plot(np.arange(1, len(alpha_delta[0])+1), np.mean(alpha_delta, axis=0), label="Alpha", color="red")
plt.plot(np.arange(1, len(beta_delta[0])+1), np.mean(beta_delta, axis=0), label="Beta", color="blue")
plt.plot(np.arange(1, len(mus_delta[0])+1), np.mean(mus_delta, axis=0), label="Mus", color="green")
plt.plot(np.arange(1, len(R0_delta[0])+1), np.mean(R0_delta, axis=0), label="R0", color="orange")
plt.legend()
plt.savefig(os.path.join(trial_dir, f'convergence_plot.png'))
plt.close()

# Calculate averages across environments
avg_alpha = np.mean([env["alpha"] for env in env_data])
avg_beta = np.mean([env["beta"] for env in env_data])
avg_mus = np.mean([env["mus"] for env in env_data], axis=0)

# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Alpha and Beta MSE
params = ['Alpha', 'Beta']
true_vals = [true_alpha, true_beta]
est_vals = [avg_alpha, avg_beta]
mse_vals = [(true - est)**2 for true, est in zip(true_vals, est_vals)]

ax1.bar(params, mse_vals)
ax1.set_title('MSE of Estimated vs True Parameters')
ax1.set_ylabel('Mean Squared Error')

# Plot 2: True mu vs Estimated mu comparison across counties
x = np.arange(n_cty)
width = 0.35
ax2.bar(x - width/2, true_mu.flatten(), width, label='True μ', alpha=0.7)
ax2.bar(x + width/2, avg_mus.flatten(), width, label='Estimated μ', alpha=0.7)
ax2.set_xlabel('County Index')
ax2.set_ylabel('Value')
ax2.set_title('True vs Estimated Background Rates (μ)')
ax2.legend()

# Alternative Plot 2 using lines:
# ax2.plot(x, true_mu.flatten(), 'b-', label='True μ', alpha=0.7)
# ax2.plot(x, avg_mus.flatten(), 'r--', label='Estimated μ', alpha=0.7)

# Plot 3: True weights vs Estimated coefficients across dimensions
x = np.arange(len(true_weights))
width = 0.35
ax3.bar(x - width/2, true_weights.flatten(), width, label='True Weights', alpha=0.7)
ax3.bar(x + width/2, coef.flatten(), width, label='Estimated Coef', alpha=0.7)
ax3.set_xlabel('Coefficient Index')
ax3.set_ylabel('Value')
ax3.set_title('True vs Estimated Weights')
ax3.legend()

# Alternative Plot 3 using lines:
# ax3.plot(x, true_weights, 'b-', label='True Weights', alpha=0.7)
# ax3.plot(x, coef, 'r--', label='Estimated Coef', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(trial_dir, f'results_comparison.png'))
plt.close()

# Print MSE values
print(f"MSE for Alpha: {(true_alpha - avg_alpha)**2:.4f}")
print(f"MSE for Beta: {(true_beta - avg_beta)**2:.4f}")
print(f"MSE for Mus: {np.mean((true_mu - avg_mus)**2):.4f}")
print(f"MSE for Weights: {np.mean((true_weights - coef)**2):.4f}")

# Add at the end of the script, after the MSE calculations
final_checkpoint = {
    'parameter_deltas': {
        'alpha_delta': alpha_delta,
        'beta_delta': beta_delta,
        'mus_delta': mus_delta,
        'R0_delta': R0_delta
    },
    'final_params': {
        'avg_alpha': avg_alpha,
        'avg_beta': avg_beta,
        'avg_mus': avg_mus,
        'final_weights': coef
    },
    'mse_metrics': {
        'alpha_mse': (true_alpha - avg_alpha)**2,
        'beta_mse': (true_beta - avg_beta)**2,
        'mus_mse': np.mean((true_mu - avg_mus)**2),
        'weights_mse': np.mean((true_weights - coef)**2)
    }
}

final_checkpoint_path = os.path.join(trial_dir, 'final_checkpoint.pkl')
with open(final_checkpoint_path, 'wb') as f:
    pickle.dump(final_checkpoint, f)

print(f"\nCheckpoints saved in {trial_dir}/")

# Create/update summary file
summary_file = os.path.join(checkpoint_dir, 'trials_summary.json')

# Prepare summary data for this trial
trial_summary = {
    'trial_id': trial_id,
    'mse_metrics': final_checkpoint['mse_metrics'],
    'final_params': {
        'avg_alpha': float(avg_alpha),  # Convert numpy types to native Python types
        'avg_beta': float(avg_beta),
        'avg_mus_mean': float(np.mean(avg_mus)),
        'weights_mean': float(np.mean(coef))
    },
    'convergence_metrics': {
        'final_iteration': itr + 1,
        'final_deltas': {
            'alpha': float(np.mean(alpha_delta[-1])) if alpha_delta else None,
            'beta': float(np.mean(beta_delta[-1])) if beta_delta else None,
            'mus': float(np.mean(mus_delta[-1])) if mus_delta else None,
            'R0': float(np.mean(R0_delta[-1])) if R0_delta else None
        }
    }
}

# Load existing summary if it exists, otherwise create new
if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        all_trials = json.load(f)
else:
    all_trials = {'trials': []}

# Add this trial's summary
all_trials['trials'].append(trial_summary)

# Calculate and add aggregate statistics
mse_metrics = defaultdict(list)
for trial in all_trials['trials']:
    for metric, value in trial['mse_metrics'].items():
        mse_metrics[metric].append(value)

all_trials['aggregate_statistics'] = {
    'mean_mse': {
        metric: float(np.mean(values)) for metric, values in mse_metrics.items()
    },
    'std_mse': {
        metric: float(np.std(values)) for metric, values in mse_metrics.items()
    },
    'total_trials': len(all_trials['trials'])
}

# Save updated summary
with open(summary_file, 'w') as f:
    json.dump(all_trials, f, indent=2)

print(f"\nTrial {trial_id} completed and added to summary.")
print(f"Total trials recorded: {len(all_trials['trials'])}")




