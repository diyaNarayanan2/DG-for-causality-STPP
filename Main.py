"""
This file parses arguments for learning a Hawkes process
    Initializes a dictionary to store all variables env wise 
    Preprocess covariates and store them env wise 
    Initialize empty variables to track EM algo performance
    Start EM algo 
        E step env wise 
        prepares endog exog and freqs for Poisson regression fitting 
    Fit on envs simultaneously 
        M step env wise 
        stores final variable values in dict 
    computes delta values 
Then it implements an EM algorithm  """


import numpy as np 
import torch
import warnings
import pandas as pd
from PrepareData import PreProcessCovariates
from MakeEnv import PopulationDensityEnvs, StateGroupedEnvs, RegionGroupedEnvs
from SyntheticEnvs import SyntheticEnvs
from EMAlgo import EStep, MStep
from poisson import *
import time
import argparse
import seaborn as sns

COVARIATES = ["Grocery_pharmacy", "Parks", "Residential", "Retail_recreation", "Transit_stations", "Workplace", "Population Density", 
              "Population_est", "#ICUbeds", "Median_Age", "Smokers%", "Diabetes%", "HeartDiseaseMosrtality", "#Hospitals"]

parser = argparse.ArgumentParser(description="Run EM algorithm with Poisson regression")
parser.add_argument("--report_path", type=str, default=r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Report_Test.csv")
parser.add_argument("--demography_path", type=str, default=r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Demo_Dconfirmed.csv")
parser.add_argument("--mobility_path", type=str, default=r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Mobility_Test.csv")
parser.add_argument("--region_path", type=str, default=r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Regions.csv")
parser.add_argument("--delta", type=int, default=4)
parser.add_argument("--n_env", type=int, default=5)
parser.add_argument("--n_days", type=int, default=10)
parser.add_argument("--emitr", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--break_diff", type=float, default=1e-3)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--n_cty", type=int, default=30)
parser.add_argument("--method", type=str, default="IRM")
parser.add_argument("--kernel_type", type=str, default="gaussian")
parser.add_argument("--env_criterion", type=str, default="population_density")
parser.add_argument("--syn_envs", type=list, default=[1.2,0.5,2.6,2.4])
args = parser.parse_args()


env_classes = {
    "population_density": PopulationDensityEnvs,
    "states": StateGroupedEnvs,
    "regions": RegionGroupedEnvs,
    "synthetic": SyntheticEnvs
}

mu = np.ones

env_args = {
    "population_density": (args.demography_path, args.mobility_path, args.report_path, args.n_env, args.n_cty),
    "states": (args.demography_path, args.mobility_path, args.report_path),
    "regions": (args.demography_path, args.mobility_path, args.report_path, args.region_path), 
    "synthetic": (args.n_cty, args.n_days, 8, 5, 2, 2, args.syn_envs)
}

if args.env_criterion not in env_classes:
    raise NotImplementedError(f"Environment criterion '{args.env_criterion}' is not implemented.")

# Dynamically create the environment object
env_class = env_classes[args.env_criterion]
env_obj = env_class(*env_args[args.env_criterion])

# Assign attributes

report_all = env_obj.covid_groups
demography_all = env_obj.demography_groups
mobility_all = env_obj.mobility_groups


# environment-wise dictionaries
env_data = [
    {
        "covariates": None,
        "covid": None,
        "key_list": None,
        "R0": np.ones((args.n_cty, args.n_days)),
        "mus": 0.5 * np.ones((args.n_cty, 1)),
        "lam": np.zeros((args.n_cty, args.n_days)),
        "prob_matrix": None,
        "Q": None
    } for _ in range(args.n_env)
]

alphas = [2] * args.n_env
betas = [2] * args.n_env

# Preprocessing
print("____________________________Preprocessing covariates:_______________________________ ")
for e in range(args.n_env): 
    cov, cases, dates, key = PreProcessCovariates(
        report_all[e], mobility_all[e], demography_all[e], args.n_days, args.delta, args.verbose
    )
    env_data[e]["covariates"] = cov
    env_data[e]["covid"] = cases
    env_data[e]["key_list"] = key

date_list = dates

# tracking updates across iterations
mus_prev = [None] * args.n_env
R0_prev = [None] * args.n_env
alphas_prev, betas_prev = [2] * args.n_env, [2] * args.n_env
alpha_delta = [[] for _ in range(args.n_env)]
beta_delta = [[] for _ in range(args.n_env)]
mus_delta = [[] for _ in range(args.n_env)]
R0_delta = [[] for _ in range(args.n_env)]

for itr in range(args.emitr):
    start_time = time.time()
    if args.verbose:
        print(f"Starting EM Iteration: {itr+1}")
    
    for e in range(args.n_env): 
        # E-step
        env_data[e]["lam"], env_data[e]["mus"], env_data[e]["prob_matrix"], env_data[e]["Q"] = EStep(
            env_data[e]["covid"], env_data[e]["mus"], env_data[e]["R0"], alphas[e], betas[e]
        )
        plt.figure(figsize=(10,6))
        sns.heatmap(env_data[e]["lam"])
        plt.title(f'Conditional intensity at itr: {e}')
        plt.xlabel('Days')
        plt.ylabel('Counties')
        plt.show()
        
        print(f"for environment {e}: ")
        print(f"Background rate spans: {min(env_data[e]["mus"])} to {max(env_data[e]["mus"])} with avg: {np.mean(env_data[e]["mus"])}")  
        print(f"Conditional intensity: {min(env_data[e]["lam"])} to {max(env_data[e]["lam"])} with avg: {np.mean(env_data[e]["lam"])}")   
        print(f"Q- Average children: {min(env_data[e]["Q"])} to {max(env_data[e]["Q"])} with avg: {np.mean(env_data[e]["Q"])}")
        print(f"Probability matrix guesses are within range: {min(env_data[e]["prob_matrix"])} to {max(env_data[e]["prob_matrix"])}")
        
    
    # Train the model
    all_covariates = [env["covariates"] for env in env_data]
    all_endog = [env["Q"].reshape(-1, 1) for env in env_data]
    all_event_freqs = [env["covid"].reshape(-1, 1) for env in env_data]
    
    if args.verbose: 
        print(f"Starting poisson model learning using Domain Generalization: {args.method}")

    if args.method == "ERM": 
        model = EmpiricalRiskMinimizer(all_covariates, all_endog, all_event_freqs, args)
    elif args.method == "IRM": 
        model = InvariantRiskMinimization(all_covariates, all_endog, all_event_freqs, args)
    elif args.method == "MMD": 
        model = MaximumMeanDiscrepancy(all_covariates, all_endog, all_event_freqs, args)
    coef = model.solution()
    coef_df = pd.DataFrame({
        'Feature':COVARIATES.flatten(),
        'Coefficients':coef.flatten()
    }).sort_values('Coefficients', ascending=False)
    print(f"Estimated coeffecients across all environments in EM itr: {itr}:")
    print(coef_df)
    
    for e in range(args.n_env):
        # Update R0 with predictions for M-step
        env_data[e]["R0"] = model.predict(env_data[e]["covariates"])
        # TODO: Confirm is R0 is supposed to be the lambda values or sampled from it 
        env_data[e]["R0"] = np.reshape(env_data[e]["R0"], (args.n_cty, args.n_days))
        
        # M-step
        env_data[e]["lam"], env_data[e]["mus"], alphas[e], betas[e] = MStep(
            env_data[e]["R0"], env_data[e]["lam"], env_data[e]["mus"], env_data[e]["prob_matrix"], env_data[e]["covid"]
        )
        print(f"Estimated weibull scale: {alphas[e]}")
        print(f"Estimated weibull shape {betas[e]}")
        
        if itr == 0:
            # Save the first iteration values
            mus_prev[e] = env_data[e]["mus"]
            R0_prev[e] = env_data[e]["R0"]
        else:
            # Calculate RMSR for convergence check
            mus_delta[e].append(np.sqrt(np.mean((mus_prev[e] - env_data[e]["mus"]) ** 2)))
            R0_delta[e].append(np.sqrt(np.mean((R0_prev[e] - env_data[e]["R0"]) ** 2)))
            alpha_delta[e].append(np.sqrt((alphas[e] - alphas_prev[e]) ** 2))
            beta_delta[e].append(np.sqrt((betas[e] - betas_prev[e]) ** 2))
            

            # Save current values for next iteration
            mus_prev[e] = env_data[e]["mus"]
            R0_prev[e] = env_data[e]["R0"]

    # Early stopping criteria
    if itr > 5:
        print("Commencing convergence check: ")
        print(f"Mus Delta: {np.mean(mus_delta, axis=0)}")
        print(f"R0 Delta: {np.mean(R0_delta, axis=0)}")
        print(f"Alpha Delta: {np.mean(alpha_delta, axis=0)}")
        print(f"Beta Delta: {np.mean(beta_delta, axis=0)}")
        
        converged = True
        for e in range(args.n_env):
            if not (np.all(np.array(mus_delta[e][-5:]) < args.break_diff) and
                    np.all(np.array(R0_delta[e][-5:]) < args.break_diff) and
                    np.all(np.array(alpha_delta[e][-5:]) < args.break_diff) and
                    np.all(np.array(beta_delta[e][-5:]) < args.break_diff)):
                converged = False
                break  # One environment not converged
        
        if converged:
            print(f"Convergence criterion met at iteration {itr + 1}. Exiting EM loop.")
            break

    if args.verbose:
        elapsed_time = time.time() - start_time
        print(f"Iteration {itr+1} completed in {elapsed_time:.2f} seconds.")


print("EM Algorithm Completed.")
        
    