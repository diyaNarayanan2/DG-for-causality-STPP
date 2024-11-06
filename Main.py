import numpy as np 
import torch
import warnings
import pandas as pd
from PrepareData import PreProcessCovariates
from MakeEnv import PopulationDensityEnvs, StateGroupedEnvs, RegionGroupedEnvs
from EMAlgo import EStep, MStep
from poisson import PoissonRegressorGLM
import argparse

parser = argparse.ArgumentParser(description="Run EM algorithm with Poisson regression")
parser.add_argument("--report_path", type=str, default=r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Report_Test.csv")
parser.add_argument("--demography_path", type=str, default=r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Demo_Dconfirmed.csv")
parser.add_argument("--mobility_path", type=str, default=r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Mobility_Test.csv")
parser.add_argument("--delta", type=int, default=4)
parser.add_argument("--n_env", type=int, default=5)
parser.add_argument("--n_days", type=int, default=10)
parser.add_argument("--emitr", type=int, default=10)
parser.add_argument("--verbose", type=int, default=1)
args = parser.parse_args()

# Global Constants
BREAK_DIFF = 1e-3

PopD_Envs = PopulationDensityEnvs(args.demography_path, args.mobility_path, args.report_path, args.n_env, args.n_days)
report_all = PopD_Envs.covid_groups
demography_all = PopD_Envs.demography_groups
mobility_all = PopD_Envs.mobility_groups

# Initialize data holders for environments
covariates_all, key_lists_all, covid_all = [], [], []
R0_all, prob_matrix_all, mus_all, lam_all, Q_all = [], [], [], [], []
alphas, betas = [2] * args.n_env, [2] * args.n_env

# initial values
R0 = np.ones((args.n_env, args.n_days))
prob_matrix = [[None for _ in range(args.n_env * args.n_days)] for _ in range(args.n_days)]
bg_rate = 0.5 * np.ones((args.n_env, 1))
cond_intensity = np.zeros((args.n_env, args.n_days))

# Preprocessing
for e in range(args.n_env): 
    cov, cases, dates, key = PreProcessCovariates(
        report_all[e], mobility_all[e], demography_all[e], args.n_days, args.delta, args.verbose
    )
    covariates_all.append(cov)
    covid_all.append(cases)
    key_lists_all.append(key)
    R0_all.append(R0)
    prob_matrix_all.append(prob_matrix)
    mus_all.append(bg_rate)
    lam_all.append(cond_intensity)

date_list = dates

# Empty arrays for tracking updates across iterations
alpha_delta, alpha_prev, beta_delta, beta_prev = [], [], [], []
mus_delta, mus_prev, K0_delta, K0_prev = [], [], [], []
theta_delta, theta_prev = [], []

# EM Algorithm Training Loop
for itr in range(args.emitr):
    if args.verbose:
        print(f"Starting EM Iteration: {itr+1}")
    
    for e in range(args.n_env): 
        # E-step
        lam_all[e], mus_all[e], prob_matrix_all[e], Q_all[e] = EStep(
            prob_matrix_all[e], covid_all[e], mus_all[e], R0_all[e], alphas[e], betas[e]
        ) 

        # Prepare data for Poisson regression
        exog = covariates_all[e]
        endo = Q_all[e].reshape(-1, 1)
        event_freqs = covid_all[e].reshape(-1, 1)
        
        if args.verbose:
            print(f"Training Poisson model on environment {e+1}:")

        # Train Poisson regression model
        model = PoissonRegressorGLM(exog, endo, event_freqs)
        
        # Output model summary and coefficients if verbose
        if args.verbose:
            model.print_summary()
        
        coefficients = model.get_coefficients()
        if args.verbose:
            print("Coefficients:", coefficients)
        
        # Update R0 with predictions for M-step
        R0_all[e] = model.predict(covariates_all[e])
        
        # M-step
        lam_all[e], mus_all[e], alphas[e], betas[e] = MStep(
            R0_all[e], lam_all[e], mus_all[e], prob_matrix_all[e], covid_all[e]
        )
