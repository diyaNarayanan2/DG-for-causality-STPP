import numpy as np
import random
import os
from scipy.stats import weibull_min

class SyntheticEnvs: 
    def __init__(self, n_cty, n_day, alpha, beta, mu, env_list, dim, h): 
        """
        Initializes variables for synthetic data generation
        assume n_cty is number of samples
        Days ,alpha, beta, mu, environment values, covariate dimension
        h is the seed for the random number generator
        covariates, weights generated here
        Only one layer of dimensionality, mu and cov constant across days 
        """
        # Set seeds for all possible sources of randomness
        self.h = h
        np.random.seed(self.h)
        random.seed(self.h)
        os.environ['PYTHONHASHSEED'] = str(self.h)
        
        self.n_cty = n_cty
        self.n_day = n_day
        self.alpha = alpha 
        self.beta = beta 
        self.mu = mu
        
        # generate events w mob demo data for 1 env
        self.dim_x = dim // 2
        self.causal_cov = np.random.rand(self.n_cty, self.dim_x)
        self.wxy = np.random.rand(self.dim_x, self.dim_x)  # scaled to [0, 0.25] range
        
        # generate acausal data and events for all envs
        covariate_groups, case_count_groups, lambda_groups = self.makeEnvs(env_list)   
        
        self.case_count_envs = case_count_groups
        self.covariate_envs = covariate_groups
        self.lambda_envs = lambda_groups
        
    def hawkes_discrete_simulation(self, mu, R):
        """
        Simulates a Hawkes process with discrete time steps using the thinning method.
        generates exponentially decaying continuous time steps, then calculates probability 
        of acceptance using discretized binned timestep
        
        Lambda = mu + Sum{R(X0)w(t-t-j) : t_j<t}        
        Parameters:
            mu (numpy array): Background rate
            R (numpy array): Reproductive rate (shape: n_cty, 1).
            xw (numpy array): Covariates 
            
        Returns:
            events (numpy array): event counts for each county (shape: n_cty, T).
            lambda_t (numpy array): conditional intensity values (shape: n_cty, T).
        """
        T = self.n_day  # time steps
                
        # Estimate lambda_max using Weibull maximum
        t_peak = self.alpha * ((self.beta - 1) / self.beta) ** (1 / self.beta) if self.beta > 1 else 0
        w_peak = weibull_min.pdf(t_peak, self.beta, scale=self.alpha)
        R_max = np.max(R) 
        lambda_max = np.max(mu) + np.sum(R_max * w_peak)
        
        # outputs
        events = np.zeros((self.n_cty, T), dtype=int)
        lambda_t = np.zeros((self.n_cty, T))
        
        # samples events n_cty times for n_cty event rates 
        for i in range(self.n_cty):  
            t = 0
            # Add safety counter to prevent infinite loops
            iteration_count = 0
            max_iterations = 1000000  # Adjust this number as needed
            
            while t < T:
                # Add safety check
                iteration_count += 1
                if iteration_count > max_iterations:
                    print(f"Warning: Maximum iterations reached for county {i}")
                    break
                
                delta_t = np.random.exponential(1 / lambda_max)
                t_candidate = t + delta_t
                
                # Add minimum time step to ensure progress
                if delta_t < 1e-10:  # Prevent extremely small time steps
                    delta_t = 1e-10
                    t_candidate = t + delta_t
                
                if t_candidate >= T:
                    break
                
                t_discrete = int(np.floor(t_candidate))
                
                # triggering kernel influence 
                past_events = np.where(events[i, :t_discrete] > 0)[0]
                hist_influence = np.sum([
                    R[i] * weibull_min.pdf(t_discrete - t_j, self.beta, scale=self.alpha)
                    for t_j in past_events
                ])
                lambda_t_candidate = mu[i] + hist_influence
                
                if np.random.uniform(0, 1) <= (lambda_t_candidate / lambda_max):
                    events[i, t_discrete] += 1
                
                t = t_candidate
            
            # Compute lambda_t for all time steps
            for t in range(T):
                past_events = np.where(events[i, :t] > 0)[0]
                hist_influence = np.sum([
                    R[i] * weibull_min.pdf(t - t_j, self.beta, scale=self.alpha)
                    for t_j in past_events
                ])
                lambda_t[i, t] = mu[i] + hist_influence
        print(f"Sampling complete for county {i}")
        
        return events, lambda_t     

    def trueSolution(self): 
        w_reduced = np.sum(self.wxy, axis=1)
        sol = np.concatenate([w_reduced, np.zeros(self.dim_x)])
        # shape of sol is (2dim_x, 1); dim_x is number of causal covariates
        return sol     
            
    def makeEnvs(self, env_list): 
        """
        Makes a set of environments using an input env_list
        one set of acausal cov geenrated with env dependent noise for each env
        X: causal and acausal cov concatenated 
        Y: y summed 
        (X, Y) returned 
        after which R will be poisson sampled from Y 
        and case count will be generated from R, mu(same for all envs)
        init shuld return sets of case count and X for each env (observed variables for EM)        
        """
        x = self.causal_cov # shape (n_cty, dim_x)
        y = x @ self.wxy  
        wyz = np.random.rand(self.dim_x, self.dim_x)
        all_covariates = []
        all_lambdas = []
        for _, e in enumerate(env_list): 
            z_e = y @ wyz + np.random.rand(self.n_cty, self.dim_x) * e  # Removed n_cty dimension
            cov = np.concatenate([x, z_e], axis=1)  # Concatenate vectors
            all_covariates.append(cov)
            
        w = np.sum(self.wxy, axis=1, keepdims=True)
        R = np.exp(x @ w).flatten() # shape (n_cty, 1)
        
        all_case_counts = []
        for e in range(len(env_list)): 
            case_count_e, lambda_e = self.hawkes_discrete_simulation(self.mu, R)
            all_case_counts.append(case_count_e)
            all_lambdas.append(lambda_e)

        return all_covariates, all_case_counts, all_lambdas
        
            

        
        
        
        


