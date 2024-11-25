import numpy as np
from scipy.stats import weibull_min

n_cty = 20
T = n_day = 10

n_demo_cov = 8
n_mob_cov = 6
#Full true data declaration
true_alpha = 1.5
true_beta = 2

class SyntheticEnvs: 
    def __init__(self, n_cty, n_day, n_demo_cov, n_mob_cov, alpha, beta, env_list): 
        self.n_cty = n_cty
        self.n_day = n_day
        self.n_demo_cov = n_demo_cov
        self.n_mob_cov = n_mob_cov
        self.alpha = alpha 
        self.beta = beta 
        
        true_demo_data = np.random.uniform(0,3,size=(self.n_cty, self.n_demo_cov))
        true_mob_data = np.random.uniform(0,3,size=(self.n_cty, self.n_mob_cov, self.n_day))
        self.true_weights = np.random.rand(self.n_demo_cov + self.n_mob_cov)
        true_mu = 0.5 * np.ones(self.n_cty)
        
        demography = np.repeat(np.expand_dims(true_demo_data, axis=2), self.n_day, axis=2) # shape: (n_cty , n_demo_cov, n_day)
        covariates = np.concatenate((demography, true_mob_data), axis=1) #shape: n_cty, n_cov, n_day
        xw = np.einsum('ijk,j->ik', covariates, self.true_weights) #shape: n_cty, n_day
        R = np.random.poisson(xw)
        #calls hawkes sim to generate covid data for different adjusted X values for diff envs 
        events, lam_t = self.hawkes_discrete_simulation(true_mu, R, xw)
        self.covid_groups = events 
        self.cond_intensity = lam_t
        
    def hawkes_discrete_simulation(self, mu, R, xw):
        """
        Simulates a Hawkes process with discrete time steps using the thinning method.
        
        Parameters:
            T (int): Total simulation time (number of time steps).
            mu (float): Baseline intensity.
            R (numpy array): Reproductive rate over time (shape: T).
            XW (numpy array): Covariates affecting R (not directly used here but can extend the model).
            alpha (float): Weibull scale parameter.
            beta (float): Weibull shape parameter.
        
        Returns:
            events (numpy array): Array of event counts at each time step (shape: T).
            lambda_t (numpy array): Array of conditional intensity values at each time step (shape: T).
        """
        events = np.zeros((self.n_cty, self.n_day), dtype=int)  # vent counts at each time step
        lambda_t = np.zeros((self.n_cty, self.n_day))           # lambda(t) 
        
        # Estimate lambda_max using weibull max
        t_peak = self.alpha * ((self.beta - 1) / self.beta) ** (1 / self.beta) if self.beta > 1 else 0
        w_peak = weibull_min.pdf(t_peak, self.beta, scale=self.alpha)
        R_max = np.max(R)  # Maximum R(XW)
        lambda_max = np.max(self.mu) + np.sum(R_max * w_peak)  
        #weibull kernel for history calc
        weibull_kernel = weibull_min.pdf(np.arange(1, T + 1), self.beta, scale=self.alpha)
        
        for i in range(len(xw)): 
            # for each county
            
            for t in range(self.n_day):
                if t == 0:
                    lambda_t[i][t] = mu[i] 
                else:
                    history_influence = 0
                    for t_j in range(t):
                        history_influence += R[i][t_j] * weibull_kernel[t - t_j - 1] * events[i][t_j]
                    lambda_t[i][t] = mu[i] + history_influence
                
                # thinning method
                if np.random.uniform(0, 1) <= (lambda_t[i][t] / lambda_max):
                    events[t] += 1
        
        return events, lambda_t 
    
    def solution(self): 
        sol = self.true_weights
        


