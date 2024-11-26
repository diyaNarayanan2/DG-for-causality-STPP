import numpy as np
from scipy.stats import weibull_min

class SyntheticEnvs: 
    def __init__(self, n_cty, n_day, n_demo_cov, n_mob_cov, alpha, beta, env_list, dim): 
        self.n_cty = n_cty
        self.n_day = n_day
        self.n_demo_cov = n_demo_cov
        self.n_mob_cov = n_mob_cov
        self.alpha = alpha 
        self.beta = beta 
        
        '''
        true_demo_data = np.random.uniform(0,3,size=(self.n_cty, self.n_demo_cov))
        true_mob_data = np.random.uniform(0,3,size=(self.n_cty, self.n_mob_cov, self.n_day))
        self.true_weights = np.random.rand(self.n_demo_cov + self.n_mob_cov)
        demography = np.repeat(np.expand_dims(true_demo_data, axis=2), self.n_day, axis=2) # shape: (n_cty , n_demo_cov, n_day)
        self.covariates = np.concatenate((demography, true_mob_data), axis=1) #shape: n_cty, n_cov, n_day
        xw = np.einsum('ijk,j->ik', self.covariates, self.weights_xy) #shape: n_cty, n_day
        R = np.random.poisson(xw)
        #calls hawkes sim to generate covid data for different adjusted X values for diff envs 
        events, lam_t = self.hawkes_discrete_simulation(self.mu, R, xw)
        self.covid_groups = events 
        self.cond_intensity = lam_t
        #self.covid_groups = make_covid_groups()
        #self.mobility_groups = make mobility_groups()
        #self.demography_groups = make_demo_groups()
        '''
        
        #generating true causal covariates (invariant across envs)
        dim_x = dim // 2
        self.causal_cov = np.random.rand(self.n_cty, dim_x)
        self.wxy = np.random.rand(dim_x, dim_x)
        self.mu = 0.65 * np.ones(self.n_cty)
        covariate_groups, case_count_groups = self.makeEnvs(env_list, dim_x)
        
        self.case_count_envs = case_count_groups
        self.covariate_envs = covariate_groups
        
        
    def hawkes_discrete_simulation(self, mu, R):
        """
        Simulates a Hawkes process with discrete time steps using the thinning method.
        generates exponentially decaying continuous time steps, then calculates probability 
        of acceptance using discretized binned timestep
        
        Lambda = mu + Sum{R(X0)w(t-t-j) : t_j<t}        
        Parameters:
            mu (numpy array): Background rate
            R (numpy array): Reproductive rate (shape: n_cty, T).
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
        
        for i in range(self.n_cty):  
            t = 0  # Reset current time for each county
            
            while t < T:
                delta_t = np.random.exponential(1 / lambda_max)
                t_candidate = t + delta_t
                
                if t_candidate >= T:
                    break
                
                t_discrete = int(np.floor(t_candidate))
                
                # triggering kernel influence 
                past_events = np.where(events[i, :t_discrete] > 0)[0]
                hist_influence = np.sum([
                    R[i, t_discrete] * weibull_min.pdf(t_discrete - t_j, self.beta, scale=self.alpha)
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
                    R[i, t] * weibull_min.pdf(t - t_j, self.beta, scale=self.alpha)
                    for t_j in past_events
                ])
                lambda_t[i, t] = mu[i] + hist_influence
        
        return events, lambda_t    

    
    def solution(self): 
        # add weights along axes to make 1 d
        # weights_xy is a dim, dim matrix
        sol = self.weights_xy     
            
    
    def makeEnvs(self, env_list, dim): 
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
        x = self.causal_cov
        y = x @ self.wxy # optional can add env varying noise 
        # wxy is the true set of weights 
        # y caused by x with no noise       
        wyz = np.random.rand(dim, dim)
        all_covariates = []
        
        for _, e in enumerate(env_list): 
            # z is acausal covariates 
            # total covariates to train w is 2 dim 
            z_e =  y @ wyz + np.random.rand(self.n_cty, dim) * e
            cov = np.concatenate([x, z_e], axis=1)
            all_covariates.append(cov)
            
        w = np.sum(self.wxy, axis=1, keepdims=True)
        R = np.tile(np.random.poisson(x @ w, size=(self.n_cty, 1)), self.n_day)
        # shape of x @ w: (n,1)
        # one reproductive rate for each county 
        # does not vary day wise but repeated for function input  
        all_case_counts = []
        for e in range (len(env_list)): 
            case_count_e, _ = self.hawkes_discrete_simulation(self.mu, R)
            all_case_counts.append(case_count_e)
            
        return all_covariates, all_case_counts
        
        
            

        
        
        
        


