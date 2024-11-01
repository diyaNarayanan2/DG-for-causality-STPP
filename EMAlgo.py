import numpy as np 
from scipy.stats import weibull_min, exponweib 
import matplotlib.pyplot as plt 
from scipy import signal

def covid_tr_ext_j(covid_tr, n_day):
    # extends covid_tr vertically
    return np.repeat(covid_tr, n_day, axis=0)

def covid_tr_ext_i(covid_tr, n_day, n_cty):
    # extends covid_tr horizontally
    return np.tile(covid_tr, (1, n_day))

def wblval(n_day, n_cty, alpha, beta):
    # creates the weibull probability distribution function 
    time_diffs = np.arange(1, n_day+1)[:, None] - np.arange(1, n_day+1)
    wblpdf = weibull_min.pdf(time_diffs, c=beta, scale=alpha)
    wblpdf = np.tril(wblpdf)
    wbl_cty_wise = np.tile(wblpdf, (n_cty, 1))
    
    return wbl_cty_wise

def wblfit(obs, freqs):
    alpha = 2
    beta = 2
    loc = 0
    
    return alpha, beta

def poisson_model(exog_data, endo_obs_data, freqs):
    #placegolder
    #replace with DG intergated fit function
    #make this a class with a init, fit, predict and coef functions
    
    pred = np.ones(n_cty * n_day)
    
    return pred


def EStep(covariates, covid_tr, mus, R0, alpha, beta):
    '''Expectation Step in EM algorithm '''
    n_cty, n_day = covid_tr.shape
    R0_ext_j = np.repeat(R0, n_day, axis=0)
    p = R0_ext_j * wblval(n_day, n_cty, alpha, beta) * (covid_tr_ext_j(covid_tr, n_day) > 0)
    eye_mu = np.tile(np.eye(n_day), (n_cty, 1)) * np.repeat(mus, n_day, axis=0)

    lam = np.sum(p * covid_tr_ext_j(covid_tr, n_day) + eye_mu, axis=1)
    lam_eq_zero = lam == 0

    # p_c(i,j) = R_c(x^tj . 0) x w(ti - tj|α,β)
    p = np.divide(p, lam)
    p[lam_eq_zero.flatten()] = 0
    lam = lam.reshape(n_day, n_cty).T
    
    Q = np.reshape(prob_matrix * covid_tr_ext_i(covid_tr, n_day, n_cty), (n_day, n_day * n_cty))
    Q = np.reshape(np.sum(Q, axis=0), (n_cty, n_day))
    
    return lam, mus, p, Q

def MStep(R0, lam, mus, p):
    '''Maxmimsation step: after poisson fitting'''
    R0 = signal.savgol_filter(R0, window_length=10, polyorder=2, axis=1)
    R0_ext_j = np.repeat(R0, n_day, axis=0)

    lam_eq_zero = lam == 0
    mus = np.divide(mus, lam)
    mus[lam_eq_zero.flatten()] = 0

    mus = np.sum(mus * covid_tr, axis=1) / n_day
    
    # FIT WEIBULL PARAMS
    time_diffs = np.arange(1, n_day)[:, None] - np.arange(1, n_day)
    obs = np
    inter_event_freqs = covid_tr_ext_j(covid_tr, n_day) * covid_tr_ext_i(covid_tr, n_day, n_cty) * p
    inter_event_freqs = inter_event_freqs.reshape(n_day, n_cty, n_day).transpose(0, 2, 1).sum(axis=2)
    ind_ret = np.where((obs >0) & (inter_event_freqs >0))
    obs = obs[ind_ret]
    inter_event_freqs = inter_event_freqs[ind_ret]

    scale_pred, shape_pred = wblfit(obs, inter_event_freqs)
    wbl_pred = wblval(n_cty, n_day, shape_pred, scale_pred)
    
    return lam, mus, scale_pred, shape_pred
    


#EM ALGO STARTS
# INPUTS
n_cty = 2
n_day = 4
alpha = 2
beta = 2
emiter = 5
break_diff = 1e-3

covariates = np.ones((n_cty * n_day, 14))
covid_tr = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])

R0 = np.ones((n_cty,n_day))
prob_matrix = [[None for _ in range(n_cty * n_day)] for _ in range(n_day)]
bg_rate = 0.5 * np.ones(n_cty, 1)
cond_intensity = np.zeros(n_cty, n_day)

alpha_delta = []
alpha_prev = []
beta_delta = []
beta_prev = []
mus_delta = []
mus_prev = []
K0_delta = []
K0_prev = []
theta_delta = []
theta_prev = []

for itr in range(emiter):
    
    # Expectation step
    # diff params for each env
    cond_intensity, bg_rate, prob_matrix, avg_child = EStep(covariates, covid_tr, bg_rate, R0, alpha, beta)

    # prep for poisson fitting
    exog = covariates
    endo = avg_child.reshape(-1,1)

    event_freqs = covid_tr.reshape(-1,1)
    event_freqs.flatten()
    
    # POISSON REGRESSION: SUPERVISED LEARNING
    R0 = poisson_model(exog, endo, event_freqs)
    # R0 will be diff for each env
    
    cond_intensity, bg_rate, alpha, beta = MStep(R0, cond_intensity, bg_rate, prob_matrix)
    
    #CONVERGENCE
    """ Convergence check """
    if itr == 0:
        # save the first value
        alpha_prev = alpha
        beta_prev = beta
        mus_prev = cond_intensity
        R0_prev = R0
        theta_prev = poisson_model.coef
    else:
        # calculate the RMSR
        alpha_delta = np.hstack((alpha_delta, np.sqrt((alpha - alpha_prev) ** 2)))
        beta_delta = np.hstack((beta_delta, np.sqrt((beta - beta_prev) ** 2)))
        mus_delta = np.hstack((mus_delta, np.sqrt(np.mean((mus_prev - bg_rate) ** 2))))
        R0_delta = np.hstack((K0_delta, np.sqrt(np.mean((K0_prev - R0) ** 2))))
        theta_delta = np.hstack(
            (
                theta_delta,
                np.sqrt(
                    (np.sum((theta_prev - np.array(poisson_model.coef)) ** 2))
                    / len(np.array(poisson_model.coef))
                ),
            )
        )

        # save the current
        alpha_prev = alpha
        beta_prev = beta
        mus_prev = bg_rate
        K0_prev = R0
        theta_prev = np.array(poisson_model.coef)

