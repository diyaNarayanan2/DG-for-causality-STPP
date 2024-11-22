import numpy as np 
from scipy.stats import weibull_min, exponweib 
import matplotlib.pyplot as plt 
from scipy import signal

def covid_tr_ext_j(covid_tr, n_day):
    # extends covid_tr vertically
    return np.repeat(covid_tr, n_day, axis=0)

def covid_tr_ext_i(covid_tr, n_day_tr, n_cty):
    # Transpose covid_tr to split into blocks by row, then reshape back
    covid_tr_blocks = covid_tr.T.reshape(n_cty, n_day_tr).T
    # Tile horizontally to repeat each block n_day_tr times
    return np.tile(covid_tr_blocks, (1, n_day_tr))

def wblval(n_day, n_cty, alpha, beta):
    # creates the weibull probability distribution function 
    time_diffs = np.arange(1, n_day+1)[:, None] - np.arange(1, n_day+1)
    wblpdf = weibull_min.pdf(time_diffs, c=beta, scale=alpha)
    wblpdf = np.tril(wblpdf)
    wbl_cty_wise = np.tile(wblpdf, (n_cty, 1))
    
    return wbl_cty_wise

def wblfit(obs, freqs):
    # Expand data according to frequencies
    expanded_data = np.repeat(obs, freqs)
    shape, loc, scale = weibull_min.fit(expanded_data, floc=0) 

    return scale, shape

def EStep(covid_tr, mus, R0, alpha, beta):
    '''Expectation Step in EM algorithm '''
    n_cty, n_day = covid_tr.shape
    R0_ext_j = np.repeat(R0, n_day, axis=0)
    prob_matrix = R0_ext_j * wblval(n_day, n_cty, alpha, beta) * (covid_tr_ext_j(covid_tr, n_day) > 0)
    eye_mu = np.tile(np.eye(n_day), (n_cty, 1)) * np.repeat(mus, n_day, axis=0)

    lam = np.sum(prob_matrix * covid_tr_ext_j(covid_tr, n_day) + eye_mu, axis=1)
    lam = lam.reshape(lam.shape[0], 1)
    lam_eq_zero = lam == 0

    # p_c(i,j) = R_c(x^tj . 0) x w(ti - tj|α,β)
    prob_matrix = np.divide(prob_matrix, lam)
    prob_matrix[lam_eq_zero.flatten()] = 0
    lam = lam.reshape(n_day, n_cty).T
    
    p = np.reshape(prob_matrix, (n_day, n_day * n_cty))
    Q = np.reshape(p * covid_tr_ext_i(covid_tr, n_day, n_cty), (n_day, n_day * n_cty))
    Q = np.reshape(np.sum(Q, axis=0), (n_cty, n_day))
    
    return lam, mus, prob_matrix, Q

def MStep(R0, lam, mus, p, covid_tr):
    '''Maxmimsation step: after poisson fitting'''
    n_cty, n_day = covid_tr.shape
    print(f"Shape of R0: {R0.shape}")
    R0 = signal.savgol_filter(R0, window_length=10, polyorder=2, axis=1)
    
    lam_eq_zero = lam == 0
    mus = np.divide(mus, lam, where=~lam_eq_zero, out=np.zeros_like(mus))
    mus = np.sum(mus * covid_tr, axis=1) / n_day
    
    # FIT WEIBULL PARAMS
    time_diffs = np.arange(1, n_day)[:, None] - np.arange(1, n_day)
    obs = np.tril(time_diffs)
    inter_event_freqs = covid_tr_ext_j(covid_tr, n_day) * covid_tr_ext_i(covid_tr, n_day, n_cty) * p
    inter_event_freqs = inter_event_freqs.reshape(n_day, n_cty, n_day).transpose(0, 2, 1).sum(axis=2)
    ind_ret = np.where((obs >0) & (inter_event_freqs >0))
    obs = obs[ind_ret]
    inter_event_freqs = inter_event_freqs[ind_ret]

    scale_pred, shape_pred = wblfit(obs, inter_event_freqs)
    
    return lam, mus, scale_pred, shape_pred
    

