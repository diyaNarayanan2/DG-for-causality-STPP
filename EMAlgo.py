import numpy as np 
from scipy.stats import weibull_min, exponweib 
import matplotlib.pyplot as plt 
from scipy import signal
import seaborn as sns

def covid_tr_ext_j(covid_tr, n_day):
    # extends covid_tr vertically
    return np.repeat(covid_tr, n_day, axis=0)

def covid_tr_ext_i(covid_tr, n_day_tr, n_cty):
    # Transpose covid_tr to split into blocks by row, then reshape back
    covid_tr_blocks = covid_tr.T.reshape(n_cty, n_day_tr).T
    # Tile horizontally to repeat each block n_day_tr times
    return np.tile(covid_tr_blocks, (1, n_day_tr))

def wblval(n_day, n_cty, alpha, loc, beta):
    eps = 1e-06
    # creates the weibull probability distribution function 
    time_diffs = np.arange(1, n_day+1)[:, None] - np.arange(1, n_day+1)
    wblpdf = weibull_min.pdf(time_diffs, c=beta, scale=alpha)
    wblpdf = np.tril(wblpdf)
    wbl_cty_wise = np.tile(wblpdf, (n_cty, 1))
    
    return wbl_cty_wise

def wblfit(obs, freqs):
    # Expand data according to frequencies
    expanded_data = np.repeat(obs, np.round(freqs).astype(int))
    beta, loc, alpha = weibull_min.fit(expanded_data, floc=0) 

    return alpha, loc, beta

def ExpStep(covid_tr, mus, R0, alpha, loc, beta): 
    """_summary_

    Args:
        covid_tr (array | (n_cty, n_day)): daily case count
        mus (array | (n_cty, 1)): background rate per county
        R0 (array | (n_cty, n_day)): Reproductivity rate
        alpha (int): weibull scale 
        beta (int): weibull shape
    """
    n_cty, n_day = covid_tr.shape
    R0_ext_j = np.repeat(R0, n_day, axis=0) # shape: n_cty*n_day, n_day
    # each row repeated n_day times
    trig_comp = R0_ext_j * wblval(n_day, n_cty, alpha, loc, beta) * (covid_tr_ext_j(covid_tr, n_day) > 0)
    # lam = mu + [sum for t_j < t]R(0)w(t-t_j)    
    mu_comp = np.tile(np.eye(n_day), (n_cty, 1)) * np.repeat(mus, n_day, axis=0)
    
    # Create figure with two subplots side by side
    plt.figure(figsize=(12, 5))
    # Plot mu component
    plt.subplot(1, 2, 1)    
    sns.heatmap(mu_comp)
    plt.title('Background Rate Component')
    
    # Plot triggering component
    plt.subplot(1, 2, 2)
    sns.heatmap(trig_comp*covid_tr_ext_j(covid_tr, n_day))
    plt.title('Triggering Component')
    
    plt.suptitle(f'Components of Lambda')
    plt.tight_layout()
    plt.show()
    
    # mu on t_j = t_i ;else 0
    lambda_t = np.sum(mu_comp + trig_comp * covid_tr_ext_j(covid_tr, n_day), axis=1, keepdims=True) 
    p_c_ij = np.divide(trig_comp, lambda_t, where= lambda_t != 0) 
    p_c_ii = np.divide(mu_comp, lambda_t, where= lambda_t != 0)
    print(f"max p_c_ii: {np.max(p_c_ii)}, min p_c_ii: {np.min(p_c_ii)}")

    #lambda gets broadcasted
    full_p = mu_comp + p_c_ij
    # full probability matrix for each (county, day, day)  
    
    P_c_j = np.reshape(p_c_ij, (n_day, n_day * n_cty))
    # expected children = probability of trigerring  * number of events 
    P_c_j = np.reshape(np.sum(P_c_j, axis=0), (n_cty, n_day))
    # sum of probabilities p_c(i,j) across all days i for some t_j 
    Q = P_c_j * covid_tr
    print(f"max Q: {np.max(Q)}, min Q: {np.min(Q)}")
    # expected children = total trigerring probabiltiy * no. of events 
    
    #poisson regression between Q and X_c
    
    return lambda_t, p_c_ij, p_c_ii, Q
    
def MaxStep(R0, p_c_ii, p_c_ij, covid_tr): 
    """_summary_

    Args:
        covid_tr (array | (n_cty, n_day)): daily case count
        mus (array | (n_cty, 1)): background rate per county
        R0 (array | (n_cty, n_day)): Reproductivity rate
        lambda_t (array | (n_cty, n_day)): Conditional intenisty
        p_c_ii (array | (n_cty*n_day, n_day)): proability that event caused by itself
        = mu / lambda when t_i = t_j, 0 elsewhere
        full_p (array | (n_cty*n_day, n_day)): probability matrix for (i,j) event pairs
    """
    n_cty, n_day = covid_tr.shape
    mus = np.sum(p_c_ii, axis=1, keepdims=True)  # shape: (n_cty*n_day, 1)
    mus = mus.reshape(n_cty, n_day)
    # mu_c = [sum over t_i]p_c(i,i)/lambda(t_i)
    mus = np.sum(mus, axis=1, keepdims=True) / n_day
    mus = np.clip((mus), 0, 0.5)
    
    # FIT WEIBULL PARAMS
    time_diffs = np.arange(1, n_day+1)[:, None] - np.arange(1, n_day+1)
    obs = np.tril(time_diffs)
    inter_event_freqs = p_c_ij * covid_tr_ext_j(covid_tr, n_day)
    inter_event_freqs = np.sum(inter_event_freqs.reshape(n_cty, n_day, n_day), axis=0)
    ind_ret = np.where((obs >0) & (inter_event_freqs >0))
    obs = obs[ind_ret]
    freqs = inter_event_freqs[ind_ret]

    alpha, loc, beta = wblfit(obs, freqs)
    alpha = np.clip(alpha, 0.1, 10)
    beta = np.clip(beta, 0.5, 5)
    
    # Plot the Weibull PDF
    plt.figure(figsize=(8, 6))
    time_range = np.linspace(0, np.max(obs), 100)
    pdf_values = weibull_min.pdf(time_range, c=beta, loc=loc, scale=alpha)
    
    plt.plot(time_range, pdf_values, 'r-', label=f'Fitted Weibull (α={alpha:.2f}, β={beta:.2f}, loc={loc:.2f})')
    plt.hist(obs, weights=freqs/np.sum(freqs), density=True, bins=30, alpha=0.5, label='Data Distribution')
    
    plt.title('Fitted Weibull Distribution')
    plt.xlabel('Time Difference (days)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return mus, alpha, loc, beta
    
def EStep(covid_tr, mus, R0, alpha, beta):
    '''Expectation Step in EM algorithm '''
    n_cty, n_day = covid_tr.shape
    R0_ext_j = np.repeat(R0, n_day, axis=0)
    prob_matrix = R0_ext_j * wblval(n_day, n_cty, alpha, beta) * (covid_tr_ext_j(covid_tr, n_day) > 0)
    eye_mu = np.tile(np.eye(n_day), (n_cty, 1)) * np.repeat(mus, n_day, axis=0)

    lam = np.sum(prob_matrix * covid_tr_ext_j(covid_tr, n_day) + eye_mu, axis=1, keepdims=True)
    #expectation of conditional intensity

    lam_eq_zero = lam == 0

    # p_c(i,j) = R_c(x^tj . 0) x w(ti - tj|α,β)
    prob_matrix = np.divide(prob_matrix, lam)
    prob_matrix[lam_eq_zero.flatten()] = 0
    lam = lam.reshape(n_day, n_cty).T
    
    p = np.reshape(prob_matrix, (n_day, n_day * n_cty))
    #reshape only takes place for multiplication for Q 
    # TODO check reshape equivalency to matlab 
    Q = np.reshape(p * covid_tr_ext_i(covid_tr, n_day, n_cty), (n_day, n_day * n_cty))
    Q = np.reshape(np.sum(Q, axis=0), (n_cty, n_day))
    
    return lam, mus, prob_matrix, Q

def MStep(R0, lam, mus, p, covid_tr):
    '''Maxmimsation step: after poisson fitting'''
    n_cty, n_day = covid_tr.shape
    R0 = signal.savgol_filter(R0, window_length=10, polyorder=2, axis=1)
    
    lam_eq_zero = lam == 0
    mus = np.divide(mus, lam, where=~lam_eq_zero, out=np.zeros_like(lam))
    #mus = (np.sum(mus * covid_tr, axis=1) / n_day).reshape(-1,1)
    # TODO check against matlab impleemntation
    mus = (np.sum(mus, axis=1) / n_day).reshape(-1,1)
    
    # FIT WEIBULL PARAMS
    time_diffs = np.arange(1, n_day+1)[:, None] - np.arange(1, n_day+1)
    obs = np.tril(time_diffs)
    inter_event_freqs = covid_tr_ext_j(covid_tr, n_day) * covid_tr_ext_i(covid_tr, n_day, n_cty).T * p
    inter_event_freqs = inter_event_freqs.reshape(n_day, n_cty, n_day).transpose(0, 2, 1).sum(axis=2)
    ind_ret = np.where((obs >0) & (inter_event_freqs >0))
    obs = obs[ind_ret]
    inter_event_freqs = inter_event_freqs[ind_ret]

    scale_pred, shape_pred = wblfit(obs, inter_event_freqs)
    
    return lam, mus, scale_pred, shape_pred
    

