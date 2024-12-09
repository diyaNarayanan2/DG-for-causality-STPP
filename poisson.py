import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

from torch.autograd import grad

import scipy.optimize
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt


def pretty(vector):
    # Convert to tensor if not already
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)
    vlist = vector.flatten().tolist()  # Using flatten() instead of view(-1)
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

def weighted_poisson_nll(input, target, weights):
    # Standard Poisson NLL terms
    loss = input - target * torch.log(input + 1e-8)
    # Multiply each observation's loss by its frequency weight
    weighted_loss = (loss * weights).mean()
    return weighted_loss


# For each method, training and compilation of results is done upon initilization, only solution is returned on solution fucntion call
# each method takes numpy input and returns numpy output 
class PoissonDevianceLoss(nn.Module):
    def __init__(self):
        super(PoissonDevianceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        """
        Compute the deviance loss for Poisson regression.

        Parameters:
        y_true (Tensor): Observed target values.
        y_pred (Tensor): Predicted mean values (must be positive).

        Returns:
        Tensor: The deviance loss.
        """
        if torch.any(y_pred <= 0):
            raise ValueError("Predicted values must be strictly positive.")

        # Compute deviance components
        term1 = y_true * torch.log(y_true / y_pred + 1e-8)  # Add small epsilon to avoid log(0)
        term1 = torch.where(y_true > 0, term1, torch.zeros_like(term1))  # Handle 0*log(0) -> 0
        term2 = y_true - y_pred

        # Deviance formula
        deviance = 2 * torch.sum(term1 - term2)
        return deviance
    
    
class InvariantRiskMinimization(object):
    def __init__(self, exog, endog, freqs, args):
        """
        Initialize and train the model with the given environments and arguments.

        Args:
            environments: List of tuples (x, y), where x and y are tensors for each environment.
            args: Dictionary of hyperparameters, e.g., learning rate and number of iterations.
        """
        self.best_reg = 0
        self.best_err = float("inf")
        
        exog = torch.tensor(np.array(exog), dtype=torch.float32)
        endog = torch.tensor(np.array(endog), dtype=torch.float32)
        freqs = torch.tensor(np.array(freqs), dtype=torch.float32)

        x_val = exog[-1]
        y_val = endog[-1]
        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(exog[:-1], endog[:-1], freqs[:-1], args, reg=reg)
            err = torch.mean((torch.exp(x_val @ self.solution()) - y_val) ** 2).item()

            if args.verbose:
                print(f"IRM (reg={reg:.3f}) has validation error: {err:.3f}.")

            if err < self.best_err:
                self.best_err = err
                self.best_reg = reg
                self.best_phi = self.phi.clone()

        #self.phi = self.best_phi

    def train(self, exog, endog, freqs, args, reg=0):
        """
        Train the IRM model using the given environments.

        Args:
            environments: List of tuples (x, y), where x and y are tensors for each environment.
            args: Dictionary of hyperparameters.
            reg: Regularization coefficient for IRM penalty.
        """
        dim_x = exog[0].size(1)
        self.phi = nn.Parameter(torch.empty(dim_x, dim_x))
        nn.init.xavier_uniform_(self.phi)
        self.w = nn.Parameter(torch.empty(dim_x, 1))
        nn.init.xavier_uniform_(self.w)

        optimizer = torch.optim.Adam([self.phi, self.w], lr=args.lr, weight_decay=1e-5)
        
        # Custom weighted Poisson loss function

        for iteration in range(args.n_iterations):
            total_penalty = 0
            total_error = 0

            for x_e, y_e, f in zip(exog, endog, freqs):
                input = torch.exp(x_e @ self.phi @ self.w)
                # Apply weighted loss for this environment
                error_e = weighted_poisson_nll(input, y_e, f)
                #error_e = nn.PoissonNLLLoss(log_input=False, reduction='mean')(input, y_e)
                
                # Compute gradient and ensure it's scalar
                grad_e = grad(error_e, self.w, create_graph=True)[0]
                penalty_e = grad_e.pow(2).mean()
                
                total_penalty += penalty_e
                total_error += error_e

            loss = reg * total_error + (1 - reg) * total_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.verbose and iteration % 100 == 0:
                w_str = pretty(self.solution())
                print(
                    "{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(
                        iteration, reg, total_error.item(), total_penalty.item(), w_str
                    ))

    def solution(self):
        """
        Returns the learned model parameters as a numpy array.
        """
        return (self.phi @ self.w).view(-1, 1).detach().numpy()
    
    def predict(self, X): 
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        # Compute the logits using the learned parameters
        pred = X @ torch.tensor(self.solution(), dtype=torch.float32)
        rates = torch.exp(pred)

        # Return the predicted Poisson rates (lambda)
        return rates.detach().numpy()
            

class MaximumMeanDiscrepancy(object):
    def __init__(self, exog, endog, freqs, args):
        best_reg = 0
        best_err = 1e6
        
        exog = torch.tensor(np.array(exog), dtype=torch.float32)
        endog = torch.tensor(np.array(endog), dtype=torch.float32) 
        freqs = torch.tensor(np.array(freqs), dtype=torch.float32)

        x_val = exog[-1]
        y_val = endog[-1]

        for gamma in [1e-3, 1e-2, 1e-1, 1]:
            self.train(exog[:-1], endog[:-1], freqs[:-1], args, gamma=gamma)   
            err = torch.mean((torch.exp(x_val @ self.solution()) - y_val) ** 2).item()

            if args.verbose:
                print("MMD (gamma={:.3f}) has {:.3f} validation error.".format(gamma, err))

            if err < best_err:
                best_err = err
                best_gamma = gamma
                best_phi = self.phi.clone()
        # Save the best model parameters
        #self.phi = best_phi

    def train(self, exog, endog, freqs, args, gamma=0):
        dim_x = exog[0].size(1)
        num_envs = exog.shape[0]

        # Initialize phi and weights
        self.phi = nn.Parameter(torch.empty(dim_x, dim_x))
        nn.init.xavier_uniform_(self.phi)
        self.w = torch.empty(dim_x, 1)
        nn.init.xavier_uniform_(self.w)
        self.w.requires_grad = True

        # Optimizer and loss function for Poisson regression
        opt = torch.optim.Adam([self.phi, self.w], lr=args.lr, weight_decay=1e-5)
        #poisson_loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='mean')

        for iteration in range(args.n_iterations):
            penalty = 0
            error = 0

            # Compute error and MMD penalty for each pair of environments
            for i, (x_e1, y_e1, f) in enumerate(zip(exog, endog, freqs)):
                input = torch.exp(x_e1 @ self.phi @ self.w)
                error += weighted_poisson_nll(input, y_e1, f) # * f

                for j, (x_e2, y_e2, f) in enumerate(zip(exog, endog, freqs)):
                    if i < j:
                        penalty += self.mmd(x_e1 @ self.phi, x_e2 @ self.phi, args.kernel_type) #* f

            # Normalize penalty
            error /= num_envs
            if num_envs > 1:
                penalty /= (num_envs * (num_envs - 1) / 2)

            # Optimize the combined loss
            opt.zero_grad()
            loss = error + gamma * penalty
            loss.backward()
            opt.step()

            if args.verbose and iteration % 100 == 0:
                w_str = pretty(self.solution())
                print(
                    "{:05d} | {:.5f} | {:.3f} | {:.3f} | {}".format(
                        iteration, gamma, error, penalty, w_str
                    ))

    def solution(self):
        """Returns the learned model parameters as a numpy array."""
        return (self.phi @ self.w).view(-1, 1).detach().numpy()

    def mmd(self, x, y, kernel_type):
        if kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
 
        elif kernel_type == "mean_cov":
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
        else:
            raise ValueError("Unknown kernel type")

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        return torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm).clamp_min_(1e-30)
        
    def predict(self, X): 
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        sol = torch.tensor(self.solution(), dtype=torch.float32)   # Flatten the solution
        pred = X @ sol
        rates = torch.exp(pred)
        return rates.detach().numpy()  # Apply exponential for Poisson



class EmpiricalRiskMinimizer(object):
    def __init__(self, exog, endog, freqs, args):
        # x is the covariates matrix
        # y is the depedent variable
        x_all = np.concatenate([e for e in exog])
        y_all = np.concatenate([e for e in endog])
        freq_all = np.concatenate([e for e in freqs])

        # w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.model = PoissonRegressor(max_iter=args.n_iterations, verbose=0)
        self.model.fit(x_all, y_all.ravel(), sample_weight=freq_all.ravel())
        w = self.model.coef_
        self.w = w.reshape(-1, 1)
    
    def predict(self, X): 
        res = self.model.predict(X)
        return res
    
    def solution(self):
        return self.w


class PoissonRegressorGLM:
    def __init__(self, exog, endo, freq):
        self.model = sm.GLM(
            endo, 
            exog, 
            family=sm.families.Poisson(), 
            freq_weights=freq,
            missing="drop"
        )

        self.result = self.model.fit(maxiter=300)
        self.coef_ = self.result.params
        self.predictions = self.result.predict(exog)
        
        print("Model fitting complete.")

    def get_coefficients(self): 
        '''return coefficients of the trained model'''
        return self.coef_

    def print_summary(self):
        print(self.result.summary())
        
    def predict(self, new_exog):
        '''predict for new data'''
        return self.result.predict(new_exog)
    



