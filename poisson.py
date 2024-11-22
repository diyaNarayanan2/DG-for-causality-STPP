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
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


# For each method, training and compilation of results is done upon initilization, only solution is returned on solution fucntion call

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

        x_val = exog[-1]
        y_val = endog[-1]
        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(exog[:-1], endog[:-1], freqs[:-1], args, reg=reg)
            err = torch.mean((torch.exp(x_val @ self.solution()) - y_val) ** 2).item()

            if args["verbose"]:
                print(f"IRM (reg={reg:.3f}) has validation error: {err:.3f}.")

            if err < self.best_err:
                self.best_err = err
                self.best_reg = reg
                self.best_phi = self.phi.clone()

        #self.phi = self.best_phi

    @staticmethod
    def irm_penalty(logits, y):
        """
        Compute the IRM penalty for Poisson regression.
        """
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
        loss1 = F.poisson_nll_loss(logits[::2] * scale, y[::2], reduction="mean", log_input=True)
        loss2 = F.poisson_nll_loss(logits[1::2] * scale, y[1::2], reduction="mean", log_input=True)
        #loss = F.poisson_nll_loss(logits * scale, y, reduction="mean", log_input=True)
        grad1 = torch.autograd.grad(loss1, [scale], create_graph=True)[0]
        grad2 = torch.autograd.grad(loss2, [scale], create_graph=True)[0]
        #return torch.sum(grad ** 2)
        return -torch.sum(grad1 * grad2)

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

        optimizer = torch.optim.Adam([self.phi, self.w], lr=args["lr"], weight_decay=1e-5)
        poisson_loss = nn.PoissonNLLLoss(log_input=True)

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0

            for x_e, y_e, f in zip(exog, endog, freqs):
                logits = x_e @ self.phi @ self.w
                error_e = poisson_loss(logits, y_e)
                penalty += self.irm_penalty(logits, y_e)
                #penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()
                error += error_e

            loss = (reg * error + (1 - reg) * penalty) * f
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args["verbose"] and iteration % 100 == 0:
                w_str = pretty(self.solution())
                print(
                    "{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(
                        iteration, reg, error, penalty, w_str
                    ))

    def solution(self):
        """
        Returns the learned model parameters.
        """
        return (self.phi @ self.w).view(-1, 1)
    
    def predict(self, X): 
        return X @ self.solution()


class MaximumMeanDiscrepancy(object):
    def __init__(self, endog, exog, freqs, args):
        best_reg = 0
        best_err = 1e6

        x_val = exog[-1]
        y_val = endog[-1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(exog[:-1], endog[:-1], freqs[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("MMD (reg={:.3f}) has {:.3f} validation error.".format(reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        # Save the best model parameters
        #self.phi = best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        # Initialize phi and weights
        self.phi = nn.Parameter(torch.empty(dim_x, dim_x))
        nn.init.xavier_uniform_(self.phi)

        self.w = torch.empty(dim_x, 1)
        nn.init.xavier_uniform_(self.w)
        self.w.requires_grad = True

        # Optimizer and loss function for Poisson regression
        opt = torch.optim.Adam([self.phi, self.w], lr=args["lr"], weight_decay=1e-5)
        loss = torch.nn.PoissonNLLLoss(log_input=True)

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0

            # Compute error and MMD penalty for each pair of environments
            for i, (x_e1, y_e1) in enumerate(environments):
                error += loss((x_e1 @ self.phi @ self.w), y_e1)

                for j, (x_e2, y_e2) in enumerate(environments):
                    if i < j:
                        penalty += self.mmd(x_e1, x_e2, args["kernel_type"])

            # Normalize penalty
            num_envs = len(environments)
            error /= num_envs
            if num_envs > 1:
                penalty /= (num_envs * (num_envs - 1) / 2)

            # Optimize the combined loss
            opt.zero_grad()
            (error + reg * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 100 == 0:
                w_str = pretty(self.solution())
                print(
                    "{:05d} | {:.5f} | {:.3f} | {:.3f} | {}".format(
                        iteration, reg, error, penalty, w_str
                    ))

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)

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
        return X @ self.soltuion()


class InvariantCausalPrediction(object):
    def __init__(self, environments, args):
        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []

        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)

            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(
            np.var(x, ddof=1) / np.var(y, ddof=1), x.shape[0] - 1, y.shape[0] - 1
        )

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients.view(-1, 1)


class EmpiricalRiskMinimizer(object):
    def __init__(self, exog, endog, freqs, args):
        # x is the covariates matrix
        # y is the depedent variable
        x_all = np.concatenate([e for e in exog])
        y_all = np.concatenate([e for e in endog])
        freq_all = np.concatenate([e for e in freqs])
        print(x_all.shape)
        print(y_all.shape)
        print(freq_all.shape)

        # w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.model = PoissonRegressor(max_iter=1000, verbose=0)
        self.model.fit(x_all, y_all.ravel(), sample_weight=freq_all.ravel())
        w = self.model.coef_
        self.w = torch.Tensor(w).view(-1, 1)
    
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
    



