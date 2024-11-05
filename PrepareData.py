#Module to preprocess data

import numpy as np
import pandas as pd 
import warnings
import torch
from datetime import datetime 
START_DAYS = 50

def MakeEnvironments():
    '''function to transform input data into environments
    will use one of the feature values to sort the data into groups or environments
    the sorting will be stored in the form of index sets, with the index values from the original data ordering'''
   

def PreProcessCovariates(report, mobility, demography, criterion, days, delta, verbose): 
    '''takes dfs as inputs(can be used for extracted environents)
    along with lambda function as criterion in input and returns four dfs
    1. full exogneous variables, i.e. all covariates
    2. corresponding endogenous variables, i.e. case count 
    3. df of dates corresponding to case count data 
    4. Keylist df of FIPS, State, County with the same ordering as data '''
    
    if delta == "":
        print("No shift parameter for mobility provided. It will set to zero ... ")
        mobiShift_in = 0
    else:
        mobiShift_in = int(delta)
    
    report_data = report.iloc[:,3+START_DAYS:3+START_DAYS+days].values 
    report_data[np.isnan(report_data)] = 0
    report_data[report_data<0] = 0
    n_county, n_day = report_data.shape
    county_keylist = report.iloc[:, :3]
    date_keylist = report.columns[3+START_DAYS:3+START_DAYS+days]
    case_count = np.hstack([np.zeros((report_data.shape[0], 1)), np.diff(report_data, axis=1)])
    
    mobility_data = mobility.iloc[:, 4+START_DAYS:4+START_DAYS+days].values     #mobility will have 6x rows
    mobility_data[np.isnan(mobility_data)] = 0
    mobility_data[mobility_data<0] = 0
    
    date_list = [datetime.strptime(date_str[1:].replace("_", "-"), "%Y-%m-%d")
            for date_str in date_keylist]   #output list of dates in datetime format
        
    n_mobitype = mobility_data.shape[0] // n_county
    
    for _ in range(mobiShift_in):
        mobility_data = np.hstack([np.mean(mobility_data[:, :7], axis=1, keepdims= True), mobility_data])
    mobility_data = mobility_data[:, :n_day]
    mob_reshape = mobility_data.reshape(n_mobitype, -1).T   #pad mobility data then crop then reshape
        
    demography = demography.iloc[:, 3:]
    demo_reshape = np.tile(demography, (n_day, 1))
    covariates = np.hstack([mob_reshape, demo_reshape])
    
    covar_mean = np.mean(covariates, axis=0)
    covar_std = np.std(covariates, axis=0)
    
    covariates = (covariates - covar_mean) / covar_std  #normalising covariate values , alternatively can use min-max or standard scaler
    
    if verbose:
        print(f"There are {n_county} counties and {n_day} days in the covid report, with {n_mobitype} mobility indices.")
        print(f"Training set: covariates has shape {covariates.shape} spanning {covariates.min()} to {covariates.max()}")
        
    return covariates, case_count, date_list, county_keylist 

