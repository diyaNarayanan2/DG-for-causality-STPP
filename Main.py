import numpy as np 
import torch
import warnings
import pandas as pd
from PrepareData import PreProcessCovariates

'''MAKE ALL COVARIATE LABELS AS GLOBAL VARIABLES
turn these input data into args for main argparser'''

REPORT_PATH = r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Report_Test.csv"
DEMOGRAPHY_PATH = r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Demo_Dconfirmed.csv"
MOBILITY_PATH = r"C:\Users\dipiy\OneDrive\Documents\GitHub\DG-for-causality-STPP\TestData\Mobility_Test.csv"
DELTA = 4
CRITERION = lambda row: row['State'] == 'Maine'
MOBITYPES = ['Grocery / pharmacy', 'Parks', 'Residential', 'Retail / recreation', 'Transit stations', 'Workplace']
DRY_CORRECT = 14
EMITR = 10
BREAK_DIFF = 10^-3
VERBOSE = 1

#ASSUMING TRAINING WITH ALL DATA
#Need to use additional dry-correct days of mobility values to account for reporting delay



report = pd.read_csv(REPORT_PATH)
demography = pd.read_csv(DEMOGRAPHY_PATH)
mobility = pd.read_csv(MOBILITY_PATH)

report = report[report.apply(CRITERION, axis=1)]    #apply criterion to filter out rows
demography = demography[demography.apply(CRITERION, axis=1)]
mobility = mobility[mobility.apply(CRITERION, axis=1)]

days=10
delta = 3

cov, cases, dates, key = PreProcessCovariates(REPORT_PATH, MOBILITY_PATH, DEMOGRAPHY_PATH, CRITERION, 10, DELTA, VERBOSE)

