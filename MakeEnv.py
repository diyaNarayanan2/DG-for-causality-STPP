#module to make environments, before data preprocessing 

import pandas as pd
import numpy as np

"""Make env function for hawkes process code 
envs will be made based on population density. as that is probability the greatest indicator of noise in the overall SCM
density --> 0 - 17k, need to make 5 bins of densities

main part to adjust is model fitting portion- 
glmtr-> covar_tr
glm_y-> q-> covid_tr
keep n_cty and n_day_tr equal for all env.n_day_tr is shape[1]
n_cty is shape[0], but env seperation will keep length of all env equal which is n_cty, choose a total length which is n_cty*5
can benchmark agaisnt 5th environment.
"""

class PopulationDensityEnvs:
    def __init__(self, demographic_path, mobility_path, covid_path, n_env, env_len):
        self.demographic_path = demographic_path
        self.mobility_path = mobility_path
        self.covid_path = covid_path
        self.n_env = n_env
        self.env_len = env_len

        # Call different functions to initialise
        self.group_indices = self.make_env()  
        self.demography_groups = self.extract_groups_from_dataset(self.demographic_path, self.group_indices)  
        self.covid_groups = self.extract_groups_from_dataset(self.covid_path, self.group_indices) 
        self.mobility_groups = self.extract_groups_from_mobility_extended()  
        
        print(f"{self.n_env} environments have been created, each with length {self.env_len}.")

        for e in range(self.n_env): 
            min_density = self.demography_groups[e].iloc[:, 3].min()  # pop. density is in fourth column
            max_density = self.demography_groups[e].iloc[:, 3].max()
            print(f"Environment {e+1}: population density ranges from {min_density} to {max_density}.")
            
            
    def make_env(self):
        '''function takes demography as reference and returns indices of diff envs 
        based on increasing population desnity data
        returns an array of sets of indices corresponding to each env'''
        demog_data = pd.read_csv(self.demographic_path)
        
        # total length to match n_env * env_len
        total_length = self.n_env * self.env_len
        demog_data_cropped = demog_data.iloc[:total_length]
        
        # sort by population density (second column), retain indexing
        demog_data_sorted = demog_data_cropped.sort_values(by=demog_data.columns[3]).reset_index()       
        # original indices for grouping
        original_indices = demog_data_sorted["index"].values
        # split fully sorted array of indices into n_env subarrays
        group_indices = np.array_split(original_indices, self.n_env)
        
        return group_indices

    def extract_groups_from_dataset(self, data_path, group_indices):
        '''takes data path as input, long with group indices. 
        returns a list of data_path data extracted based on indices'''
        data = pd.read_csv(data_path)
        grouped_data = [data.iloc[group] for group in group_indices]
        
        return grouped_data

    def extract_groups_from_mobility_extended(self):
        '''adjusted for mibility since, each index in group indices corresponds to 6x rows in mobility'''
        data = pd.read_csv(self.mobility_path)
        extended_grouped_data = []
        
        for group in self.group_indices:
            extended_rows = []
            for index in group:
                # Extract the 6 subtypes corresponding to the FIPS position
                extended_rows.append(data.iloc[index * 6 : (index + 1) * 6])
            
            # Concatenate 6-row blocks and crop to match total length
            extended_grouped_data.append(pd.concat(extended_rows))
        
        # crop groups of 6*env_len
        cropped_data = [group.iloc[: self.env_len * 6] for group in extended_grouped_data]
        
        return cropped_data
