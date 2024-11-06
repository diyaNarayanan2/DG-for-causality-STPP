#module to make environments, before data preprocessing 

import pandas as pd
import numpy as np

"""Make env function for hawkes process code 
envs will be made based on population density. as that is probability the greatest indicator of noise in the overall SCM
density --> 0 - 17k, need to make 5 bins of densities
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
        
        print("envrionemnts have been created based on popluation density (equal sizes)")
        print(f"{self.n_env} environments, each with length {self.env_len}.")

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


class StateGroupedEnvs:
    def __init__(self, demographic_path, mobility_path, covid_path):
        self.demographic_path = demographic_path
        self.mobility_path = mobility_path
        self.covid_path = covid_path
        self.state_groups = []

        self.group_indices, self.state_groups = self.make_state_groups()  
        self.state_groups = list(self.state_groups.keys())
        self.demography_groups = self.extract_groups_from_dataset(self.demographic_path, self.group_indices)  
        self.covid_groups = self.extract_groups_from_dataset(self.covid_path, self.group_indices)  
        self.mobility_groups = self.extract_groups_from_mobility_extended()
        
        print("Envrionments have been created state wise (different sizes)")
        print(f"Total number of states: {len(self.group_indices)}")
        for i, group in enumerate(self.group_indices):
            print(f"State {i+1}: {self.state_groups[i]} : {len(group)} counties")

    def make_state_groups(self):
        '''Create groups based on state. all counties in one state in one environment'''
        demog_data = pd.read_csv(self.demographic_path)

        # group indices by unique states (second column: state)
        state_groups = demog_data.groupby(demog_data.columns[1]).indices  #dictionary of indices by state

        # convert dictionary into list of arrays (which have the indices)
        group_indices = [state_groups[state] for state in state_groups]
        
        return group_indices, state_groups

    def extract_groups_from_dataset(self, data_path, group_indices):
        data = pd.read_csv(data_path)
        grouped_data = [data.iloc[group] for group in group_indices]
        
        return grouped_data

    def extract_groups_from_mobility_extended(self):
        data = pd.read_csv(self.mobility_path)
        extended_grouped_data = []
        
        for group in self.group_indices:
            extended_rows = []
            for index in group:
                extended_rows.append(data.iloc[index * 6 : (index + 1) * 6])
            extended_grouped_data.append(pd.concat(extended_rows))
        
        return extended_grouped_data
    
class RegionGroupedEnvs:
    def __init__(self, demographic_path, mobility_path, covid_path, regions_path):
        self.demographic_path = demographic_path
        self.mobility_path = mobility_path
        self.covid_path = covid_path
        self.regions_path = regions_path
        self.region_groups = []

        self.group_indices, self.region_groups = self.make_region_groups()  
        
        self.demography_groups = self.extract_groups_from_dataset(self.demographic_path, self.group_indices)
        self.covid_groups = self.extract_groups_from_dataset(self.covid_path, self.group_indices)
        self.mobility_groups = self.extract_groups_from_mobility_extended()

        print("Environments have been created region-wise (different sizes)")
        print(f"Total number of regions: {len(self.group_indices)}")
        for i, group in enumerate(self.group_indices):
            print(f"Region {i+1}: {self.region_groups[i]} : {len(group)} counties")

    def make_region_groups(self):
        '''Create groups based on regions. All counties in the same region in one environment.'''
        demog_data = pd.read_csv(self.demographic_path)
        region_data = pd.read_csv(self.regions_path)

        # Merge demographic data with region data on the state column (second column)
        demog_with_regions = demog_data.merge(region_data, left_on=demog_data.columns[1], right_on=region_data.columns[0])

        region_groups = demog_with_regions.groupby(region_data.columns[1]).indices  # Dictionary of indices by region

        group_indices = [region_groups[region] for region in region_groups]
        
        return group_indices, list(region_groups.keys())

    def extract_groups_from_dataset(self, data_path, group_indices):
        data = pd.read_csv(data_path)
        grouped_data = [data.iloc[group] for group in group_indices]
        
        return grouped_data
    
    def extract_groups_from_mobility_extended(self):
        data = pd.read_csv(self.mobility_path)
        extended_grouped_data = []
        
        for group in self.group_indices:
            extended_rows = []
            for index in group:
                extended_rows.append(data.iloc[index * 6 : (index + 1) * 6])
            extended_grouped_data.append(pd.concat(extended_rows))
        
        return extended_grouped_data
