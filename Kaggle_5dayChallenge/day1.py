"""
    Handling missing values
    author: @Youngway
    26-03-2018
"""

import numpy as np
import pandas as pd

# read in the data
nfl_data = pd.read_csv("NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("Building_Permits.csv")

nfl_sample = nfl_data.sample(5)
missing_value_count = nfl_data.isnull().sum()
# print(missing_value_count[0:10])
total_cells = np.prod(nfl_data.shape)
total_missing = missing_value_count.sum()
miss_percentage = total_missing/total_cells * 100
# print(miss_percentage)
columns_with_na_dropped = nfl_data.dropna(axis=1)
# print(columns_with_na_dropped.head())
# print("Column in original dataset: %d \n" % nfl_data.shape[1])
# print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])

columns_with_na_dropped_sf = sf_permits.dropna(axis=1)
# print("Columns in original dataset sf_permit: %d \n" % sf_permits.shape[1])
# print("Columns with na's dropped in sf_permit: %d" % columns_with_na_dropped_sf.shape[1])

subset_nfl_data = nfl_data.loc[:,'EPA':'Season'].head()
print(subset_nfl_data)
print(subset_nfl_data.fillna(method = 'bfill', axis = 0).fillna("0"))

revised_sf_permits = sf_permits.fillna(method='bfill', axis = 0).fillna("0")
