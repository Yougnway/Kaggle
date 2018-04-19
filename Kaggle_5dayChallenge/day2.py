# modules we'll use

import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

# read in our data
earthquakes = pd.read_csv("./earthquake-database/database.csv")
landslides = pd.read_csv("./landslide-events/catalog.csv")
volcanos = pd.read_csv("./volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)

print(landslides['date'].head())
print(landslides['date'].dtype)
# print(earthquakes['Date'].head())
# print(earthquakes['Date'].dtype)
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
print(landslides['date_parsed'].head())
# earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m-%d-%Y")
# print(earthquakes['Date_parsed'])
# print(landslides['date_parsed'].head())
day_of_month_landslides = landslides['date_parsed'].dt.day
# print(day_of_month_landslides.isnull().sum())
day_of_month_landslides = day_of_month_landslides.dropna()
sns.distplot(day_of_month_landslides,kde=False, bins=31)
plt.show()