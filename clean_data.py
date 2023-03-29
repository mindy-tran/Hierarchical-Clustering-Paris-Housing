import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# read in the data
features_data = pd.read_csv("ParisHousing.csv")

# Feature selection for the model
features_data = features_data[['squareMeters','price']]

print('Sum of Null values:')
print(features_data.isnull().sum())
print(np.shape(features_data))

features_data.to_csv('cleaned_data.csv')