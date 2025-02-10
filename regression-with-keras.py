# Author: Jeremy Packer
# Date: December 19, 2024
# Description: Regression with Keras to determine concrete strength.
# Completed for Module 3 assignment.

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', FutureWarning)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing

# confirm machine handles this venv properly or if we need to switch
print('----')
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("TensorFlow version:", tf.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print('----')

# load the dataset and read into a pandas dataframe
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

# Returns the first row
print('First Row:')
print(concrete_data.head(1))
print('----')

# Head of the data returns first 5 rows by default
print('First Five Rows:')
print(concrete_data.head())
print('----')

# Check the shape of the DataFrame (number of rows and columns)
print('Data Shape: Rows , Columns')
print(concrete_data.shape)
print('----')

# Take a look at the descriptive statistics
print('Descriptive Statistics:')
print(concrete_data.describe())
print('----')

# Check for null values
print('Null Values:')
print(concrete_data.isnull().sum())
print('----')

# Lets split data into predictors and target
concrete_data_columns = concrete_data.columns

# This returns all columns except 'Strength'
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]

# This returns only 'Strength'
target = concrete_data['Strength'] 

# adding sanity check
print('Predictors Head:')
print(predictors.head())

print('Target Head:')
print(target.head())

# normalize the data by subtracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print('Predictors Normalized Head:') 
print(predictors_norm.head())
print('----')

# add the target column into the normalized dataframe
predictors_norm['Strength'] = target

# check for null values should be 
print('Null Values:')
print(predictors_norm.isnull().sum())
print('----')

# les check the number of predictors
n_cols = predictors_norm.shape[1]
print('Number of Predictors:')
print(n_cols)
print('----')

# checking strngth column
print('Strength Column:')
print(predictors_norm['Strength'])
print('----')

# define regression model
def regression_model():
    # create models
    model = Sequential()
    model.add(Input(shape=(n_cols,)))

    # create model that has two hidden layers of size 50 units
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))

    # Output layer for regression
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# define regression model with increased neurons
def regression_model_with_increased_neurons():
    # create models
    model = Sequential()
    model.add(Input(shape=(n_cols,)))

    # create model that has two hidden layers of size 100 units
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    # Output layer for regression
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# define regression model with decreased neurons
def regression_model_with_decreased_neurons():
    # create models
    model = Sequential()
    model.add(Input(shape=(n_cols,)))

    # create model that has two hidden layers of size 25 units
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))

    # Output layer for regression
    model.add(Dense(1))

    # compile models
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model    

# define regression model with more layers
def regression_model_more_layers():
    # Create model with more hidden layers
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # Additional two hidden layer
    model.add(Dense(50, activation='relu'))  
    model.add(Dense(50, activation='relu'))
    # Output layer for regression
    model.add(Dense(1))  

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# train and test the different regression models with configured parameters
# model = regression_model()
model = regression_model_with_increased_neurons()
# model = regression_model_with_decreased_neurons()
# model = regression_model_more_layers()

# fit the model with standard 100 epochs
# model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# fit the model with 500 epochs
model.fit(predictors_norm, target, validation_split=0.3, epochs=500, verbose=2)