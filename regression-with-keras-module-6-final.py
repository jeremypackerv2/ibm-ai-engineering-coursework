# Author: Jeremy Packer
# Date: December 23, 2024
# Description: Regression with Keras to determine concrete strength.
# Completed for Module 6 assignment built using local and converted to jupyter.

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', FutureWarning)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# split the data into train and test sets and use my own random state to play with test
X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=33)

# Part A Requirement
def baseline_model():
    # create models
    model = Sequential()
    model.add(Input(shape=(n_cols,)))

    # create model that has one hidden layer of 10 nodes
    model.add(Dense(10, activation='relu'))

    # Output layer for regression
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# prepare for 50 loops for 50 mean squared errors inn array
mse_list = []
for i in range(50):
    # build the model
    model = baseline_model()

    # fit the model with 50 epochs
    model.fit(X_train, y_train, validation_split=0.3, epochs=50, verbose=2)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Compute MSE for this run
    mse = mean_squared_error(y_test, predictions)
    mse_list.append(mse)

    print(f"Run {i+1} - MSE: {mse}")

# Calculate mean and standard deviation of the MSE results
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

# Print results
print(f"Mean MSE over 50 runs: {mean_mse}")
print(f"Standard Deviation of MSE: {std_mse}")