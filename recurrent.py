#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:43:48 2018

@author: kishankumar
"""

# Recurrent Neural Network

# part-1 Data Preprocessing
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the training dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
'''
we'll be using the normalization method to get the feature scaling
xnorm = x - xmin/(xmax-xmin)

'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
# feature_range actually means that the value will be between 0 to 1
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timestamps and 1 output
X_train = []
y_train = []
# basically we need to create a input data in a particular format so that it can be fed to the model

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    # basically we are taking all the datas from 0 to 60 and appending that to one row
    y_train.append(training_set_scaled[i, 0])
    # y contains the next data that the recurrent network has to predict
    
    '''
    so we will train our data on past 60 values and by interpreting 
    the data it should output the next trend which is stord in the y 
    and we will check it whethere it is matching or not and try to minimize the error
    '''
    
X_train, y_train = np.array(X_train), np.array(y_train)
# it is required because the model only accepts numpy array so we need to convert it into the numpy array

# reshaping
# refer to keras documentation to know what kind of data the recurrent network take as their input
#(batch_size, number_of_timestamp, indicator)
# here indicator is only 1 i.r google opening and timestamp is 60
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# part-2 Building the rnn

# importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# initialising the RNN
regressor = Sequential()
# adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# adding the output layer
regressor.add(Dense(units = 1))

# compile the rnn
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting thr rnn to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# part-3 Predicting the result and visualizing it

# getting the real google stock price
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# getting the predicted google stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)


X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_google_stock_price = regressor.predict(X_test)
predicted_google_stock_price = sc.inverse_transform(predicted_google_stock_price)

# visualising the result

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_google_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock price')
plt.legend()
plt.show()



