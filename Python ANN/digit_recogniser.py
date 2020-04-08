# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:03:52 2020

@author: ricardog
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from NeuralNetwork import NeuralNetwork

# Extract data MNIST
#train_data = pd.read_csv('MNIST/train.csv')
#y_data = pd.get_dummies(train_data['label'])
#x_train = train_data.drop(columns=['label'])
#x_train = x_train.values.tolist()[0:4000]
#y_train = y_data.values.tolist()[0:4000]

# Load NIST data
x_table = pd.read_csv('NIST/nist_x_data.csv', header=None)
y_table = pd.read_csv('NIST/nist_y_data.csv', header=None)

# Create labels
y_labels = pd.get_dummies(y_table[0])
y_labels.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] # Fix 0 appearing as 10

# Normalise features
scaler = MinMaxScaler()
x_features = scaler.fit_transform(x_table)

# Create training data
x_data = x_features.tolist()
y_data = y_labels.values.tolist()

# Generate network
ns = [{'neurons':400, 'type':'sigmoid'},
      {'neurons':200, 'type':'sigmoid'},
      {'neurons':10, 'type':'sigmoid'}]
digit_ann = NeuralNetwork(ns)

# Train
digit_ann.train(x_train, y_train, 0.1, 3, batch_size=10, iterations=2)