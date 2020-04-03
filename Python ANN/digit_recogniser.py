# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:03:52 2020

@author: ricardog
"""

import pandas as pd
from NeuralNetwork import NeuralNetwork

# Extract data
train_data = pd.read_csv('MNIST/train.csv')
y_data = pd.get_dummies(train_data['label'])
x_train = train_data.drop(columns=['label'])
x_train = x_train.values.tolist()[0:4000]
y_train = y_data.values.tolist()[0:4000]

# Generate network
ns = [{'neurons':784, 'type':'sigmoid'},
      {'neurons':700, 'type':'sigmoid'},
      {'neurons':10, 'type':'sigmoid'}]
digit_ann = NeuralNetwork(ns)

# Train
digit_ann.train(x_train, y_train, 0.1, 100)