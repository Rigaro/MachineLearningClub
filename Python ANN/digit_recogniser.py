# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:03:52 2020

@author: ricardog
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from DenseNeuralNetwork import DenseNeuralNetwork

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=4)

x_train_np = np.array(x_train).T
y_train_np = np.array(y_train).T
x_test_np = np.array(x_test).T
y_test_np = np.array(y_test).T

# Generate network
ns = [{'inputs':400, 'neurons':25, 'type':'relu'},
      #{'neurons':25, 'type':'sigmoid'},
      {'neurons':10, 'type':'sigmoid'}]
digit_dnn = DenseNeuralNetwork(ns, 'classification')

# Train
# Mini-batch
#costs, metrics = digit_dnn.train(x_train_np, y_train_np, learning_rate=0.1, epochs=100, optimiser='mini_batch', lambd=0.1, batch_size=64, evaluate=True, X_test=x_test_np, Y_test=y_test_np)
# Momentum
costs, metrics = digit_dnn.train(x_train_np, y_train_np, learning_rate=0.1, epochs=100, optimiser='momentum', lambd=0.1, batch_size=64, beta=0.9, evaluate=True, X_test=x_test_np, Y_test=y_test_np)

# Test set
digit_dnn.estimate(x_test_np)
test_accuracy = 100*digit_dnn.cost(y_test_np, 'accuracy')
print('Test accuracy: {:.1f}%'.format(test_accuracy))

# Plots
plt.figure(1)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.plot(metrics['accuracy']) # training
plt.plot(digit_dnn.eval_metrics['accuracy']) # test
plt.title('Network Accuracy')
plt.legend(['Training', 'Test'])

# Plots
plt.figure(2)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.plot(metrics['logloss']) # training
plt.plot(digit_dnn.eval_metrics['logloss']) # test
plt.title('Network Cost')
plt.legend(['Training', 'Test'])