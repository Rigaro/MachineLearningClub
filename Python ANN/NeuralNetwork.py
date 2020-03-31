# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:28:32 2020

@author: ricardog
"""

from random import uniform
import numpy as np

class NeuralNetwork:
    neuron_types = ['sigmoid', 'tanh', 'relu', 'leakyrelu']
    
    def __init__(self, input_num, layer_num, hidden_neuron_num, hidden_type, output_num, output_type):
        # Create the input layer by assigning random weights and biases
        self.input_layer = [self.create_neuron(input_num) for i in range(input_num)]
        
        # Create all the hidden layers with random weights and biases
        self.hidden_layers = []
        for hl in range(layer_num):
            if hl == 0: # First hidden layer
                self.hidden_layers.append([self.create_neuron(input_num) for i in range(hidden_neuron_num)])
            else: # All the other layers
                self.hidden_layers.append([self.create_neuron(hidden_neuron_num) for i in range(hidden_neuron_num)])
        
        # Create the output layer
        if layer_num > 0:
            self.output_layer = [self.create_neuron(hidden_neuron_num) for i in range(output_num)]
        else:
            self.output_layer = [self.create_neuron(input_num) for i in range(output_num)]
        
        # Create layer output array and layer potentials
        self.outputs = []
        self.potentials = []
        
        # Define neuron types
        self.hidden_type = hidden_type.lower()
        self.output_type = output_type.lower()
        if self.hidden_type not in self.neuron_types:
            raise Exception('Hidden layer type {} not available.'.format(self.hidden_type))
        if self.output_type not in self.neuron_types:
            raise Exception('Output layer type {} not available.'.format(self.output_type))
        
    def create_neuron(self, conn_num):
        # Create a neuron with the given number of connections
        return {'weights': np.array([uniform(-1, 1) for i in range(conn_num)]),
                    'bias': uniform(-1, 1)}
    
    def neuron_potential(self, neuron, inputs):
        # Computes the total input (activation) of a neuron.
        # Make sure the data are in a numpy array.
        # Weights
        if type(inputs) is np.ndarray:
            w = neuron['weights']
        else:
            w = np.array(neuron['weights'])
        # Biases
        if type(inputs) is np.ndarray:
            b = neuron['bias']
        else:
            b = np.array(neuron['bias'])
        # Inputs
        if type(inputs) is np.ndarray:
            x = inputs
        else:
            x = np.array(inputs)
        return activation_type, np.dot(w, x) + b
    
    def layer_potential(self, layer, inputs):
        # Computes the activation of a given layer
        
        # Conver inputs to np.array
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
            
        # Extract the weights and biases from the layer
        weights = np.vstack([neuron['weights'] for neuron in layer])
        biases = np.array([neuron['bias'] for neuron in layer])
        a = np.dot(weights, inputs) + biases
        return a
        
    
    def neuron_transfer(self, activation_type, a):
        # Determines the neuron's output (action potential)
        # for a given activation input.
        
        # Conver inputs to np.array
        if type(a) is not np.ndarray:
            a = np.array(a)
            
        # Compute output depending on activation function type.
        # Sigmoid
        if activation_type == 'sigmoid':
            return 1/(1 + np.exp(-a))
        # Hyperbolic tangent
        elif activation_type == 'tanh':
            return np.tanh(a)
        # Rectified learning unit
        elif activation_type == 'relu':
            return np.maximum(0,a)
        # Leaky rectified learning unit
        elif activation_type == 'leakyrelu':
            if a < 0:
                return 0.01*a
            
            else:
                return a
        # Not available
        else:
            raise Exception('Activation function not available.')     
            
    
    def evaluate_layer(self, layer, inputs, activation_type):
        # Evaluates the given layer with the given inputs and activation function
        return self.neuron_transfer(activation_type, self.layer_potential(layer, inputs))
    
    def evaluate_neuron(self, neuron, inputs, activation_type):
        # Evaluates the given neuron with the given inputs and activation function
        return self.neuron_transfer(activation_type, self.neuron_potential(neuron, inputs))
            
    
    def forward_propagation(self, nn_inputs):
        # Compute the output for each layer
        
        # Clear arrays
        self.outputs = []
        self.potentials = []
        
        # Input layer
        self.outputs.append(self.evaluate_layer(self.input_layer, nn_inputs, self.hidden_type))
        self.outputs.append(self.layer_potential(self.input_layer, nn_inputs))
        
        # Hidden layers: get each layer out and use the previous
        # layer output to calculate its output
        if len(self.hidden_layers) > 0:
            [self.outputs.append(self.evaluate_layer(layer, self.outputs[-1], self.hidden_type)) for layer in self.hidden_layers]
        
        # Output layer
        self.outputs.append(self.evaluate_layer(self.output_layer, self.outputs[-1], self.output_type))       
        
        # Return only the activation of the last layer (i.e. the result)
        return self.outputs[-1]
    
    
    def neuron_diff_transfer(self, activation_type, a):
        # Determines the neuron's differential output for back-prop
        # for a given activation input.
        
        # Conver inputs to np.array
        if type(a) is not np.ndarray:
            a = np.array(a)
            
        # Compute output depending on activation function type.
        # Sigmoid
        if activation_type == 'sigmoid':
            return np.exp(a)/((1 + np.exp(a))*(1 + np.exp(a)))
        # Hyperbolic tangent
        elif activation_type == 'tanh':
            return 1 - (np.tanh(a)*np.tanh(a))
        # Rectified learning unit
        elif activation_type == 'relu':
            if a > 0:
                return 1.0
            else:
                return 0.0
        # Leaky rectified learning unit
        elif activation_type == 'leakyrelu':
            if a < 0:
                return 0.01
            elif a == 0:
                return 0.0
            else:
                return 1
        # Not available
        else:
            raise Exception('Activation function not available.')
    
    