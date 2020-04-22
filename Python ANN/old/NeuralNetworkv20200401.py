# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:28:32 2020

@author: ricardog
"""

from random import uniform
from random import randint
import numpy as np
import json

class NeuralNetwork:
    neuron_types = ['sigmoid', 'tanh', 'relu', 'leakyrelu']
    
    def __init__(self, input_num, hidden_layer_num, hidden_neuron_num, hidden_type, output_num, output_type):
        # Create the input layer by assigning random weights and biases
        self.input_layer = [self.create_neuron(input_num) for i in range(input_num)]
        
        # Create all the hidden layers with random weights and biases
        self.hidden_layers = []
        for hl in range(hidden_layer_num):
            if hl == 0: # First hidden layer
                self.hidden_layers.append([self.create_neuron(input_num) for i in range(hidden_neuron_num)])
            else: # All the other layers
                self.hidden_layers.append([self.create_neuron(hidden_neuron_num) for i in range(hidden_neuron_num)])
        
        # Create the output layer
        if hidden_layer_num > 0:
            self.output_layer = [self.create_neuron(hidden_neuron_num) for i in range(output_num)]
        else:
            self.output_layer = [self.create_neuron(input_num) for i in range(output_num)]
        
        # Create variable to hold computations.
        self.activations = [] # Neuron outputs
        self.potentials = [] # Weighted sums
        self.deltas = []
        self.gradient_w = []
        self.gradient_b = []
        
        # Define neuron types
        self.hidden_type = hidden_type.lower()
        self.output_type = output_type.lower()
        if self.hidden_type not in self.neuron_types:
            raise Exception('Hidden layer type {} not available.'.format(self.hidden_type))
        if self.output_type not in self.neuron_types:
            raise Exception('Output layer type {} not available.'.format(self.output_type))
        
    def create_neuron(self, conn_num):
        # Create a neuron with the given number of connections
        return {'weights': [uniform(-1, 1) for i in range(conn_num)],
                    'bias': uniform(-1, 1)}
    
    def neuron_potential(self, neuron, inputs):
        # Computes the total input (activation) of a neuron.
        # Make sure the data are in a numpy array.
        # Weights
        if type(neuron['weights']) is np.ndarray:
            w = neuron['weights']
        else:
            w = np.array(neuron['weights'])
        # Biases
        if type(neuron['bias']) is np.ndarray:
            b = neuron['bias']
        else:
            b = np.array(neuron['bias'])
        # Inputs
        if type(inputs) is np.ndarray:
            x = inputs
        else:
            x = np.array(inputs)
        return np.dot(w, x) + b
    
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
            return np.where(a >= 0, a, 0.01*a)  
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
        
        # Conver inputs to np.array
        if type(nn_inputs) is not np.ndarray:
            nn_inputs = np.array(nn_inputs)
        
        # Clear arrays
        self.activations = []
        self.potentials = []
        
        # Input layer
        self.activations.append(self.evaluate_layer(self.input_layer, nn_inputs, self.hidden_type))
        self.potentials.append(self.layer_potential(self.input_layer, nn_inputs))
        
        # Hidden layers: get each layer out and use the previous
        # layer output to calculate its output
        if len(self.hidden_layers) > 0:
            [self.activations.append(self.evaluate_layer(layer, self.activations[-1], self.hidden_type)) for layer in self.hidden_layers]
            [self.potentials.append(self.layer_potential(layer, self.potentials[-1])) for layer in self.hidden_layers]
        
        # Output layer
        self.activations.append(self.evaluate_layer(self.output_layer, self.activations[-1], self.output_type))    
        self.potentials.append(self.layer_potential(self.output_layer, self.potentials[-1]))      
        
        # Return only the activation of the last layer (i.e. the result)
        return self.activations[-1]
    
    
    def neuron_diff_transfer(self, activation_type, a):
        # Determines the neuron's differential output for back-prop
        # for a given activation input.
        
        # Conver inputs to np.array
        if type(a) is not np.ndarray:
            a = np.array(a)
            
        # Compute output depending on activation function type.
        # Sigmoid
        if activation_type == 'sigmoid':
            return (1/(1 + np.exp(-a)))*(1 - 1/(1 + np.exp(-a)))
        # Hyperbolic tangent
        elif activation_type == 'tanh':
            return 1 - (np.tanh(a)*np.tanh(a))
        # Rectified learning unit
        elif activation_type == 'relu':
            return np.where(a > 0, 1.0, 0.0)
        # Leaky rectified learning unit
        elif activation_type == 'leakyrelu':
            return np.where(a > 0, 1.0, 0.01)
        # Not available
        else:
            raise Exception('Activation function not available.')
            
    
    def back_propagation(self, nn_inputs, target_outputs):
        # Performs the backwards propagation algorithm with the given inputs
        # and target outputs to calculate errors and gradients in the network.
        
        # Conver parameters to np.array
        if type(nn_inputs) is not np.ndarray:
            nn_inputs = np.array(nn_inputs)
        if type(target_outputs) is not np.ndarray:
            target_outputs = np.array(target_outputs)
            
        # Initialise variables
        self.deltas = []
        self.gradient_w = []
        self.gradient_b = []
        
        ##################################
        # Step 1: Forward pass.
        # Perform forward propagation to obtain all the neuron
        # potentials and activations.
        self.forward_propagation(nn_inputs)
        
        ##################################
        # Step 2: Output layer.
        L = -1
        
        # Compute the error
        y_tilde = self.activations[L] - target_outputs
        self.deltas.append(np.multiply(y_tilde, self.neuron_diff_transfer(self.output_type, self.potentials[L])))
        
        # Calculate gradients for each neuron!
        w_grad = []
        for n in range(len(self.output_layer)):
            w_grad.append(self.activations[L-1] * self.deltas[L][n]) # grad^L = a^(L-1)*diff_sigma(error)
        
        w_grad = np.vstack(w_grad)    
        
        self.gradient_w.append(w_grad)
        self.gradient_b.append(self.deltas[L])
        
        ##################################
        # Step 3: Hidden layers
        # Compute the error and gradient for the hidden layers
        if len(self.hidden_layers) > 0:
            # Loop backwards through all layers
            for l in range(-2, -len(self.hidden_layers) - 2, -1):
                # create weights matrix
                if l == -2: # output layer case
                    weights = np.vstack([neuron['weights'] for neuron in self.output_layer])
                else:
                    weights = np.vstack([neuron['weights'] for neuron in self.hidden_layers[l+1]]) # Pick the right index backwards
                    
                # Calculate errors
                a_tilde = np.matmul(np.transpose(weights), self.deltas[l+1]) # Proportional error propagation
                self.deltas.insert(0, np.multiply(a_tilde, self.neuron_diff_transfer(self.hidden_type, self.potentials[l])))
                
                # Calculate gradients for each neuron!
                w_grad = []
                for n in range(len(self.hidden_layers[l+1])):
                    w_grad.append(self.activations[l-1] * self.deltas[l][n])
                
                w_grad = np.vstack(w_grad)
                
                self.gradient_w.insert(0, w_grad)
                self.gradient_b.insert(0, self.deltas[l])
                
        ##################################
        # Step 4: Input layer
        # Select the right weight matrix
        if len(self.hidden_layers) == 0: # Take output layer when no hidden
            weights = np.vstack([neuron['weights'] for neuron in self.output_layer])
        else: # The first of the hidden otherwise
            weights = np.vstack([neuron['weights'] for neuron in self.hidden_layers[0]]) 
          
        
        # Calculate errors
        a_tilde = np.matmul(np.transpose(weights), self.deltas[0]) # The first error
        self.deltas.insert(0, np.multiply(a_tilde, self.neuron_diff_transfer(self.hidden_type, self.potentials[0])))
                
        # Calculate gradients for each neuron!
        w_grad = []
        for n in range(len(self.input_layer)):
            w_grad.append(nn_inputs * self.deltas[0][n]) # grad^L = a^(L-1)*diff_sigma(error)
        
        w_grad = np.vstack(w_grad)
        
        self.gradient_w.insert(0, w_grad)
        self.gradient_b.insert(0, self.deltas[0])
        
        
    def update_network(self, learning_rate, nn_inputs, target_outputs):
        # Updates the network's weights and biases using gradient descent.
        
        ##################################
        # Step 1: Get gradients
        self.back_propagation(nn_inputs, target_outputs)
        
        
        ##################################
        # Step 2: Input layer
        # Extract the weights and biases from the layer
        weights = np.vstack([neuron['weights'] for neuron in self.input_layer])
        biases = np.array([neuron['bias'] for neuron in self.input_layer])
        
        # Update weights
        weights -= np.multiply(learning_rate,self.gradient_w[0])
        biases -= np.multiply(learning_rate,self.gradient_b[0])
        
        # Store weights and biases
        self.set_layer_weights(weights, self.input_layer)
        self.set_layer_biases(biases, self.input_layer)
        
        
        ##################################
        # Step 3: Hidden layers
        if len(self.hidden_layers) > 0:
            # Extract the weights and biases from each layer
            l = 1
            for layer in self.hidden_layers:
                weights = np.vstack([neuron['weights'] for neuron in layer])
                biases = np.array([neuron['bias'] for neuron in layer])
                
                # Update weights
                weights -= np.multiply(learning_rate,self.gradient_w[l])
                biases -= np.multiply(learning_rate,self.gradient_b[l])
                
                # Store weights and biases
                self.set_layer_weights(weights, layer)
                self.set_layer_biases(biases, layer)
                l = l+1
                
        
        ##################################
        # Step 2: Output layer
        # Extract the weights and biases from the layer
        weights = np.vstack([neuron['weights'] for neuron in self.output_layer])
        biases = np.array([neuron['bias'] for neuron in self.output_layer])
        
        # Update weights
        weights -= np.multiply(learning_rate,self.gradient_w[-1])
        biases -= np.multiply(learning_rate,self.gradient_b[-1])
        
        # Store weights and biases
        self.set_layer_weights(weights, self.output_layer)
        self.set_layer_biases(biases, self.output_layer)
        
        
    def set_layer_weights(self, weights, layer):
        # Sets the weights of the given layer.
        n = 0
        for neuron in layer:
            neuron['weights'] = weights[n]
            n = n + 1
                       
        
    def set_layer_biases(self, biases, layer):
        # Sets the weights of the given layer.
        n = 0
        for neuron in layer:
            neuron['bias'] = biases[n]
            n = n + 1
            
    def save_network(self, file_name, path=""):
        # Saves the network weights and configuration in a file.
                
        network = {'input_layer':self.input_layer,
                   'hidden_layers':self.hidden_layers,
                   'output_layer':self.output_layer,
                   'hidden_type':self.hidden_type,
                   'output_type':self.output_type}
        
        file_path = path + file_name + '.json'
        with open(file_path, 'w') as outfile:
            json.dump(network, outfile, indent = 4, sort_keys=True)
            
    def load_network(self, file_name, path=""):
        # Loads the network information from the given file
        
        # Read file and extract json data
        file_path = path + file_name + '.json'
        with open(file_path, 'r') as network_file:
            network_json = network_file.read()
        
        network_data = json.loads(network_json)
        
        self.input_layer = network_data['input_layer']
        self.hidden_layers = network_data['hidden_layers']
        self.output_layer = network_data['output_layer']
        self.hidden_type = network_data['hidden_type']
        self.output_type = network_data['output_type']
        
        
    def estimate(self, inputs):
        return self.forward_propagation(inputs)
    
    def train(self, inputs, targets, learning_rate, epochs, batch_size=1):
        # Trains a network with the given training data arrays, number
        # of epochs and batch size
        
        # Make sure the inputs and targets lists have the same lengths.
        if len(inputs) != len(targets):
            raise Exception('The inputs and targets lists must have the same length.')
        if batch_size < 1:
            raise Exception('The batch size should be greater than 0.')
            
        # Loop through epochs and batches
        for e in range(epochs):
            sqe = 0
            for b in range(batch_size):
                # Pick a random point from the dataset
                i = randint(0, len(inputs) - 1)
                x = inputs[i]
                y_d = targets[i]
                
                # Update network
                self.update_network(learning_rate, x, y_d)
                
                # Calculate sqe
                sqe = np.power(self.activations[-1] - y_d , 2)
                
        return sqe
        
        
        
        