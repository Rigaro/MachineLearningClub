# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:28:32 2020

@author: ricardog
"""

from random import uniform
from random import randint
import numpy as np
import json

# Easy metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    neuron_types = ['sigmoid', 'tanh', 'relu', 'leakyrelu']
    
    def __init__(self, network_structure, file_name="", load_path=""):
        # Creates a neural network with the given network structure
        # The network structure is defined as a list with dictionaries, where
        # each dictionary represents a layer structure as follows:
        # 'neurons': The number of neurons in this layer.
        # 'type': the type of activation function
        # If the file_name is provided, it will load the weights
        # and biases from the given file, and load_path if given.
        
        # Set structure
        self.network_structure = network_structure
        
        # Initialise variables
        self.layers = []
        self.potentials = []
        self.activations = []
        self.deltas = []
        self.w_grads = []
        self.b_grads = []
        
        # Create network
        l = 0
        for layer in network_structure:
            if file_name == "": # Create random
                weights = []
                biases = []
                # Loop through all the requested neurons in the layer
                for n in range(layer['neurons']):
                    biases.append(uniform(-1, 1)) # There are as many biases as there are neurons in the layer
                    if l == 0 : # First layer has as many connections as neurons
                        weights.append([uniform(-1, 1) for i in range(layer['neurons'])])
                    else: # All the other layers have as many connections as the previous layer (i.e. the last to be added)
                        weights.append([uniform(-1, 1) for i in range(len(self.layers[-1]['biases']))])
                
                # Add weights and biases to layer
                self.layers.append({'weights':weights, 'biases':biases, 'type':layer['type']})
                
            else: # Load from file
                self.load_network(file_name, load_path)
                
            l += 1
    
    def layer_potentials(self, layer, inputs):
        # Computes the potentials of a given layer
        
        # Conver inputs to np.array
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
            
        # Extract the weights and biases from the layer
        weights = np.array(layer['weights'])
        biases = np.array(layer['biases'])
        z = np.dot(weights, inputs) + biases
        return z.tolist()
        
    
    def layer_transfer(self, z, activation_type):
        # Determines the neuron's activation (output)
        # for a given potential input.
        
        # Conver inputs to np.array
        if type(z) is not np.ndarray:
            z = np.array(z)
            
        # Compute output depending on activation function type.
        # Sigmoid
        if activation_type == 'sigmoid':
            z = 1/(1 + np.exp(-z))
        # Hyperbolic tangent
        elif activation_type == 'tanh':
            z = np.tanh(z)
        # Rectified learning unit
        elif activation_type == 'relu':
            z = np.maximum(0,z)
        # Leaky rectified learning unit
        elif activation_type == 'leakyrelu':
            z = np.where(z >= 0, z, 0.01*z)  
        # Not available
        else:
            raise Exception('Activation function not available.')     
        
        return z.tolist()
                            
    
    def forward_propagation(self, nn_inputs):
        # Compute the output for each layer
        
        # Conver inputs to np.array
        if type(nn_inputs) is not np.ndarray:
            nn_inputs = np.array(nn_inputs)
        
        # Clear arrays
        self.activations = []
        self.potentials = []
        
        # Add inputs to activations as these are the activations for the first layer.
        self.activations.append(nn_inputs)
        
        # Loop through all layers calculating activations and potentials
        for layer in self.layers:
            self.potentials.append(self.layer_potentials(layer, self.activations[-1]))
            self.activations.append(self.layer_transfer(self.potentials[-1], layer['type']))
        
        # Return only the activation of the last layer (i.e. the result)
        return self.activations[-1]
    
    
    def layer_diff_transfer(self, z, activation_type):
        # Determines the neuron's differential activation (output)
        # for a given potential input.
        
        # Conver inputs to np.array
        if type(z) is not np.ndarray:
            z = np.array(z)
            
        # Compute output depending on activation function type.
        # Sigmoid
        if activation_type == 'sigmoid':
            z = (1/(1 + np.exp(-z)))*(1 - 1/(1 + np.exp(-z)))
        # Hyperbolic tangent
        elif activation_type == 'tanh':
            z = 1 - (np.tanh(z)*np.tanh(z))
        # Rectified learning unit
        elif activation_type == 'relu':
            z = np.where(z > 0, 1.0, 0.0)
        # Leaky rectified learning unit
        elif activation_type == 'leakyrelu':
            z = np.where(z > 0, 1.0, 0.01)
        # Not available
        else:
            raise Exception('Activation function not available.')
            
        return z.tolist()
    
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
        self.w_grads = []
        self.b_grads = []
        
        ##################################
        # Step 1: Forward pass.
        # Perform forward propagation to obtain all the neuron
        # potentials and activations.
        #
        self.forward_propagation(nn_inputs)
        
        ##################################
        # Step 2: Loop backwards through all layers to calculate deltas.
        # Then calculate gradients and add to list.        
        # 
        for l in range(-1, -len(self.layers)-1, -1):
            # Output layer
            if l == -1:
                # Compute the error
                y_tilde = np.array(self.activations[l]) - target_outputs
            # All other layers
            else:
                # Compute the proportional error
                weights = np.array(self.layers[l+1]['weights'])
                y_tilde = np.dot(weights.T, np.array(self.deltas[l+1]))
            
            # Compute delta and append
            delta = np.multiply(y_tilde,
                                np.array(self.layer_diff_transfer(np.array(self.potentials[l]), self.layers[l]['type'])))
            self.deltas.insert(0, delta.tolist())
            
            # Calculate weight gradient
            w_grad = np.outer(delta[np.newaxis], np.array(self.activations[l-1])[np.newaxis].T)
            self.w_grads.insert(0, w_grad.tolist())
            
            # Add bias gradients
            self.b_grads.insert(0, delta.tolist())
        
        
    def update_network(self, learning_rate, nn_inputs, target_outputs):
        # Updates the network's weights and biases using gradient descent.
        
        ##################################
        # Step 1: Get gradients
        self.back_propagation(nn_inputs, target_outputs)
        
        
        ##################################
        # Step 2: Update weights and biases in all layers
        #
        l = 0
        for layer in self.layers:
            # Get weights and biases
            weights = np.array(layer['weights'])
            biases = np.array(layer['biases'])
            # Update
            weights -= learning_rate*np.array(self.w_grads[l])
            biases -= learning_rate*np.array(self.b_grads[l])
            # Set
            layer['weights'] = weights.tolist()
            layer['biases'] = biases.tolist()
            l += 1       
            
            
    def save_network(self, file_name, path=""):
        # Saves the network weights and configuration in a file.
                
        network = {'structure':self.network_structure,
                   'layers':self.layers,}
        
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
        
        self.network_structure = network_data['structure']
        self.layers = network_data['layers']
        
        
    def estimate(self, inputs):
        return self.forward_propagation(inputs)
    
    def train(self, inputs, targets, learning_rate, epochs, batch_size=1, iterations=1):
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
            # Generate batch
            data_size = len(inputs)
            num_batches = int(data_size/batch_size)
            # Loop through the batches
            for b in range(num_batches):
                sqe = []
                ls = []
                acc = []
                # Pick the indeces of the training samples randomly
                batch_indeces = [randint(0, len(inputs)-1) for n in range(batch_size)]
                # Repeat for as many iterations
                for i in range(iterations):
                    # Loop through all the indeces in the batch
                    for d in range(len(batch_indeces)):
                        # Pick a random point from the dataset
                        x = inputs[batch_indeces[d]]
                        y_d = targets[batch_indeces[d]]
                        
                        # Update network
                        self.update_network(learning_rate, x, y_d)
                        
                        # Get predicted values
                        y_p = np.array(self.activations[-1])
                        
                        # Calculate metrics
                        sqe.append(np.mean(np.power(y_p - y_d , 2))) # Mean square error
                        ls.append(log_loss(y_d, y_p)) # Log-loss
                        acc.append(accuracy_score(y_d, y_p.round())) # Accuracy
                
                # Get the mean of the metrics
                msqe = np.mean(sqe)
                mls = np.mean(ls)
                macc = 100*np.mean(acc)
                
                # Show statistics
                print("Epoch {}/{}. Batch {}/{}. MSQE: {:.5f}, Log-Loss: {:.5f}, Accuracy: {:.1f}%".format(e+1, epochs, b+1, num_batches, msqe, mls, macc))
                
        return sqe
        