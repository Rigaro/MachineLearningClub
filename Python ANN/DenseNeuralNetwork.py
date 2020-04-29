# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:28:32 2020

@author: ricardog
"""

# Useful libraries
import numpy as np
import json

# Easy metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

# Timing stuff
import time

class DenseNeuralNetwork:
    neuron_types = ['sigmoid', 'tanh', 'relu', 'leakyrelu']
    network_types = ['regression', 'classification']
    optimisers = ['gradient_descent', 'mini_batch', 'momentum']
    
    def __init__(self, network_structure, network_type, file_name="", load_path=""):
        # Creates a neural network with the given network structure
        # The network structure is defined as a list with dictionaries, where
        # each dictionary represents a layer structure as follows:
        # 'neurons': The number of neurons in this layer.
        # 'type': the type of activation function
        # If the file_name is provided, it will load the weights
        # and biases from the given file, and load_path if given.
        
        # Initialise variables
        self.layers = []
        self.Z = []
        self.A = []
        self.dA = []
        self.dZ = []
        self.dW = []
        self.db = []
        self.cost_type = ''
        self.VdW = []
        self.Vdb = []
        
        # Analysis buffers
        self.costs = []
        self.metrics = {}
        self.eval_metrics = {}
        
        # Create network
        if file_name == "":        
            # Set structure & type
            self.network_structure = network_structure
            self.network_type = network_type
            
            # Create random
            l = 0
            layer_prev = None
            for layer in network_structure:
                # Create biases
                b  = np.zeros((layer['neurons'], 1)) # There are as many biases as there are neurons in the layer    
                # Initialise the momentum terms
                self.Vdb.append(np.zeros((layer['neurons'], 1)))
                
                # Create weights
                if l == 0 : # First layer has as many connections as neurons
                    # Set the weight variance
                    if layer['type'] == 'sigmoid':
                        w_var = np.sqrt(1/layer['inputs'])
                    elif layer['type'] == 'tanh':
                        w_var = np.sqrt(6/(layer['neurons']+layer['inputs']))
                    else:
                        w_var = w_var = np.sqrt(2/layer['inputs'])
                    # Initialise weights
                    W  = np.random.randn(layer['neurons'], layer['inputs'])*w_var            
                    # Initialise the momentum terms
                    self.VdW.append(np.zeros((layer['neurons'], layer['inputs'])))
                    
                else: # All the other layers have as many connections as the previous layer (i.e. the last to be added)
                    # Set the weight variance
                    if layer['type'] == 'sigmoid':
                        w_var = np.sqrt(1/layer_prev['neurons'])
                    elif layer['type'] == 'tanh':
                        w_var = np.sqrt(6/(layer['neurons']+layer_prev['neurons']))
                    else:
                        w_var = w_var = np.sqrt(2/layer_prev['neurons'])
                    # Initialise weights
                    W  = np.random.randn(layer['neurons'], layer_prev['neurons'])*w_var            
                    # Initialise the momentum terms
                    self.VdW.append(np.zeros((layer['neurons'], layer_prev['neurons'])))
                
                # Add weights and biases to layer
                self.layers.append({'W':W, 'b':b, 'type':layer['type']})
            
                layer_prev = layer
                l += 1
                    
        else: # Load from file
            self.load_network(file_name, load_path)     
            
        # Set type of cost function
        if self.network_type == 'regression':
            self.cost_type = 'square_error'
        elif self.network_type == 'classification':
            self.cost_type = 'log_loss'
        else:
            raise Exception('Network type not available.')  
            
    def save_network(self, file_name, path=""):
        # Saves the network weights and configuration in a file.
                
        network = {'structure':self.network_structure,
                   'layers':self.layers,
                   'type':self.network_type}
        
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
        self.network_type = network_data['type']
    
    def layer_potentials(self, layer, X):
        # Computes the potentials of a given layer
        
        # Conver inputs to np.array
        if type(X) is not np.ndarray:
            X = np.array(X)
            
        # Extract the weights and biases from the layer
        W = np.array(layer['W'])
        b = np.array(layer['b'])
        Z = np.dot(W, X) + b
        return Z
        
    
    def layer_activations(self, Z, activation_type):
        # Determines the neuron's activation (output)
        # for a given potential input.
        
        # Conver inputs to np.array
        if type(Z) is not np.ndarray:
            Z = np.array(Z)
            
        # Compute output depending on activation function type.
        # Sigmoid
        if activation_type == 'sigmoid':
            Z = 1/(1 + np.exp(-Z))
        # Hyperbolic tangent
        elif activation_type == 'tanh':
            Z = np.tanh(Z)
        # Rectified learning unit
        elif activation_type == 'relu':
            Z = np.maximum(0,Z)
        # Leaky rectified learning unit
        elif activation_type == 'leakyrelu':
            Z = np.where(Z >= 0, Z, 0.01*Z)  
        # Not available
        else:
            raise Exception('Activation function not available.')     
        
        return Z
                            
    
    def forward_propagation(self, X):
        # Compute the output for each layer
        
        # Conver inputs to np.array
        if type(X) is not np.ndarray:
            X = np.array(X)
        
        # Clear arrays
        self.A = []
        self.Z = []
        
        # Add inputs to activations as these are the activations for the first layer.
        self.A.append(X)
        
        # Loop through all layers calculating activations and potentials
        for layer in self.layers:
            self.Z.append(self.layer_potentials(layer, self.A[-1]))
            self.A.append(self.layer_activations(self.Z[-1], layer['type']))
        
        # Return only the activation of the last layer (i.e. the result)
        return self.A[-1]
    
    
    def layer_diff_activations(self, Z, activation_type):
        # Determines the neuron's differential activation (output)
        # for a given potential input.
        
        # Conver inputs to np.array
        if type(Z) is not np.ndarray:
            Z = np.array(Z)
            
        # Compute output depending on activation function type.
        # Sigmoid
        if activation_type == 'sigmoid':
            Z = (1/(1 + np.exp(-Z)))*(1 - 1/(1 + np.exp(-Z)))
        # Hyperbolic tangent
        elif activation_type == 'tanh':
            Z = 1 - (np.tanh(Z)*np.tanh(Z))
        # Rectified learning unit
        elif activation_type == 'relu':
            Z = np.where(Z > 0, 1.0, 0.0)
        # Leaky rectified learning unit
        elif activation_type == 'leakyrelu':
            Z = np.where(Z > 0, 1.0, 0.01)
        # Not available
        else:
            raise Exception('Activation function not available.')
            
        return Z
    
    def cost(self, Y, cost_type, lambd=0):
        # Get samples and reshape Y
        m = Y.shape[1]
        #Y = Y.reshape(self.A[-1].shape)
        
        # Calculate the frobenious norm of the weight matrices when required
        W_norm = 0
        JW = 0
        if lambd > 0:
            for layer in self.layers:
                W = layer['W']
                W_norm += np.linalg.norm(W, 'fro')
            
            JW = lambd*(W_norm)/(2*m)                
        
        # Calculate cost
        if cost_type == 'square_error':
            return np.sum(np.power(self.A[-1] - Y, 2), axis=1, keepdims=True)/m + JW
        elif cost_type == 'log_loss':
            ls = - np.sum(np.multiply(np.log(self.A[-1]),Y) + np.multiply(np.log(1 - self.A[-1]),(1 - Y)))/m  + JW
            return np.squeeze(ls)
        elif cost_type == 'accuracy':
            acc = np.sum(np.where((np.argmax(self.A[-1], axis=0) - np.argmax(Y, axis=0)) == 0, 1, 0))/m
            return np.squeeze(acc)
        else:
            raise Exception('Cost type not available.')   
            
    
    def cost_derivative(self, Y, cost_type):
        # Reshape Y
        Y = Y.reshape(self.A[-1].shape)
        
        # Calculate cost
        if cost_type == 'square_error':
            return self.A[-1] - Y
        elif cost_type == 'log_loss':
            return - (np.divide(Y, self.A[-1]) - np.divide(1 - Y, 1 - self.A[-1]))
        else:
            raise Exception('Cost type not available.')   
            
    
    def back_propagation(self, X, Y, lambd=0):
        # Performs the backwards propagation algorithm with the given inputs
        # and target outputs to calculate errors and gradients in the network.
        
        # Conver parameters to np.array
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(Y) is not np.ndarray:
            Y = np.array(Y)
        
        # Initialise variables
        self.dA = []
        self.dZ = []
        self.dW = []
        self.db = []
        
        ##################################
        # Step 1: Forward pass.
        # Perform forward propagation to obtain all the neuron
        # potentials and activations.
        #
        self.forward_propagation(X)
        
        ##################################
        # Step 2: Loop backwards through all layers to calculate deltas.
        # Then calculate gradients and add to list.        
        #
            
        # Get samples and reshape Y
        m = Y.shape[1]
        Y = Y.reshape(self.A[-1].shape)
        
        # Set the first dA (output layer)
        self.dA.append(self.cost_derivative(Y, self.cost_type)) # Output layer dJ/da
        
        # Loop through all layers to calculate gradients
        for l in range(-1, -len(self.layers)-1, -1):                
            W = self.layers[l]['W'] # Extract weights
            
            # Compute dZ and append
            dZ = np.multiply(self.dA[l], self.layer_diff_activations(self.Z[l], self.layers[l]['type']))
            self.dZ.insert(0, dZ)
            
            # Calculate weight gradient
            dW = np.dot(dZ, self.A[l-1].T)/m + lambd*W/m # Cost derivative + weight decay
            self.dW.insert(0, dW)
            
            # Add bias gradients
            db = np.sum(dZ,axis=1,keepdims=True)/m
            self.db.insert(0, db)            
            
            # Calculate previous layer dJ/da
            self.dA.insert(0, np.dot(W.T, self.dZ[l])) # Insert dA for previous layer
        
        
    def update_network(self, X, Y, learning_rate, lambd=0, beta=0):
        # Updates the network's weights and biases using gradient descent.
        
        ##################################
        # Step 1: Get gradients
        self.back_propagation(X, Y, lambd)
        
        
        ##################################
        # Step 2: Update weights and biases in all layers
        #
        l = 0
        for layer in self.layers:
            # Get weights and biases
            W = layer['W']
            b = layer['b']
            # Update
            # Standard gradient descent
            if beta == 0:
                W -= learning_rate*self.dW[l]
                b -= learning_rate*self.db[l]
            # Momentum version
            else:
                self.VdW[l] = beta*self.VdW[l] + (1-beta)*self.dW[l]
                self.Vdb[l] = beta*self.Vdb[l] + (1-beta)*self.db[l]
                W -= learning_rate*self.VdW[l]
                b -= learning_rate*self.Vdb[l]
            # Set
            layer['W'] = W
            layer['b'] = b
            l += 1       
        
        
    def estimate(self, X):
        return self.forward_propagation(X)
    
    def train(self, X, Y, learning_rate, epochs, optimiser='gradient_descent', lambd=0, batch_size=None, beta=None, evaluate=False, X_test=None, Y_test=None):
        # Trains a network with the given training data arrays, number
        # of epochs and batch size
        
        # Make sure the inputs and targets lists have the same lengths.
        if X.shape[1] != Y.shape[1]:
            raise Exception('The inputs and targets lists must have the same length.')
        # Check evaluation ones when enabled
        if evaluate & (X_test.shape[1] != Y_test.shape[1]):
            raise Exception('The evaluation inputs and targets lists must have the same length.')
            
        training_start_time = time.time()
        m = Y.shape[1]
        
        # Initialise metrics arrays
        ls = []
        acc = []
        if evaluate:
            eval_ls = []
            eval_acc = []
            
        epoch_start_time = time.time()    
        # Loop through epochs and batches
        for e in range(epochs):        
            # Sttandar gradient descent
            if optimiser == 'gradient_descent':
                # Update network
                self.update_network(X, Y, learning_rate)
            
            # Mini-batch
            elif (optimiser == 'mini_batch') & (batch_size >= 1) & (batch_size <= m):
                batch_num = int(m/batch_size)                
                # Generate batch
                X_batches = np.array_split(X, batch_num, axis=1)
                Y_batches = np.array_split(Y, batch_num, axis=1)
                # Repeat for all batches
                for X_t, Y_t in zip(X_batches, Y_batches):
                    # Update network with batch
                    self.update_network(X_t, Y_t, learning_rate)
            
            # Mini-batch with momentum
            elif (optimiser == 'momentum') & (beta < 1) & (beta>0) & (batch_size >= 1) & (batch_size <= m):
                batch_num = int(m/batch_size)                
                # Generate batch
                X_batches = np.array_split(X, batch_num, axis=1)
                Y_batches = np.array_split(Y, batch_num, axis=1)
                # Repeat for all batches
                for X_t, Y_t in zip(X_batches, Y_batches):
                    # Update network with batch and beta
                    self.update_network(X_t, Y_t, learning_rate, beta=beta)
                
            else:
                raise Exception('Optimiser not available.') 
            
            # Update predicted values
            self.estimate(X)
            
            # Get cost
            self.costs.append(self.cost(Y, self.cost_type, lambd))
            
            if self.network_type == 'classification':
                # Calculate metrics
                ls.append(self.cost(Y, 'log_loss', lambd)) # Log-loss
                acc.append(100*(self.cost(Y, 'accuracy'))) # Accuracy
                if evaluate:
                    self.estimate(X_test)
                    eval_ls.append(self.cost(Y_test, 'log_loss', lambd)) # Log-loss
                    eval_acc.append(100*(self.cost(Y_test, 'accuracy'))) # Accuracy
            else:
                raise Exception('Network type not available.')
                
            # Evaluate version (log test data metrics)
                    
            # Show statistics every 100 epochs
            if e%100 == 0:
                # Get the elapsed time for the epoch
                current_time = time.time()
                epoch_time = current_time - epoch_start_time
                epoch_start_time = time.time()
                    
                # Show statistics
                print("Epoch {}/{}. Log-Loss: {:.5f}, Accuracy: {:.1f}%. Elapsed time: {:.4} seconds.".format(e+1, epochs, ls[-1], acc[-1], epoch_time))
               
        
        # Get the elapsed time for the epoch
        current_time = time.time()
        epoch_time = current_time - epoch_start_time
            
        # Show statistics
        print("Epoch {}/{}. Log-Loss: {:.5f}, Accuracy: {:.1f}%. Elapsed time: {:.4} seconds.".format(e+1, epochs, ls[-1], acc[-1], epoch_time))
        
        # Build metrics dictionary
        if self.network_type == 'classification':
            self.metrics = {'logloss':ls, 'accuracy':acc}
            if evaluate:
                self.eval_metrics = {'logloss':eval_ls, 'accuracy':eval_acc}
        else:
            raise Exception('Network type not available.') 
        
        # Get the total training time
        current_time = time.time()
        training_time = current_time - training_start_time
        print("Training time: {:.4} seconds".format(training_time))
        
        return self.costs, self.metrics
        