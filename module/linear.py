"""
Implementation of fully connected linear layer.
author: Vojtěch Tóth
year: 2026
Some code is taken from a homework from SSU course at CTU Faculty of Electircal Engineering.
"""

import numpy as np

from module import Module

class Linear(Module):
    """
    Linear layer module accepting input X of size (..., n_inputs) and outputing matrix of size (..., n_outputs)
    """
    def __init__(self, n_inputs, n_outputs, name=None, rng=None, init=None):
        super(Linear, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.name = name
        self.rng = np.random.default_rng() if rng is None else rng
        self.trainable = True

        if init == "he":
            self._init_he()
        elif init == "xavier":
            self._init_xavier()
        else: 
            raise ValueError(f"Unknown init method {init} in linear layer. Use 'he' or 'xavier'.")

    def forward(self, X):
        """
        Forward message.
        :param X: layer inputs, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, n_outputs)
        """
        assert X.shape[1] == self.n_inputs, f"Dimensions mismatch. {X.shape[1]} != {self.n_inputs}."
        self.X = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, dY):
        """
        Computes backwards message for linear layer.
        
        :param dY: backward message from following layer
        :return: 
        """
        # gradients wrt parameters
        self.dW = np.dot(self.X.T, dY) / self.X.shape[0]
        self.db = dY.mean(axis=0)

        # gradient wrt input
        dX = np.dot(dY, self.W.T)

        return dX

    def _init_he(self):
        scale = np.sqrt(2.0 / self.n_inputs)
        self.W = self.rng.normal(0.0, scale, (self.n_inputs, self.n_outputs))
        self.b = np.zeros(self.n_outputs)

    def _init_xavier(self):
        scale = np.sqrt(1.0 / self.n_inputs)
        self.W = self.rng.normal(0.0, scale, (self.n_inputs, self.n_outputs))
        self.b = np.zeros(self.n_outputs)

    def has_params(self):
        return True
    
    def update_params(self, learning_rate=1.0):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db