"""
Implementation of ReLu activation function.
author: Vojtěch Tóth
year: 2026
Some code is taken from a homework from SSU course at CTU Faculty of Electircal Engineering.
"""

import numpy as np

class ReluLayer():
    """
    ReLu activation layer module accepting input X of arbitrary size, outputing matrix of the same dimensions.
    """
    def __init__(self, name=None):
        super().__init__()
        
        self.name = name
        self.trainable = False


    def forward(self, X):
        """
        Forward message.
        """
        self.X = X
        return X * (X > 0)
    
    def backward(self, dY):
        """
        Computes backwards message for ReLu layer.
        """
        return dY * (self.X > 0)

    def has_params(self):
        return False
