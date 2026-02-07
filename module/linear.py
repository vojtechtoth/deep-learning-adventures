"""
Implementation of fully connected linear layer using dict-based parameters.
author: Vojtěch Tóth
year: 2026
Some code is taken from a homework from SSU course at CTU Faculty of Electrical Engineering.
"""

import numpy as np

class LinearLayer():
    """
    Linear layer module accepting input X of size (..., n_inputs) and outputting matrix of size (..., n_outputs)
    """
    def __init__(self, n_inputs, n_outputs, name=None, rng=None, init="xavier"):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.name = name
        self.rng = np.random.default_rng() if rng is None else rng
        self.trainable = True
        self.grads = {}
        self.params = {}

        if init == "he":
            self._init_he()
        elif init == "xavier":
            self._init_xavier()
        else: 
            raise ValueError(f"Unknown init method {init} in linear layer. Use 'he' or 'xavier'.")

    def forward(self, X):
        """
        Forward pass.
        :param X: input array, shape (n_samples, n_inputs)
        :return: output array, shape (n_samples, n_outputs)
        """
        assert X.shape[1] == self.n_inputs, f"Dimension mismatch: {X.shape[1]} != {self.n_inputs}"
        self.X = X
        return np.dot(X, self.params["W"]) + self.params["b"]

    def backward(self, dY):
        """
        Backward pass: compute gradients and return backward message.
        :param dY: gradient from next layer
        :return: gradient wrt input X
        """
        # gradients wrt parameters
        dW = np.dot(self.X.T, dY) / self.X.shape[0]
        db = dY.mean(axis=0)
        self.grads = {"W": dW, "b": db}

        # gradient wrt input
        dX = np.dot(dY, self.params["W"].T)
        return dX

    def _init_he(self):
        scale = np.sqrt(2.0 / self.n_inputs)
        W = self.rng.normal(0.0, scale, (self.n_inputs, self.n_outputs))
        b = np.zeros(self.n_outputs)
        self.params = {"W": W, "b": b}

    def _init_xavier(self):
        scale = np.sqrt(1.0 / self.n_inputs)
        W = self.rng.normal(0.0, scale, (self.n_inputs, self.n_outputs))
        b = np.zeros(self.n_outputs)
        self.params = {"W": W, "b": b}

    def has_params(self):
        return True

    def parameters(self):
        """Generator to yield (param, grad) pairs."""
        assert set(self.params.keys()) == set(self.grads.keys()), (
            f"Parameter keys and gradient keys mismatch in LinearLayer '{self.name}'."
        )
        for key in self.params:
            yield self.params[key], self.grads[key]
