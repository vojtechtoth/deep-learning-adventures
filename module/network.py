class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        """Forward pass through the whole network"""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta):
        """Backward pass through the network"""
        # propagate backward through layers in reverse
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta

    def _parameters(self):
        """Yield all parameters from all layers"""
        for layer in self.layers:
            if layer.has_params():
                yield from layer.parameters()

    def parameters(self   ):
        """Public method returns a fresh generator"""
        return self._parameters()  