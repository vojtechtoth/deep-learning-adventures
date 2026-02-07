import numpy as np

class MSELoss():
    def forward(self, y, y_pred):
        assert len(y.shape) == 2 and len(y_pred.shape) == 2, "Not a 2D array."
        assert y.shape == y_pred.shape, "Dimension mismatch"
        
        return np.mean(np.power(y - y_pred, 2))
    
    def backward(self, y, y_pred):
        assert len(y.shape) == 2 and len(y_pred.shape) == 2, "Not a 2D array."
        assert y.shape == y_pred.shape, "Dimension mismatch"
        
        n = y.shape[1]
        return (2.0 / n) * (y_pred - y)
