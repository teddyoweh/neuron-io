import numpy as np
class Loss:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-7   
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)   
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-7   
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)   
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))