# File where we make our BO class, which will be used to perform Bayesian Optimization on our model

import numpy as np

max_kernel_number = 64

class BO:
    def __init__(self, model, min_kernel_number=1, max_kernel_number=64, min_dropout_rate=0.0, max_dropout_rate=0.5):
        self.model = model
        self.min_kernel_number = min_kernel_number
        self.max_kernel_number = max_kernel_number
        self.min_dropout_rate = min_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        self.X = []
        self.y = []
    
    def suggest(self):
        if len(self.X) < 2:
            # If we don't have enough data, return random values
            kernel_number = np.random.randint(self.min_kernel_number, self.max_kernel_number + 1)
            dropout_rate = np.random.uniform(self.min_dropout_rate, self.max_dropout_rate)
            return kernel_number, dropout_rate
        
        # Here we would implement the actual BO algorithm to suggest the next hyperparameters
    

    def BO_step(self, array):
        # Get the next hyperparameters to try
    

    def prob_of_improvement(self, y_pred, y_true):
        # Use y_pred and y_true to calculate the probability of improvement for the next suggestion
        pass
        