# File where we make our BO class, which will be used to perform Bayesian Optimization on our model

import numpy as np
from scipy.stats import norm

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
        pass

    def BO_step(self, array):
        # Get the next hyperparameters to try
        pass

    def prob_of_improvement(self, mean, std, current_best, xi=0.01):
        # Use y_pred and y_true to calculate the probability of improvement for the next suggestion
        return norm.cdf((mean - current_best - xi) / (std + 1e-9))
    
    def expected_improvement(self, mean, std, current_best, xi=0.01):
        # start by computing the Z as we did in the probability of improvement function
        # to avoid division by 0, add a small term eg. np.spacing(1e6) to the denominator
        Z = (mean - current_best - xi) / (std + 1e-9) # or Z = (mean - current_best - eps) / (std + np.spacing(1e6))
        # now we have to compute the output only for the terms that have their std > 0
        EI = (mean - current_best - xi) * norm.cdf(Z) + std * norm.pdf(Z)
        EI[std == 0] = 0
        
        return EI
    
    def GP_UCB(mean, std, t, dim = 1.0, v = 1.0, delta = .1):
        '''
        Implementation of the Gaussian Process - Upper Confident Bound:
            GP-UBC(x) = mu + sqrt(v * beta) * sigma

        where we are usinv v = 1 and beta = 2 log( t^(d/2 + 2) pi^2 / 3 delta)
        as proved in Srinivas et al, 2010, to have 0 regret.

        :param mean: this is the mean function from the GP over the considered set of points
        :param std: this is the std function from the GP over the considered set of points
        :param t: iteration number
        :param dim: dimension of the input space
        :param v: hyperparameter that weights the beta for the exploration-exploitation trade-off. If v = 1 and another
                condition, it is proved we have 0 regret
        :param delta: hyperparameter used in the computation of beta
        :return: the value of this acquisition function for all the points
        '''
        beta =  2 * np.log((t**( (dim/2) + 2) * np.pi**2 / (3 * delta)) + 1e-9)
        UCB = mean + np.sqrt(v * beta) * std
        
        return UCB
        