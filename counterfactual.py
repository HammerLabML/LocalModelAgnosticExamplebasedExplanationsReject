import numpy as np
import random
from scipy.optimize import minimize


class CounterfactualExplanationReject():
    def __init__(self, reject_option, C_reg=1., X_train=None, solver="Nelder-Mead", max_iter=None, **kwds):
        self.reject_option = reject_option
        self.solver = solver
        self.max_iter = max_iter
        self.C_reg = C_reg
        
        self.X_train = X_train
        y_train = [not self.reject_option(x) for x in self.X_train]
        idx = np.where(np.array(y_train) == 1)[0]
        self.X_train = self.X_train[idx,:]

        super().__init__(**kwds)

    def __compute_counterfactual(self, x_orig):
        # Loss function
        similarity_orig_loss = lambda x: np.linalg.norm(x - x_orig, 1)
        feasability_loss = lambda x:  abs(min([self.reject_option.criterion(x) - self.reject_option.threshold, 0]))

        loss = lambda x: similarity_orig_loss(x) + self.C_reg * feasability_loss(x)

        # Minimize loss function
        x0 = random.choice(self.X_train)
        res = minimize(fun=loss, x0=x0, method=self.solver, options={'maxiter': self.max_iter})
        x_cf = res["x"]

        return x_cf


    def compute_explanation(self, x_orig):
        return self.__compute_counterfactual(x_orig)



class MemoryCf():
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def compute_explanation(self, x_orig, y_target):
        idx = np.where(np.array(self.y_data) == y_target)[0]
        X_data = self.X_data[idx,:]  # Only consider samples that are also rejected
        X_diff = X_data - x_orig
        dist = [np.linalg.norm(X_diff[i,:].flatten(), 1) for i in range(X_diff.shape[0])]
        idx = np.argmin(dist)

        return X_data[idx,:]

