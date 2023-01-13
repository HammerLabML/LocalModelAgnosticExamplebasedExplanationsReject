import numpy as np
from scipy.optimize import minimize

from utils import non_zero_threshold_sparsity



def generate_semifactualexplanation(model, x_orig, C_simple=.1, C_reg=1., C_feasibility=1., C_sf=1., sparsity_upper_bound=2., max_iter=None, solver="Nelder-Mead"):
        y_target = int(model.predict([x_orig])[0])
        y_proba = model.predict_proba([x_orig])[0][y_target]

        low_complexity_expl_loss = lambda x: C_simple * max([np.sum(np.abs(x[i] - x_orig[i]) > non_zero_threshold_sparsity for i in range(x_orig.shape[0])) - sparsity_upper_bound, 0])
        similarity_orig_loss = lambda x: -1. * C_reg * np.linalg.norm(x - x_orig, 2)
        feasibility_loss = lambda x: C_feasibility * min([model.predict_proba([x])[0][y_target] - .5, 0])
        sf_loss = lambda x: C_sf * max([y_proba - model.predict_proba([x])[0][y_target], 0])
        
        loss = lambda x: low_complexity_expl_loss(x) + similarity_orig_loss(x) + feasibility_loss(x) + sf_loss(x)

        # Minimize loss function
        res = minimize(fun=loss, x0=x_orig, method=solver, options={'maxiter': max_iter})
        x_sf = res["x"]

        return x_sf



class SemifactualExplanationReject():
    def __init__(self, reject_option, C_simple=.1, C_reg=1., C_feasibility=1., C_sf=1., sparsity_upper_bound=2., solver="Nelder-Mead", max_iter=None, **kwds):
        self.reject_option = reject_option
        self.solver = solver
        self.max_iter = max_iter
        self.C_simple = C_simple
        self.C_reg = C_reg
        self.C_feasibility = C_feasibility
        self.C_sf = C_sf
        self.sparsity_upper_bound = sparsity_upper_bound

        super().__init__(**kwds)

    def __compute_semifactual(self, x_orig):
        # Loss function
        low_complexity_expl_loss = lambda x: self.C_simple * max([np.sum(np.abs(x[i] - x_orig[i]) > non_zero_threshold_sparsity for i in range(x_orig.shape[0])) - self.sparsity_upper_bound, 0])
        similarity_orig_loss = lambda x: -1. * self.C_reg * np.linalg.norm(x - x_orig, 2)
        feasibility_loss = lambda x: self.C_feasibility * max([self.reject_option.criterion(x) - self.reject_option.threshold, 0])
        sf_loss = lambda x: self.C_sf * max([self.reject_option.criterion(x_orig) - self.reject_option.criterion(x), 0])
        
        loss = lambda x: low_complexity_expl_loss(x) + similarity_orig_loss(x) + feasibility_loss(x) + sf_loss(x)

        # Minimize loss function
        res = minimize(fun=loss, x0=x_orig, method=self.solver, options={'maxiter': self.max_iter})
        x_sf = res["x"]

        return x_sf

    def compute_explanation(self, x_orig):
        return self.__compute_semifactual(x_orig)
