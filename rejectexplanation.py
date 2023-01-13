import numpy as np
import random
from scipy.sparse import rand
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from ceml.sklearn import generate_counterfactual

from semifactual import generate_semifactualexplanation
from counterfactual import MemoryCf


class RejectExplanation():
    def __init__(self, reject_option, model_desc="dectree", tree_max_depth=3, regularization_strength=1., num_samples=100, scale=2., **kwds):
        self.reject_option = reject_option
        self.model_desc = model_desc
        self.tree_max_depth = tree_max_depth
        self.C = regularization_strength
        self.num_samples = num_samples
        self.scale = scale

        super().__init__(**kwds)

    def __sample(self, x_orig):
        return x_orig + self.scale * rand(1, x_orig.shape[0], density=random.random()).A.flatten()

    def __fit_local_approximatation(self, x_orig):
        # Sample around x_orig
        X = [x_orig] + [self.__sample(x_orig) for _ in range(self.num_samples)]

        # Label samples according the output of the reject option
        y = [self.reject_option(x) for x in X]

        X = np.array(X)
        y = np.array(y)

        # Fit decision tree to labeled data set
        model = None
        if self.model_desc == "dectree":
            model = DecisionTreeClassifier(max_depth=self.tree_max_depth)
        elif self.model_desc == "logreg":
            model = LogisticRegression(penalty="l1", C=self.C, solver="saga", multi_class="multinomial")
        else:
            raise ValueError(f"Invalid value of 'model_desc' -- must be either 'dectree' or 'logreg' but not '{self.model_desc}'")

        model.fit(X, y.ravel())
        print(model.score(X, y.ravel()))

        return model, X, y

    def compute_explanation(self, x_orig, features_whitelist=None):
        # Fit a local (simple) approximation of the reject option around x_orig
        model, X, y = self.__fit_local_approximatation(x_orig)
        y = np.array([0 if model.predict(x.reshape(1, -1)) == 0 and not self.reject_option(x) else 1 for x in X])

        # Compute a counterfactual explanation of x_orig under the local approximation
        if model.predict(x_orig.reshape(1, -1)) == 0:  # Is sample originally rejected? If not, computing an explanation does not make much sense!
            xcf = None;xsf=None
            print("Sample is missclassified.")
        else:
            cf_algo = MemoryCf(X, y);xcf = cf_algo.compute_explanation(x_orig, y_target=0)  # Compute "plausible" counterfactual

            # Compute a semifactual explanation of x_orig under the local approximation
            xsf = generate_semifactualexplanation(model, x_orig)

        return {"cf": xcf, "sf": xsf}
