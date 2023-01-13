import numpy as np


class FactualExplanationReject():
    def __init__(self, reject_option, X_data, **kwds):
        self.reject_option = reject_option
        self.X_data = X_data
        self.y_data = [int(self.reject_option.criterion(x) < self.reject_option.threshold) for x in self.X_data]
        if sum(self.y_data) == 0:
            raise ValueError("None of the training samples is rejected!")

        super().__init__(**kwds)

    def compute_explanation(self, x_orig):
        idx = np.where(np.array(self.y_data) == 1)[0]
        X_data = self.X_data[idx,:]  # Only consider samples that are also rejected
        X_diff = X_data - x_orig
        dist = [np.linalg.norm(X_diff[i,:].flatten(), 1) for i in range(X_diff.shape[0])]
        idx = np.argmin(dist)

        return X_data[idx,:]
