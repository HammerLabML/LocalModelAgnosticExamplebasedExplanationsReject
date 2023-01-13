import numpy as np
from kneed import KneeLocator


def check_rejections(X_test, reject_option_model):
    # For each sample in the test set, check if it is rejected
    y_rejects = []
    for i in range(X_test.shape[0]):
        x = X_test[i,:]
        if reject_option_model(x):
            y_rejects.append(i)

    return y_rejects

def rate_of_lists(part, whole):
    return len(part) / len(whole)

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

def compute_threshold_knee_point(x, y, thresholds):
    x = np.array(x)
    y = np.array(y)
    thresholds = np.array(thresholds)

    # Filter duplicate x,y coordinates
    # Remove non duplicate rejection_rates
    # x, indices = np.unique(x, return_index=True)
    # y = y[indices]
    # thresholds = thresholds[indices]
    
    # remove non-increasing elements
    # indices = [0] + [idx for idx in range(1, len(x)) if y[idx] >= y[idx-1]]
    # x = x[indices]
    # y = y[indices]
    # thresholds = thresholds[indices]

    # compute knee point rejection rate
    kneedle = KneeLocator(x, y, S=1.0, curve="concave", online=True)

    if kneedle.knee is not None:
        # TODO find/approximate rejection_threshold for the knee point rejection rate
        idx_nearest = find_nearest(x, kneedle.knee)
        # kneedle.plot_knee()
        # plt.plot(kneedle.x_difference, kneedle.y_difference)
        # kneedle.plot_knee_normalized()
        return thresholds[idx_nearest]

    return None

def sort_increasing(rejection_rates, accuracies):
    rejection_rates = np.array(rejection_rates)
    accuracies = np.array(accuracies)
    
    indices = np.argsort(rejection_rates)

    rejection_rates = rejection_rates[indices]
    accuracies = accuracies[indices]
    return rejection_rates, accuracies
