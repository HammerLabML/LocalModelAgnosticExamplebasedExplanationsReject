import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from sklearn.metrics import auc
from sklearn.model_selection import (KFold, ParameterGrid, train_test_split)

from conformalprediction import ConformalPredictionClassifier, ConformalPredictionClassifierRejectOption, MyClassifierSklearnWrapper
from modelselection import (check_rejections, rate_of_lists, compute_threshold_knee_point, sort_increasing)


def generate_model(model, params, X_train, y_train, X_validation, y_validation):
    model = model(**params)
    model.fit(X_train, y_train)
    score = model.score(X_validation, y_validation)
    return model, score


def conformal_reject_threshold_performance(model, threshold,
                                 X_train, y_train, X_validation, y_validation):
    # Split training set into train and calibtration set (calibration set is needed for conformal prediction)
    X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=444)

    # Create conformal prediction model
    conformal_model = ConformalPredictionClassifier(MyClassifierSklearnWrapper(model))
    conformal_model.fit(X_calib, y_calib)
    reject_option_model = ConformalPredictionClassifierRejectOption(conformal_model, threshold=threshold)

    y_reject = check_rejections(X_validation, reject_option_model)

    index = list(range(len(y_validation)))
    index = [x for x in index if x not in y_reject]
    if len(X_validation[index]) == 0:  # Stop if all elements are rejected for threshold
        return

    reject_rate = rate_of_lists(y_reject, X_validation)

    # Compute accuracy of model
    new_score = model.score(X_validation[index], y_validation[index])
    return {'rejection_rate': reject_rate, 'reject_accuracy': new_score}


@ray.remote
def reject_cv_performance(trial_settings, parameters, X, y):
    """Ray Remote function used for multiprocessing/threading all gridsearch trials"""
    model_class = trial_settings.model_class
    # reject_option_model = trial_settings.reject_option_model
    rejection_thresholds = trial_settings.rejection_thresholds
    cv = trial_settings.cv

    # Generate folds
    folds = KFold(n_splits=cv)
    fold_accuracy_scores = []
    fold_models = []
    fold_rejections = []

    for train, validation in folds.split(X):
        X_train, y_train, X_validation, y_validation = X[train], y[train], X[
            validation], y[validation]
        model, accuracy_score = generate_model(
            model_class,
            parameters,
            X_train, y_train,
            X_validation, y_validation
        )
        fold_accuracy_scores.append(accuracy_score)
        fold_models.append(model)

        # compute rejects
        accuracies = [accuracy_score]
        rejection_rates = [0]
        rejection_thresholds_performance = [
            conformal_reject_threshold_performance(model, threshold,
                                         X_train, y_train, 
                                         X_validation, y_validation)
            for threshold in rejection_thresholds
        ]

        for x in rejection_thresholds_performance:
            if x is not None:
                rejection_rates.append(x['rejection_rate'])
                accuracies.append(x['reject_accuracy'])

        if rejection_rates[-1] != 1:  # Add final point of ARC curve (by defintion of ARC)
            accuracies.append(1)
            rejection_rates.append(1)

        # Compute auc for ARC
        sorted_rates, sorted_accuracies = sort_increasing(rejection_rates, accuracies)
        au_arc_score = auc(sorted_rates, sorted_accuracies)

        rejection_model_outputs = {
            'accuracies': accuracies,
            'rejection_rates': rejection_rates,
            'au_arc_score': au_arc_score
        }
        fold_rejections.append(rejection_model_outputs)

    average_au_arc = sum([x['au_arc_score'] for x in fold_rejections]) / len(fold_rejections)
    std_au_arc = np.std([x['au_arc_score'] for x in fold_rejections])

    avg_score = sum(fold_accuracy_scores) / len(fold_accuracy_scores)
    std_score = np.std(fold_accuracy_scores)

    return {
        'params': parameters,
        'fold_models': fold_models,
        'accuracy_score': fold_accuracy_scores,
        'avg_accuracy_score': avg_score,
        'std_accuracy_score': std_score,
        'fold_rejection_outputs': fold_rejections,
        'avg_au_arc': average_au_arc,
        'std_au_arc': std_au_arc
    }


class ConformalRejectOptionGridSearchCV():

    def __init__(self,
                 model_class,
                 parameter_grid,
                 rejection_thresholds,
                 cv=5):
        self.model_class = model_class
        self.parameter_grid = parameter_grid
        self.rejection_thresholds = rejection_thresholds
        self.cv = cv

        ray.shutdown()
        ray.init()

    def fit(self, X, y):
        # self.X, self.y = X, y
        self.models = self._generate_models(
            X, y)  # Fit LVQ and rejection models
        return self.best_model_params()  # Return best score

    def _generate_models(self, X, y):
        # Load datasets into ray shared memory store (optimized for numpy arrays, works best for large datasets)
        X_id = ray.put(X)
        y_id = ray.put(y)

        models = []
        for params in ParameterGrid(self.parameter_grid):
            models.append(
                reject_cv_performance.remote(self, params, X_id, y_id))

        model_ouputs = ray.get(
            models
        )  # Evaluate remote functions (i.e. trials with different parameterizations)
        return model_ouputs

    def best_model_params(self):
        # select best lvq model (by avg accuracy)
        df = self.results_df()
        idx_best_model = df['avg_accuracy_score'].idxmax()

        model_output = self.models[idx_best_model]
        params = model_output['params']

        # select best rejection threshold
        fold_rejection_output = model_output['fold_rejection_outputs']
        best_rejection_threshold_folds = [
            compute_threshold_knee_point(fold['rejection_rates'],
                                         fold['accuracies'],
                                         self.rejection_thresholds)
            for fold in fold_rejection_output
        ]
        best_rejection_threshold_folds = list(
            filter(None, best_rejection_threshold_folds))
        best_rejection_threshold = np.mean(
            best_rejection_threshold_folds
        )  # Compute best threshold as mean of thresholds

        return {
            'model_params': params,
            'rejection_threshold': best_rejection_threshold
        }

    def plot_arc_curve(self, model_idx=None, params=None):
        if model_idx is not None:
            model_output = self.models[model_idx]
        elif params is not None:
            raise NotImplementedError
        else:
            raise ValueError(
                "Supply either index of lvq model or lvq hyperparameter setting"
            )

        plt.figure()
        for idx, rejection_output in enumerate(
                model_output['fold_rejection_outputs']):
            plt.plot(rejection_output['rejection_rates'],
                     rejection_output['accuracies'],
                     label="ARC of fold " + str(idx))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Rejection Rate")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")

    def results_df(self):
        output_keys = [
            'params', 'avg_accuracy_score', 'std_accuracy_score',
            'avg_au_arc', 'std_au_arc'
        ]
        filter_keys_from_dict = lambda x, keys: {key: x[key] for key in keys}
        model_scores = [
            filter_keys_from_dict(model_ouput, output_keys)
            for model_ouput in self.models
        ]
        return pd.DataFrame(model_scores)
