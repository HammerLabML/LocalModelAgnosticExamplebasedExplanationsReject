from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from reject_option import RejectOption


class MyClassifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict_proba(self, X):
        raise NotImplementedError()


class MyClassifierSklearnWrapper(MyClassifier):
    def __init__(self, model, **kwds):
        if not isinstance(model, ClassifierMixin):
            raise TypeError(f"'model' must be an instance of 'sklearn.base.ClassifierMixin' not of '{type(model)}'")
        if not (hasattr(type(model), 'predict_proba') and callable(getattr(type(model), 'predict_proba'))):
            raise TypeError(f"'model' does not have a method 'predict_proba'")
        self.model = model

        super().__init__(**kwds)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ConformalPredictionClassifier():
    def __init__(self, clf_model):
        if not isinstance(clf_model, MyClassifier):
            raise TypeError(f"'clf_model' must be an instance of 'MyClassifier' not of '{type(clf_model)}'")

        self.clf_model = clf_model
        self.calibration_set = None
        self.classes = None
        
    def _non_conformity_measure(self, Y_proba, y_target):
        alphas = []

        for i in range(len(y_target)):
            d = Y_proba[i,:] - Y_proba[i, y_target[i]]
            d[y_target[i]] = float("-inf")

            alphas.append(np.max(d))

        return alphas

    def score(self, X, y):
        y_pred = self.predict(X)[0] # Prediction of labels only
        return accuracy_score(y, y_pred)

    def fit(self, X_calib, y_calib):
        Y_calib_proba = self.clf_model.predict_proba(X_calib)
        self.calibration_set = self._non_conformity_measure(Y_calib_proba, y_calib)

        self.classes = np.unique(y_calib)

    def predict(self, X):
        if self.calibration_set is None or self.classes is None:
            raise Exception("Conformal predictor have not been fitted -- call method 'fit()' before making predictions!")
        
        # Compute non-conformity measure for each possible target class
        Y_pred_proba = self.clf_model.predict_proba(X)
        alphas_pred = np.array([self._non_conformity_measure(Y_pred_proba, np.repeat(y_target, repeats=Y_pred_proba.shape[0])) for y_target in self.classes]).T

        # Compute final prediciton, confidence and credibility scores
        p_values = []   # Compute p-values
        n_calibration_samples = len(self.calibration_set)
        for i in range(X.shape[0]):
            p_values_ = []
            for j in range(len(self.classes)):
                p = 1. / (n_calibration_samples + 1.) * np.sum(self.calibration_set >= alphas_pred[i,j])
                p_values_.append(p)
            p_values.append(p_values_)
        p_values = np.array(p_values)

        y_pred = np.argmax(p_values, axis=1)    # Compute prediction, ...
        credibility = np.max(p_values, axis=1)
        confidence = []
        for i in range(X.shape[0]):
            confidence.append(1. - p_values[i, np.argsort(p_values[i,:])[-2]])
        confidence = np.array(confidence)

        return y_pred, confidence, credibility


class ConformalPredictionClassifierRejectOption(RejectOption):
    def __init__(self, conformal_prediction_model, **kwds):
        if not isinstance(conformal_prediction_model, ConformalPredictionClassifier):
            raise TypeError(f"'conformal_prediction_model' must be an instance of 'ConformalPredictionClassifier' not of '{type(conformal_prediction_model)}'")
        self.conformal_predictor = conformal_prediction_model

        super().__init__(**kwds)

    def criterion(self, x):
        return self.conformal_predictor.predict(x.reshape(1, -1))[2]  # Use credibility as a score 
