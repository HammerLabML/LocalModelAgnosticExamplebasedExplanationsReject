import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold, train_test_split

from modelselection_conformal import ConformalRejectOptionGridSearchCV
from conformalprediction import MyClassifierSklearnWrapper, ConformalPredictionClassifier, ConformalPredictionClassifierRejectOption
from factualexplanations import FactualExplanationReject
from counterfactual import CounterfactualExplanationReject
from semifactual import SemifactualExplanationReject
from rejectexplanation import RejectExplanation
from utils import *


def get_model(model_desc):
    if model_desc == "knn":
        return KNeighborsClassifier
    elif model_desc == "randomforest":
        return RandomForestClassifier
    elif model_desc == "gnb":
        return GaussianNB
    else:
        raise ValueError(f"Invalid value of 'model_desc' -- must be one of the following 'knn', 'dectree', 'gnb'; but not '{model_desc}'")

def get_model_parameters(model_desc):
    if model_desc == "knn":
        return knn_parameters
    elif model_desc == "randomforest":
        return random_forest_parameters
    elif model_desc == "gnb":
        return {}
    else:
        raise ValueError(f"Invalid value of 'model_desc' -- must be one of the following 'knn', 'dectree', 'gnb'; but not '{model_desc}'")



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <dataset> <model> <local-explanation-model>")
        os._exit(1)

    # Specifications (provided as an input by the user)
    data_desc = str(sys.argv[1])
    model_desc = str(sys.argv[2])
    local_model_desc = str(sys.argv[3])
    """
    data_desc = "breastcancer"
    model_desc = "gnb"
    local_model_desc = "logreg"
    """
    print(data_desc, model_desc, local_model_desc)

    # Load data
    X, y = load_data(data_desc)
    print(X.shape)

    # In case of an extremly large majority class, perform simple downsampling
    if data_desc == "t21":
        rus = RandomUnderSampler()
        X, y = rus.fit_resample(X, y)
    
    # K-Fold
    xorig=[];reject_groundtruth = []
    surrogate_cf = [];surrogate_sf = []
    global_cf = [];global_sf = [];global_f = []
    global_cf_feasibility = [];global_sf_feasibility = []
    surrogate_cf_feasibility = [];surrogate_sf_feasibility = []

    for train_index, test_index in KFold(n_splits=n_folds, shuffle=True, random_state=None).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # If necessary (in case of an highly imbalanced data set), apply Synthetic Minority Over-sampling Technique (SMOTE)
        try:
            if data_desc == "flip":
                sm = SMOTE(k_neighbors=1)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                X_test, y_test = sm.fit_resample(X_test, y_test)
        except Exception as ex:
            print(ex)
            continue

        # Hyperparameter tuning
        model_search = ConformalRejectOptionGridSearchCV(model_class=get_model(model_desc), parameter_grid=get_model_parameters(model_desc), rejection_thresholds=reject_thresholds)
        best_params = model_search.fit(X_train, y_train)
        
        # Split training set into train and calibtration set (calibration set is needed for conformal prediction)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2)
        
        # Fit & evaluate model and reject option
        model = get_model(model_desc)(**best_params["model_params"])
        model.fit(X_train, y_train)
        print(f"Model score: {model.score(X_train, y_train)}, {model.score(X_test, y_test)}")

        conformal_model = ConformalPredictionClassifier(MyClassifierSklearnWrapper(model))
        conformal_model.fit(X_calib, y_calib)
        print(f"Conformal predictor score: {conformal_model.score(X_train, y_train)}, {conformal_model.score(X_test, y_test)}")

        print(f'Rejection threshold: {best_params["rejection_threshold"]}')
        reject_option = ConformalPredictionClassifierRejectOption(conformal_model, threshold=best_params["rejection_threshold"])
        
        # For each sample in the test set, check if it is rejected
        y_rejects = []
        for i in range(X_test.shape[0]):
            x = X_test[i,:]
            if reject_option(x):
                y_rejects.append(i)
        print(f"{len(y_rejects)}/{X_test.shape[0]} are rejected")
        
        # Select random subset of features which are going to be perturbed
        perturbed_features_idx = select_random_feature_subset(X_train.shape[1])
        print(f"Perturbed features: {perturbed_features_idx}")

        # Find all samples in the test set that are rejected because of the perturbation
        X_test = apply_perturbation(X_test, perturbed_features_idx)  # Apply perturbation
        
        y_rejects_due_to_perturbations = []
        for i in range(X_test.shape[0]):    # Check which samples are now rejected
            x = X_test[i,:]
            if reject_option(x) and i not in y_rejects:
                y_rejects_due_to_perturbations.append(i)
        print(f"{len(y_rejects_due_to_perturbations)}/{X_test.shape[0]} are rejected due to perturbations")
        if len(y_rejects_due_to_perturbations) == 0:
            continue

        # Compute explanations of rejected samples
        try:
            factual_explainer = FactualExplanationReject(reject_option, X_train)
        except Exception as ex:
            print(ex)
            continue
        cf_explainer = CounterfactualExplanationReject(reject_option, C_reg=100., X_train=X_train, solver="Nelder-Mead")
        sf_explainer = SemifactualExplanationReject(reject_option, C_simple=.1, C_reg=1., C_feasibility=100., C_sf=100., sparsity_upper_bound=2., solver="Nelder-Mead")
        surrogate_explainer = RejectExplanation(reject_option, model_desc=local_model_desc, regularization_strength=1., num_samples=500, scale=2.)

        xorig_ = []
        global_cf_ = [];global_sf_=[];global_f_=[]
        global_cf_feasibility_ = [];global_sf_feasibility_ = []
        surrogate_cf_ = [];surrogate_sf_=[]
        surrogate_cf_feasibility_ = [];surrogate_sf_feasibility_ = []

        for idx in y_rejects_due_to_perturbations:
            try:
                x_orig = X_test[idx, :]

                # Compute explanations using the gloal black-box model
                x_f = factual_explainer.compute_explanation(x_orig)
                x_cf = cf_explainer.compute_explanation(x_orig)
                x_sf = sf_explainer.compute_explanation(x_orig)

                # Compute explanations using the local surrogate
                surrogate_expl = surrogate_explainer.compute_explanation(x_orig)

                # Evaluate feasibility & Save results
                if x_cf is not None:
                    global_cf_feasibility_.append(not reject_option.reject(x_cf))
                if x_sf is not None:
                    global_sf_feasibility_.append((reject_option.reject(x_sf)) and (reject_option.criterion(x_sf) >= reject_option.criterion(x_orig)))
                if surrogate_expl["cf"] is not None:
                    surrogate_cf_feasibility_.append(not reject_option.reject(surrogate_expl["cf"]))
                if surrogate_expl["sf"] is not None:
                    surrogate_sf_feasibility_.append((reject_option.reject(surrogate_expl["sf"])) and (reject_option.criterion(surrogate_expl["sf"]) >= reject_option.criterion(surrogate_expl["sf"])))

                xorig_.append(x_orig)
                global_cf_.append(x_cf);global_sf_.append(x_sf);global_f_.append(x_f)
                surrogate_cf_.append(surrogate_expl["cf"]);surrogate_sf_.append(surrogate_expl["sf"])
            except Exception as ex:
                print(ex)
        
        xorig.append(xorig_)
        global_cf.append(global_cf_);global_sf.append(global_sf_);global_f.append(global_f_)
        global_cf_feasibility.append(global_cf_feasibility_);global_sf_feasibility.append(global_sf_feasibility_)
        surrogate_cf.append(surrogate_cf_);surrogate_sf.append(surrogate_sf_)
        if len(surrogate_cf_feasibility_) != 0:
            surrogate_cf_feasibility.append(surrogate_cf_feasibility_)
        if len(surrogate_sf_feasibility_) != 0:
            surrogate_sf_feasibility.append(surrogate_sf_feasibility_)
        reject_groundtruth.append(perturbed_features_idx)

    # Store results for post-processing and analyze
    np.savez(f"results/results_{data_desc}_{model_desc}_{local_model_desc}.npz", xorig=xorig, reject_groundtruth=reject_groundtruth, global_cf=global_cf, global_sf=global_sf, global_f=global_f,
                        global_cf_feasibility=global_cf_feasibility, global_sf_feasibility=global_sf_feasibility,
                        surrogate_cf=surrogate_cf, surrogate_sf=surrogate_sf, surrogate_cf_feasibility=surrogate_cf_feasibility, surrogate_sf_feasibility=surrogate_sf_feasibility)
