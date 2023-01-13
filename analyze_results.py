import sys
import os
import csv
import numpy as np

from utils import evaluate_sparsity, evaluate_perturbed_features_recovery


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: <results.npz>")
        os._exit(1)
    
    f_in = sys.argv[1]

    # Load results
    data = np.load(f_in, allow_pickle=True)
    reject_groundtruth = data["reject_groundtruth"];xorig = data["xorig"]
    surrogate_cf = data["surrogate_cf"];surrogate_sf = data["surrogate_sf"]
    global_cf = data["global_cf"];global_sf = data["global_sf"];global_f = data["global_f"]

    # Results
    global_cf_sparsity = [];global_cf_fidelity = [];global_cf_failure = [];global_cf_feasibility = np.concatenate(data["global_cf_feasibility"])
    global_sf_sparsity = [];global_sf_fidelity = [];global_sf_failure = [];global_sf_feasibility = np.concatenate(data["global_sf_feasibility"])
    global_f_failure = []; global_f_sparsity = []
    surrogate_cf_sparsity = [];surrogate_cf_fidelity = [];surrogate_cf_failure = [];surrogate_cf_feasibility = np.concatenate(data["surrogate_cf_feasibility"])
    surrogate_sf_sparsity = [];surrogate_sf_fidelity = [];surrogate_sf_failure = [];surrogate_sf_feasibility = np.concatenate(data["surrogate_sf_feasibility"])

    # Analysis
    for feature_idx, xorig_, surrogate_cf_, surrogate_sf_, global_cf_, global_sf_, global_f_ in zip(reject_groundtruth, xorig, surrogate_cf, surrogate_sf, global_cf, global_sf, global_f):
        for cf, x_orig in zip(global_cf_, xorig_):
            if cf is not None:
                global_cf_sparsity.append(evaluate_sparsity(cf, x_orig))
                global_cf_fidelity.append(evaluate_perturbed_features_recovery(cf, x_orig, feature_idx))
                global_cf_failure.append(0)
            else:
                global_cf_failure.append(1. / len(xorig_))
        for sf, x_orig in zip(global_sf_, xorig_):
            if sf is not None:
                global_sf_sparsity.append(evaluate_sparsity(sf, x_orig))
                global_sf_fidelity.append(1. - evaluate_perturbed_features_recovery(sf, x_orig, feature_idx))
                global_sf_failure.append(0)
            else:
                global_sf_failure.append(1. / len(xorig_))
        for factual, x_orig in zip(global_f_, xorig_):
            if factual is not None:
                global_f_sparsity.append(evaluate_sparsity(factual, x_orig))
                global_f_failure.append(0)
            else:
                global_f_failure.append(1. / len(xorig_))

        for cf, x_orig in zip(surrogate_cf_, xorig_):
            if cf is not None:
                surrogate_cf_sparsity.append(evaluate_sparsity(cf, x_orig))
                surrogate_cf_fidelity.append(evaluate_perturbed_features_recovery(cf, x_orig, feature_idx))
                surrogate_cf_failure.append(0)
            else:
                surrogate_cf_failure.append(1. / len(xorig_))
        for sf, x_orig in zip(surrogate_sf_, xorig_):
            if sf is not None:
                surrogate_sf_sparsity.append(evaluate_sparsity(sf, x_orig))
                surrogate_sf_fidelity.append(1. - evaluate_perturbed_features_recovery(sf, x_orig, feature_idx))
                surrogate_sf_failure.append(0)
            else:
                surrogate_sf_failure.append(1. / len(xorig_))
        
    # Summary
    f_out = f_in.replace(".npz", ".csv")
    with open(f_out, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(["Global-Cf Failure", np.round(np.mean(global_cf_failure), 2), np.round(np.var(global_cf_failure), 2)])
        csvwriter.writerow(["Global-Cf Sparsity", np.round(np.mean(global_cf_sparsity), 2), np.round(np.var(global_cf_sparsity), 2)])
        csvwriter.writerow(["Global-Cf Fidelity", np.round(np.mean(global_cf_fidelity), 2), np.round(np.var(global_cf_fidelity), 2)])
        csvwriter.writerow(["Global-Cf Feasibility", np.round(np.mean(global_cf_feasibility), 2), np.round(np.var(global_cf_feasibility), 2)])

        csvwriter.writerow(["Global-Sf Failure", np.round(np.mean(global_sf_failure), 2), np.round(np.var(global_sf_failure), 2)])
        csvwriter.writerow(["Global-Sf Sparsity", np.round(np.mean(global_sf_sparsity), 2), np.round(np.var(global_sf_sparsity), 2)])
        csvwriter.writerow(["Global-Sf Fidelity", np.round(np.mean(global_sf_fidelity), 2), np.round(np.var(global_sf_fidelity), 2)])
        csvwriter.writerow(["Global-Sf Feasibility", np.round(np.mean(global_sf_feasibility), 2), np.round(np.var(global_sf_feasibility), 2)])

        csvwriter.writerow(["Global-f Failure", np.round(np.mean(global_f_failure), 2), np.round(np.var(global_f_failure), 2)])
        csvwriter.writerow(["Global-f Sparsity", np.round(np.mean(global_f_sparsity), 2), np.round(np.var(global_f_sparsity), 2)])

        csvwriter.writerow(["Surrogate-Cf Failure", np.round(np.mean(surrogate_cf_failure), 2), np.round(np.var(surrogate_cf_failure), 2)])
        csvwriter.writerow(["Surrogate-Cf Sparsity", np.round(np.mean(surrogate_cf_sparsity), 2), np.round(np.var(surrogate_cf_sparsity), 2)])
        csvwriter.writerow(["Surrogate-Cf Fidelity", np.round(np.mean(surrogate_cf_fidelity), 2), np.round(np.var(surrogate_cf_fidelity), 2)])
        csvwriter.writerow(["Surrogate-Cf Feasibility", np.round(np.mean(surrogate_cf_feasibility), 2), np.round(np.var(surrogate_cf_feasibility), 2)])

        csvwriter.writerow(["Surrogate-Sf Failure", np.round(np.mean(surrogate_sf_failure), 2), np.round(np.var(surrogate_sf_failure), 2)])
        csvwriter.writerow(["Surrogate-Sf Sparsity", np.round(np.mean(surrogate_sf_sparsity), 2), np.round(np.var(surrogate_sf_sparsity), 2)])
        csvwriter.writerow(["Surrogate-Sf Fidelity", np.round(np.mean(surrogate_sf_fidelity), 2), np.round(np.var(surrogate_sf_fidelity), 2)])
        csvwriter.writerow(["Surrogate-Sf Feasibility", np.round(np.mean(surrogate_sf_feasibility), 2), np.round(np.var(surrogate_sf_feasibility), 2)])
