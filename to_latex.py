import sys
import os
import pandas as pd


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: <results.csv>")
        os._exit(1)

    f_in = str(sys.argv[1])    
    df_results = pd.read_csv(f_in, header=None)

    print(f"& ${df_results.loc[df_results[0] == 'Surrogate-Cf Feasibility'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Surrogate-Cf Feasibility'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Global-Cf Feasibility'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Global-Cf Feasibility'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Surrogate-Cf Fidelity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Surrogate-Cf Fidelity'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Global-Cf Fidelity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Global-Cf Fidelity'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Surrogate-Cf Sparsity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Surrogate-Cf Sparsity'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Global-Cf Sparsity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Global-Cf Sparsity'].to_numpy()[0][2]}$")

    print(f"& ${df_results.loc[df_results[0] == 'Surrogate-Sf Feasibility'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Surrogate-Sf Feasibility'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Global-Sf Feasibility'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Global-Sf Feasibility'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Surrogate-Sf Fidelity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Surrogate-Sf Fidelity'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Global-Sf Fidelity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Global-Sf Fidelity'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Surrogate-Sf Sparsity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Surrogate-Sf Sparsity'].to_numpy()[0][2]}$ \
            & ${df_results.loc[df_results[0] == 'Global-Sf Sparsity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Global-Sf Sparsity'].to_numpy()[0][2]}$")

    print(f"& ${df_results.loc[df_results[0] == 'Global-f Sparsity'].to_numpy()[0][1]} \pm {df_results.loc[df_results[0] == 'Global-f Sparsity'].to_numpy()[0][2]}$")
