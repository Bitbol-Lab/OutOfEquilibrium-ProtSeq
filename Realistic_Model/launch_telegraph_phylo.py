import random

import numpy as np
import generate as ge
import argparse
import numpy as np

from generate import MSAGenerator

# Load the fields and couplings
fields = np.load('/home/malbrank/Documents/phylogeny-nico/data/Synthetic_data/bmDCA_parameters/PF00004_h.npy')
couplings = np.load('/home/malbrank/Documents/phylogeny-nico/data/Synthetic_data/bmDCA_parameters/PF00004_J.npy')


# Initialize the MSA generator


def generate_and_save_msa(temphigh, templow, nbrmut, nbrseq):
    intermediate_states = [int(x) for x in nbrmut.split(",")]
    print(f"Generating telegraph_msa_{nbrseq}_{nbrmut}_{templow}_{temphigh}.npy")
    msagen = ge.MSAGenerator(fields, couplings, temp_high=temphigh, temp_low=templow, tau=10,
                             max_mutations=intermediate_states[-1])
    msa_final = msagen.generate_msa_no_phylo(nbrseq, intermediate_states[-1],
                                             keep_intermediate_states=intermediate_states, n_workers=8)
    for i, nbrmut_ in enumerate(intermediate_states):
        filename = f"telegraph_msa_{nbrseq}_{nbrmut_}_{templow}_{temphigh}.npy"
        np.save(filename, msa_final[:, i])
        print(f"{filename} saved")

    msagen.switch_temps, msagen.switch_times

def final_plot():
    tau=300
    initial_state = np.load("msa_70000_10000_0.5.npy")
    intermediate_states = [int(x) for x in np.arange(0, 4001, 10)]
    random.seed(43)
    switch_times = MSAGenerator._pre_sample_switch_times(tau=tau, max_mutations=intermediate_states[-1])
    np.save("plot_pf00004-2/switch_times.npy", switch_times)
    params = [(2, 1, 70000), (2, 0.5, 70000), (1, 0.5, 70000),]
    for temphigh, templow, nbrseq in params:
        msagen = ge.MSAGenerator(fields, couplings, temp_high=temphigh, temp_low=templow, tau=tau,
                                 max_mutations=intermediate_states[-1], switch_times=switch_times)
        msa_final = msagen.generate_msa_phylo(initial_state, intermediate_states[-1],
                                             keep_intermediate_states=intermediate_states, n_workers=16)
        for i, nbrmut_ in enumerate(intermediate_states):
            filename = f"plot_pf00004/telegraph_msa_phylo_{nbrseq}_{nbrmut_}_{templow}_{temphigh}_{tau}.npy"
            np.save(filename, msa_final[:, i])
            print(f"{filename} saved")

if __name__ == "__main__":
    final_plot()
    # Define the parameters
    #nbrseq = 70000
    #temphighs = [2]
    #templows = [1, 0.5]
    nbrmuts = ",".join([str(x) for x in np.arange(0, 4001, 10)])
    # python launch_cython_cyril_telegraph.py --temphigh 2 --templow 1 --nbrmut "1000,2000,4000"
    # python launch_cython_cyril_telegraph.py --temphigh 2 --templow 0.5 --nbrmut "1000,2000,4000"
    # generate_and_save_msa(2, 1, "1000,2000,4000", 70000)
