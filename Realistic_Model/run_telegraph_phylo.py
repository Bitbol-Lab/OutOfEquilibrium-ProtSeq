import random

import numpy as np
import generate as ge
import argparse
import numpy as np

from generate import MSAGenerator

# Load the fields and couplings
fields = np.load('data/PF00004_h.npy')
couplings = np.load('data/PF00004_J.npy')


# Initialize the MSA generator
def generate_gray_areas(switch_times, end_time):
    """
    Generate gray areas (high-temperature regions) from switch times.

    Parameters:
    - switch_times (list): A sorted list of switch times.
    - end_time (int): The maximum time to consider.

    Returns:
    - List of (start, end) tuples representing gray areas.
    """
    gray_areas = []
    for i in range(1, len(switch_times), 2):  # Every other interval is high-temperature
        start = switch_times[i]
        end = switch_times[i + 1] if i + 1 < len(switch_times) else end_time
        gray_areas.append((start, end))
    # get sum of high-temperature regions and low-temperature regions
    high_temp = 0
    low_temp = 0
    for start, end in gray_areas:
        high_temp += end - start
    for i in range(0, len(switch_times), 2):
        start = switch_times[i]
        end = switch_times[i + 1] if i + 1 < len(switch_times) else end_time
        low_temp += end - start
    return gray_areas, high_temp, low_temp


def main():
    tau = 300
    initial_state = np.load("data/msa_70000_10000_0.5.npy")
    intermediate_states = [int(x) for x in np.arange(0, 4001, 10)]
    random.seed(41)
    switch_times = MSAGenerator._pre_sample_switch_times(tau=tau, max_mutations=intermediate_states[-1])
    np.save("data/switch_times.npy", switch_times)
    params = [(2, 1, 70000), (2, 0.5, 70000), (1, 0.5, 70000), ]
    for temphigh, templow, nbrseq in params:
        msagen = ge.MSAGenerator(fields, couplings, temp_high=temphigh, temp_low=templow, tau=tau,
                                 max_mutations=intermediate_states[-1], switch_times=switch_times)
        msa_final = msagen.generate_msa_phylo(initial_state, intermediate_states[-1],
                                              keep_intermediate_states=intermediate_states, n_workers=16)
        for i, nbrmut_ in enumerate(intermediate_states):
            filename = f"data/msas/telegraph_msa_phylo_{nbrseq}_{nbrmut_}_{templow}_{temphigh}_{tau}.npy"
            np.save(filename, msa_final[:, i])
            print(f"{filename} saved")


if __name__ == "__main__":
    main()

