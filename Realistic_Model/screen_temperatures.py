import numpy as np
import generation_sequence as ge
from multiprocessing import Pool, cpu_count

# Load the fields and couplings
fields = np.load('/home/malbrank/Documents/phylogeny-nico/data/Synthetic_data/bmDCA_parameters/PF00004_h.npy')
couplings = np.load('/home/malbrank/Documents/phylogeny-nico/data/Synthetic_data/bmDCA_parameters/PF00004_J.npy')

# Initialize the MSA generator
msagen = ge.Creation_MSA_Generation(fields, couplings)

# Define the parameters
nbrseq = 70000
temperatures = [0.2, 0.5, 0.75, 1, 2]
nbrmuts = [200, 500, 1000, 1500]


# Function to generate and save MSA
def generate_and_save_msa(temp, nbrmut):
    print(f"Generating msa_{nbrseq}_{nbrmut}_{temp}.npy")
    msa_final = msagen.msa_no_phylo_setT(nbrseq, nbrmut, temp)
    np.save(f"msa_{nbrseq}_{nbrmut}_{temp}.npy", msa_final)
    print(f"msa_{nbrseq}_{nbrmut}_{temp}.npy saved")


# Create a list of all parameter combinations
params = [(temp, nbrmut) for temp in temperatures for nbrmut in nbrmuts]

# Use multiprocessing to parallelize the computation
if __name__ == '__main__':
    with Pool(cpu_count()) as pool:
        pool.starmap(generate_and_save_msa, params)
