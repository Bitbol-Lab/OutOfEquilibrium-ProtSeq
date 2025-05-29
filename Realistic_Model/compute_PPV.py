#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:50:35 2022

Compute PPV natural sequences -> from Umberto's code
"""

import pathlib
import warnings

import pandas as pd
from scipy.spatial.distance import pdist, squareform


from matplotlib import pyplot as plt
import h5py
import os

from Bio.PDB import *
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# from utils import *

import swalign

msa_data = {
    "PF00004": {
        "pdb_id": "4d81",
        "chain_id": "A",
        "pfam_seq": "ILLYGPPGCGKTMIAAAVANELDSEFIHVDAASIMSKWLGEAEKNVAKIFKTARELSKPAIIFIDELDALLASY-TSEVGGEARVRNQFLKEMDGLADKISKVYVIGATNKPWRLDEPFL-RRFQKRIYIT-"
    }
}

nonstandard_aa_substitutions = {
    "MSE": "M",
    "ADP": "A",
    "HG": "X",
    "ACE": "X",
    "ATP": "X",
    "CA": "X",
    "CLR": "X",
    "HOH": "X",
    "MG": "X",
    "NH2": "X",
    "NI": "X",
    "PNM": "X",
    "PO4": "X",
    "SO4": "X",
    "ZN": "X",
    "CL": "X",
    "ANP": "X",
    "GOL": "X",
    "CXT": "X",
    "CDL": "X",
    "PC1": "X",
    "LDM": "X",
    "BOG": "X",
    "GDM": "X",
    "FOS": "X",
    "V4O": "X",
    "FUC": "X",
    "NGA": "X",
    "B46": "X",
    "ACT": "X",
    "EDO": "X",
    "AF3": "X",
    "MK7": "X"
}


def indices_in_ref_and_query(alignment):
    i_ref = alignment.r_pos
    i_query = alignment.q_pos
    idxs_ref = []
    idxs_query = []
    for pair in alignment.cigar:
        if pair[1] == "M":
            j_ref = i_ref + pair[0]
            idxs_ref += list(range(i_ref, j_ref))
            i_ref = j_ref
            j_query = i_query + pair[0]
            idxs_query += list(range(i_query, j_query))
            i_query = j_query
        elif pair[1] == "D":
            i_ref += pair[0]

        elif pair[1] == "I":
            i_query += pair[0]

    return idxs_ref, idxs_query


def three_to_one(residue_name):
    return Polypeptide.index_to_one(Polypeptide.three_to_index(residue_name))


def to_one_letter_seq(chain):
    seq = ""
    for residue in chain.get_residues():
        three_letter_name = residue.get_resname()
        try:
            seq += three_to_one(three_letter_name)
        except KeyError:
            if three_letter_name in nonstandard_aa_substitutions:
                seq += nonstandard_aa_substitutions[three_letter_name]
            else:
                print(f"Non-standard amino acid {three_letter_name}, placed an X")
                seq += "X"
    return seq


def calc_Calpha_dist_matrix(chain, idx_subset):
    """Returns a matrix of C-alpha distances in a (subset of a) chain"""
    idx_subset = set(idx_subset)
    residue_coords = [residue["CA"].coord for i, residue in enumerate(chain.get_residues()) if i in idx_subset]

    return squareform(pdist(residue_coords))


def calc_min_dist_matrix(chain, idx_subset):
    """Returns a matrix of minimum distances between residues in a (subset of a) chain"""
    idx_subset = set(idx_subset)

    return squareform(
        np.array([min([atom_in_res_i - atom_in_res_j for atom_in_res_i in res_i for atom_in_res_j in res_j])
                  for i, res_i in enumerate(chain.get_residues()) if i in idx_subset
                  for j, res_j in enumerate(chain.get_residues()) if j > i and j in idx_subset])
    )


def top_n_contact_mask(scores_matrix, sequence_proximity_mask, n, return_flattened_scores=False):
    sq = squareform(scores_matrix, checks=False)
    sq_proximity = np.logical_not(squareform(sequence_proximity_mask, checks=False))
    sq[sq_proximity] = -np.inf
    if return_flattened_scores:
        return sq
    contact_mask = np.zeros(len(sq), dtype=bool)
    argsrt = np.argsort(sq)[::-1]
    contact_mask[argsrt[:n]] = True

    return squareform(contact_mask)


def contact_matrix_comparison(dist_matrix,
                              scores_matrix,
                              max_eucl_dist=8,
                              n_pred=None,
                              min_sequence_dist=5,
                              contact_mask=None,
                              return_data_for_roc=False):
    assert dist_matrix.shape == scores_matrix.shape
    n_residues = dist_matrix.shape[0]
    sequence_proximity_mask = np.abs(
        np.arange(n_residues) - np.arange(n_residues)[:, None]
    ) >= min_sequence_dist

    if contact_mask is None:
        eucl_dist_mask = dist_matrix < max_eucl_dist
        contact_mask = np.triu(np.logical_and(eucl_dist_mask, sequence_proximity_mask))

    if n_pred is None:
        n_contacts = int(np.sum(contact_mask))
        n_pred = n_contacts

    contact_mask_pred = top_n_contact_mask(scores_matrix,
                                           sequence_proximity_mask,
                                           n=n_pred,
                                           return_flattened_scores=return_data_for_roc)
    if return_data_for_roc:
        contact_scores = contact_mask_pred
        return contact_scores, squareform(contact_mask, checks=False)

    contact_mask_pred = np.tril(contact_mask_pred)

    contact_mask_true_pos = np.logical_and(contact_mask_pred.T, contact_mask).T
    contact_mask_false_pos = np.logical_xor(contact_mask_true_pos, contact_mask_pred)
    contact_mask_false_neg = np.logical_and(np.logical_not(contact_mask_pred).T, contact_mask)

    full_matrix = contact_mask_true_pos.T * 1. + contact_mask_false_neg * 3 + contact_mask_true_pos * 1. + contact_mask_false_pos * 4.
    full_matrix[full_matrix == 0] = np.nan

    return full_matrix


def plot_contact_matrices(dist_matrix,
                          scores_matrix,
                          max_eucl_dist=8,
                          n_pred=None,
                          min_sequence_dist=5,
                          title=None):
    full_matrix = contact_matrix_comparison(dist_matrix,
                                            scores_matrix,
                                            max_eucl_dist=max_eucl_dist,
                                            n_pred=n_pred,
                                            min_sequence_dist=min_sequence_dist)

    plt.figure(figsize=(10, 10))
    plt.matshow(full_matrix, fignum=1, cmap="viridis")

    plt.colorbar()
    if title is not None:
        plt.title(title)

    plt.show()


def zero_sum_gauge_frob_scores(J2, apc=True):
    """Compute a score for contacts between sites by first passing to a zero-sum gauge, then computing a Frobenius
    norm, and finally applying the average product correction (APC) if desired."""
    # Pass to zero-sum gauge
    J2_zs = J2.copy()
    J2_zs -= np.mean(J2, axis=2, keepdims=True)
    J2_zs -= np.mean(J2, axis=3, keepdims=True)
    J2_zs += np.mean(J2, axis=(2, 3), keepdims=True)
    nbrpos, _, _, _ = J2_zs.shape
    # Frobenius norm
    S_frob = np.linalg.norm(J2_zs, axis=(2, 3), ord='fro')
    # S_frob = np.linalg.norm(J2_zs[:,:,:-1,:-1],axis=(2, 3), ord = 'fro')
    # Average-product correction
    # print('diag el',S_frob[20,20])
    S = S_frob.copy()
    if apc:
        S -= (np.mean(S_frob, axis=1, keepdims=True) * np.mean(S_frob, axis=0, keepdims=True)) / np.mean(S_frob)
        # S -= (np.sum(S_frob, axis=1, keepdims=True) * np.sum(S_frob, axis=0, keepdims=True)) / (np.sum(S_frob)*(1-1/nbrpos))

    return S


def convertjltonpy_scores(path, filename):
    file = h5py.File(path + filename, 'r')
    couplingsplm = file['couplings'].value
    file.close()
    scores_plm = zero_sum_gauge_frob_scores(couplingsplm, apc=True)
    np.save(path + 'scores_plm.npy', scores_plm)


def ComputePPV_OneTimePt_PLMDCA(scores, scoresbm, dists, idx_subset):
    scores = scores[np.asarray(idx_subset)][:, np.asarray(idx_subset)]

    # dists = ComputeContactMaskDistMat(msa_name,path_tociffolder)

    pfam_length = len(idx_subset)
    n_pred = 2 * pfam_length

    max_eucl_dist = 8
    min_sequence_dist = 5
    contact_matrix_kwargs = {"max_eucl_dist": max_eucl_dist,
                             "min_sequence_dist": min_sequence_dist}

    cm = contact_matrix_comparison(
        dists,
        scoresbm,
        **contact_matrix_kwargs,
        n_pred=2 * pfam_length)

    cm = np.tril(np.isfinite(cm)).T

    full_matrix = contact_matrix_comparison(dists,
                                            scores,
                                            contact_mask=cm,
                                            min_sequence_dist=min_sequence_dist,
                                            n_pred=n_pred)

    ppv = np.sum(np.tril(full_matrix == 1)) / np.sum(np.tril(np.isfinite(full_matrix)))

    return ppv


def Build_Groundtruth_BMstructure(PATH_PDB, PATH_BMPARAM):
    match = 2
    mismatch = -2
    gap_penalty = -2

    scoring = swalign.IdentityScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring, gap_penalty=gap_penalty)

    idxs_chains = {}
    idxs_pfam_seqs = {}
    dist_mat = {}

    PDB_DIR = pathlib.Path(PATH_PDB)
    if not PDB_DIR.exists():
        os.mkdir(PDB_DIR)

    for msa_name in msa_data:
        print(msa_name)
        pdb_id = msa_data[msa_name]["pdb_id"]
        chain_id = msa_data[msa_name]["chain_id"]

        # Download and parse structure
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id,
                               pdir=PDB_DIR,
                               file_format="mmCif")
        pdb_parser = MMCIFParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PDBConstructionWarning)
            chain = pdb_parser.get_structure(pdb_id, "{}/{}.cif".format(PDB_DIR, pdb_id))[0][chain_id]
        # Convert to one-letter encoding
        pdb_seq = to_one_letter_seq(chain)
        pfam_seq = msa_data[msa_name]["pfam_seq"]
        # print(f"Ref: {pdb_seq = }".fo)
        # print(f"Query: {pfam_seq = }")

        # Align PDB sequence with PFAM sequence
        alignment = sw.align(pdb_seq, pfam_seq)
        alignment.dump()
        # Store matching indices and PDB distance matrix for matching indices
        idxs_chain, idxs_pfam_seq = indices_in_ref_and_query(alignment)
        idxs_chains[msa_name] = idxs_chain
        idxs_pfam_seqs[msa_name] = idxs_pfam_seq
        dist_mat[msa_name] = calc_min_dist_matrix(chain, idxs_chain)
        print()
    bmDCA_PARAMETERS_DIR = pathlib.Path(PATH_BMPARAM)

    bmDCA_scores = {}
    for pfam_family in msa_data:
        bmDCA_scores[pfam_family] = {}
        J = np.load(bmDCA_PARAMETERS_DIR / f"{pfam_family}_J.npy")
        idx_subset = np.asarray(idxs_pfam_seqs[pfam_family])  # Restrict to sites matching with the PDB
        bmDCA_scores[pfam_family] = zero_sum_gauge_frob_scores(J)[idx_subset, :][:, idx_subset]  # Use APC (default)

    return bmDCA_scores, idxs_pfam_seqs, dist_mat


def LoadJLD_File(path_jldfile):
    file = h5py.File(path_jldfile, 'r')
    couplingsplm = np.array(file['couplings'])
    file.close()
    scores_plm = zero_sum_gauge_frob_scores(couplingsplm, apc=True)
    return scores_plm


PATH_PDB = "data/pdb/"
PATH_BMPARAM = "data/"
PATH_JLDFOLDER = 'data/dca/'
pfam_family = 'PF00004'

filenames = list(os.listdir(PATH_JLDFOLDER))

import multiprocessing as mp

def process_file(filename):
    """Process a single file to compute PPV."""
    bmDCA_scores, idxs_pfam_seqs, dist_mat = Build_Groundtruth_BMstructure(PATH_PDB, PATH_BMPARAM)

    dists = dist_mat[pfam_family]
    idx_subset = np.asarray(idxs_pfam_seqs[pfam_family])

    scores = LoadJLD_File(os.path.join(PATH_JLDFOLDER, filename))
    ppv = ComputePPV_OneTimePt_PLMDCA(scores, bmDCA_scores[pfam_family], dists, idx_subset)

    T_low, T_high, T_step, tau, n_sequences = filename.split('_')[3:8]

    return {
        'filename': filename,
        'ppv': ppv,
        'T_low': T_low,
        'T_high': T_high,
        'T_step': T_step,
        'tau': tau,
        'n_sequences': n_sequences
    }

if __name__ == "__main__":
    num_workers = min(mp.cpu_count(), len(filenames))  # Use available CPU cores, but not more than needed
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_file, filenames)

    df = pd.DataFrame(results)
    df.to_csv('data/results.csv', index=False)
    print(df)
