#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import h5py
import pdb
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 28})
from tqdm import tqdm

def opendata_eq(path):
    file = h5py.File(path,'r')
        
    matrix_chains = np.array(file['Chains'])
    templist = np.array(file['Temperatures'])
    file.close()
    
    return matrix_chains, templist


def Histogram(path):
    """
    Parameters
    ----------
    path : path to data

    Returns
    -------
    plots the histograms of magnetisation to find the critical temperature 
    for the phase transition.
    """
    sequences,tlist = opendata_eq(path)
    
    magnetisation = np.sum(sequences[:,:2000],axis = 2)
    nbrt,nbrseq = magnetisation.shape
    bins_m = np.linspace(np.min(magnetisation),np.max(magnetisation),30)
    
    fig,axs = plt.subplots(2,6)
    
    for idx,t in enumerate(tlist):
        idxc = idx%6
        idxl = int(idx/6)
        axs[idxl,idxc].hist(magnetisation[idx],bins =bins_m ,label = 'T = '+str(t),density = True)
        axs[idxl,idxc].legend(prop={'size': 11})
    
def Opendata(path):
    file = h5py.File(path,'r')
        
    matrix_chains = file['Chains']
    
    temperaturelist = file['TemperatureList']
    mutationslist = file['MutationList']
    para = file['Parameters']
    matrix_chains = np.array(matrix_chains)
    temperaturelist = list(temperaturelist)
    mutationslist = list(mutationslist)
    para = list(para)

        
    file.close()
    number_sequences = matrix_chains.shape[0]
    return matrix_chains, temperaturelist, mutationslist, number_sequences,para

def ComputeCorrelationMatrix2(mat, pseudocount):
    
    """
    Parameters
    ----------
    mat : MSA data.
    pseudocount : parameter between 0 and 1 to avoid inversion errors, when 
    computing the mean-field approximation during inference.

    Returns
    -------
    correlation_matrix : returns the correlation matrix with pseudocount
    correction

    """

    nbr_chains, nbr_spins = mat.shape
    mat = np.array(mat,ndmin = 2, dtype = np.float64)
    average_spin = np.average(mat, axis = 0)[:,None]
    
    directcorr = np.dot(mat.T, mat)

    directcorr *= np.true_divide(1, nbr_chains, dtype = np.float64)
    
    correlation_matrix = np.dot(1.0-pseudocount, directcorr) - np.dot(pow(1-pseudocount,2),np.outer(average_spin.T, average_spin)) + np.dot(pseudocount,np.identity(nbr_spins))
    
    return correlation_matrix

def Inference_MF(mat_corr, matrix_contacts,bl_abs,bl_apc,nbrcontacts):
    """
    Parameters
    ----------
    mat_corr : Correlation matrix computed with ComputeCorrelationMatrix2
    
    matrix_contacts : ground truth (the graph modelling contacts)
    
    bl_abs : take the absolute value of couplings 
    (equivalent to frob norm in the 2 states case)
    
    bl_apc : Make an APC correction.

    nbrcontacts : number of true contacts in the graph

    Returns
    -------
    TP fraction list at different thresholds of number of predicted contacts.

    """

    flag = True
    # inverse of the correlation matrix to get the couplings
    
    try:
        inferred_couplings = np.linalg.inv(mat_corr)
    except:
        flag = False
        
    if flag:
        if bl_abs:
            inferred_couplings = np.abs(inferred_couplings)
            
        if bl_apc:
            np.fill_diagonal(inferred_couplings,0)
            S = inferred_couplings.copy()
            inferred_couplings -= (np.mean(S, axis=1, keepdims=True) * np.mean(S, axis=0, keepdims=True)) / np.mean(S)
    
        
        
        TP = []
    
        # order the 2d array and find the index of the sorted values in the matrix
        if bl_abs:
            index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(-inferred_couplings, axis=None), inferred_couplings.shape)
        else:
            index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(inferred_couplings, axis=None), inferred_couplings.shape)
    
    
        idx_flip = list(index_sorted_array_x)
        idy_flip = list(index_sorted_array_y)
    
        # indirect_corr_second_order =[]
    
        FP = []
    
        TP_coords = []
        all_coords = []
        N = 0 
        number_pairs = []
    
        list_tp = []
        TP = 0
    
        list_tp_fraction_allpairs = []
    
    
        for x, y in zip(idx_flip, idy_flip):
    
            # just look at the elements above the diagonal as symmetric matrix
            # to not count twice each contact
            if y > x:
    
                N = N + 1
    
                number_pairs.append(N)

    
                if matrix_contacts[x,y] == 1:
                    TP = TP + 1
                    if N <= nbrcontacts:
                        TP_coords.append([x,y])
                else:
    
                    if N <= nbrcontacts:
                        FP.append([x,y])

    
                list_tp.append(TP)
    
                all_coords.append([x,y])
    
                list_tp_fraction_allpairs.append(TP/N)
    
        return list_tp_fraction_allpairs, FP
    
    else:
        mat = np.zeros(nbrcontacts)
        mat[:] = np.nan
        return mat,mat

def InferenceTemperatureMutations(matrix_chains,contactmatrix, number_spins, pseudocount,tlist,bl_abs,bl_apc):
    """
    Parameters
    ----------
    matrix_chains : MSA data.
    contactmatrix : ground truth (the graph modelling contacts)
    number_spins : length of protein (i.e. number of nodes in graph)

    pseudocount : parameter to avoid inversion issues during mf inference
    tlist : lists of temperatures (as function of MC time)
    of the telepgraph process
    bl_abs : bool to take the quivalent of frobenius norm in 2 states case.
    bl_apc : bool to apply APC correction.

    Returns
    -------
    inferencematrix : TP fraction (at Npred = number of contacts) at every MC 
    time point.
    """
    nbr_possiblecontacts = int(number_spins*(number_spins-1)/2)
    inferencematrix = np.zeros((len(tlist)+1))
    val,cts = np.unique(contactmatrix, return_counts=True)
    nbrcontacts = cts[val==1][0]
    
    for idxt in tqdm(range(0,len(tlist)+1)):
        matrix_correlation = ComputeCorrelationMatrix2(matrix_chains[:,idxt,:], pseudocount)

        mflist,_ = Inference_MF(matrix_correlation, contactmatrix,bl_abs,bl_apc,nbrcontacts)

        inferencematrix[idxt] = mflist[nbrcontacts-1]
 
    return inferencematrix


def InferenceDataset(path,savepath,bl_abs,bl_apc,pseudocount):
    """
    Parameters
    ----------
    path : Path to file where data is stored.
    savepath : path to save the TP fraction value.
    bl_abs : bool to take the equivalent of frobenius norm in 2 states case.
    bl_apc : bool to apply APC correction.
    pseudocount : parameter to avoid inversion issues during mf inference


    Returns
    -------
    None. It saves the values in the savepath folder.

    """
    contactmatrix = np.load('./data/contactmap/N200_p002.npy')

    number_spins = contactmatrix.shape[0]
    #if data generation started with an eq. MSA
    EQSTART = True

    matrixchains, tlist, mlist, number_sequences,para = Opendata(path)

    if EQSTART:
        nbr_flips_priorstart = para[-2]
        T_priorstart = para[-1]
    else: 
        nbr_flips_priorstart = 0; T_priorstart = 0

    
    inferencematrix = InferenceTemperatureMutations(matrixchains,contactmatrix, number_spins, pseudocount,tlist,bl_abs,bl_apc)
    
    np.save(savepath, inferencematrix)
    
    l = savepath.split('/')
    lprime = l[:-1]
    fdpath = '/'.join(lprime)
    np.save(fdpath+'/tlist.npy', tlist)
    np.save(fdpath+'/mlist.npy', mlist)


def manyreal(pathfolder,savefolder):
    """
        Parameters
    ----------
    pathfolder :path to folder where files are saved, different realisations.
    savefolder : folder where data is stored for different realisations.

    Returns
    -------
    None.

    """
    list_files = os.listdir(pathfolder)
    for idx,f in enumerate(list_files):
        print(idx)
        if f.startswith('.') == False:
            path = pathfolder + '/'+f
            os.mkdir(savefolder+'/real_{}'.format(idx))
            savepath = savefolder +'/real_{}/mf_a001_inference.npy'.format(idx)
            InferenceDataset(path, savepath, False,False,pseudocount)
            
    
if __name__ == '__main__':  
    pseudocount = 0.01
    
    #inference for a specific dataset (one realisation)
    # pathfile = './data/sequences/telegraph/T1_1_T2_15_tau1000/Nspins200_probagraph020_flips30000_Nchains2048_seed_13.h5'
    # savefile = './data/inference/telegraph/T1_1_T2_15_tau1000/mf_a001_inference.npy'
    # InferenceDataset(pathfile, savefile, False,False,pseudocount)
    
    #infer for a list of files with same parameters, with many realisations
    pathfolder = './data/sequences/telegraph/T1_1_T2_15_tau5/'
    savefolder = './data/inference/telegraph/T1_1_T2_15_tau5/'
    manyreal(pathfolder,savefolder)
    
    # Histogram('./data/sequences/equilibrium/T_4_5_N10k_200flips_TC/2023_11_20_16_55_17_Nspins200_probagraph0_02_flips200_Nchains10000_seed_10_filenbr0.h5')