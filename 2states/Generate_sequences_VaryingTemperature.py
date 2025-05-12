#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# master
import sys
import os
import numpy as np
import random
import multiprocessing as mp
from joblib import Parallel, delayed
import time
# set seed
seed = 13


import matplotlib.pyplot as plt
from math import exp, expm1
from datetime import datetime, date
import h5py
startTime = datetime.now()
today = date.today()

from tqdm import tqdm
from numba import jit
import pdb

def Generate_Graph(Nspin, p):

    mat = np.zeros((Nspin, Nspin))


    for i in range(0,Nspin):
        for j in range(i+1, Nspin):

            randnumber = random.uniform(0,1)

            if randnumber < p:
                mat[i,j] = 1
            else:
                mat[i,j] = 0

    return mat

def Openseq(patheq,temp,idx):

    file = h5py.File(patheq,'r')
        
    matrix_chains = np.array(file['Chains'])
    templist = np.array(file['Temperatures'])
    file.close()
    return matrix_chains[templist==temp,idx,:][0,:]

@jit(nopython=True)
def Generate_chain(N_spin):

    N_spin = int(N_spin)

    chain = np.zeros(N_spin)

    for k in range(0,N_spin):
        chain[k] = random.randrange(-1,2,2)

    return chain

@jit(nopython=True)
def Compute_Energy(mat_contact, chain):
    return (-1)*np.dot(chain,(mat_contact.dot(chain)))

@jit(nopython=True)
def Compute_magnetization(chain):

    return  np.sum(chain)

@jit(nopython=True)
def Flip_spin(chain, nbr_spin):
    #choose at random one lattice site == index in the chain
    rand_site1 = random.randrange(0,nbr_spin,1)

    #make spin flip at location rand_site
    chain[rand_site1] = chain[rand_site1]*-1


    return chain

@jit(nopython=True)
def NewConfig(mat, chain, nbr_spin):
    #calculate the new configurations
    new_chain = Flip_spin(chain, nbr_spin)
    new_energy_configuration = Compute_Energy(mat, new_chain)
    return new_chain, new_energy_configuration

@jit(nopython=True)
def EvolutionFixedTemperature(matcontact, chain, energ, nbr_spin, thr_accepted_flips, Temp):
    """
    Parameters
    ----------
    matcontact : array containing contacts between nodes (vertices between nodes)
    chain : sequence at equilibrium (before evolution)
        
    energ : energy of the sequence (hamiltonian)
        
    nbr_spin : length of protein (number of nodes)
      
    thr_accepted_flips : Numnber of accepted flips the current (t1 or t2) phase
        
    Temp : temperature either t1 or t2 (depending in which phase of telegraph)
       

    Returns
    -------
    chain : sequence after it has evolved.
    energ : energy of the sequence.

    """

    curr_accepted_flips = 0

    while(curr_accepted_flips < thr_accepted_flips):

        
        newchain = chain.copy()
        rand_site1 = random.randrange(0,nbr_spin,1)
        newchain[rand_site1] = newchain[rand_site1]*-1
        new_energ_config = (-1)*np.dot(newchain,(matcontact.dot(newchain)))

        if new_energ_config < energ:

            energ = new_energ_config
            chain = newchain.copy()
            curr_accepted_flips = curr_accepted_flips + 1
 

        else:

            boltzman_prob = exp((energ-new_energ_config)/Temp)
            
            unif_number = random.uniform(0,1)

            if unif_number < boltzman_prob:

                energ = new_energ_config
                chain = newchain.copy()

                curr_accepted_flips = curr_accepted_flips + 1

    
    return chain, energ

def EvolvutionVaryingTemperature(temperature_list, interval_mut, number_spins, EQSTART,idxeq,T_PRIOR_START,matcontact):
    """
    
    Parameters
    ----------
    temperature_list : list of t1 and t2 phases
    interval_mut : number of accepted mutations between two saved time pts.
        
    number_spins : length of proteins (number of nodes in the graph)

    EQSTART : bool to start from equilibrium sequences
        
    idxeq : randomly selects one starting sequence among possible ones.
    
    T_PRIOR_START : temperature of the equilibrated sequence.
        
    matcontact : array containing the contacts between nodes (vertices of the graph)

    Returns
    -------
    matrix_chains : array containing sequence at different time points

    """
    chain = Generate_chain(number_spins)
    energ = Compute_Energy(matcontact, chain)
    if EQSTART:
        patheq = './data/sequences/equilibrium/T1_7_N2048_300flips/Nspins200_probagraph0_02_flips300_Nchains2048_seed_10_filenbr0.h5'
        chain = Openseq(patheq,T_PRIOR_START,idxeq).astype(np.float64)

        energ = Compute_Energy(matcontact, chain)

    matrix_chains = np.zeros((len(temperature_list)+1,number_spins))
    matrix_chains[0,:] = chain
    
    
    for idxT, t in enumerate(temperature_list):
        chain,energ = EvolutionFixedTemperature(matcontact, chain, energ, number_spins, interval_mut, t)
        matrix_chains[idxT+1,:] = chain

    return matrix_chains
        
def Telegraph_Process_FixedMCTime(tau,t1,t2,number_mut):
    """
    

    Parameters
    ----------
    tau : time parameter for the exponential distribution for the telegraph process
    t1 : Temperature T1
    t2 : Temperature T2
        
    number_mut : total number of accepted flips (=mutations for the data generation

    Returns
    -------
    number_flips : for each t1 and t2 phase it gives the number of flips in each phase.
    temperature_list : gives the temperature in each phase (as a function of time)

    """
    number_flips = []
    temperature_list =[]
    counter_nmut = 0
    
    while counter_nmut < number_mut:
    
        off_times = np.random.exponential(scale=tau)
        off_times = np.round(off_times,decimals = 0)
        off_times = off_times.astype(int)
        if (counter_nmut + off_times) > number_mut:
            tmp = (counter_nmut + off_times) - number_mut
            number_flips.append(off_times-tmp)
        else:
            number_flips.append(off_times)
            
        temperature_list.append(t1)
        counter_nmut = counter_nmut + off_times
        
        if counter_nmut < number_mut:
            on_times = np.random.exponential(scale=tau)
            on_times = np.round(on_times,decimals = 0)
            on_times = on_times.astype(int)
            if counter_nmut + on_times > number_mut:
                tmp = (counter_nmut + on_times) - number_mut 
                number_flips.append(on_times-tmp)
            else:
                number_flips.append(on_times)

            temperature_list.append(t2)
            counter_nmut = counter_nmut + on_times

    assert np.sum(number_flips) == number_mut

    
    return number_flips,temperature_list



def Run_DataGeneration(sd,filenbr):
    """
    Parameters
    ----------
    sd : seed to set random 
    filenbr : number of realisation (for each realisation this code procudes one file)

    Returns
    -------
    saves a file with the sequences as a function of the MC time.

    """
    start_time = time.time()
    random.seed(sd)
    number_spins = 200
    number_sequences = 2048
    proba_graph = 0.02
    tau = 5
    matcontact = np.load('./data/contactmap/N200_p002.npy')
    #matcontact =  Generate_Graph(number_spins, proba_graph)
    t1 = 1
    t2 = 15
    
    #The number of accepted mutations in total.
    number_mut = 5000
    
    #tau appears twice here, if one would generate an asymetric telegraph process
    para = [number_spins, proba_graph, number_sequences, sd,tau,tau,t1,t2,number_mut]
    
    #generate the telegraph process, define the temperatures as a function
    #of the Monte Carlo time; number of flips for each t1 phase and t2phase. 
    numberflips,temperature_list = Telegraph_Process_FixedMCTime(tau,t1,t2,number_mut)
    
    #this variable sets the interval at which the state of the sequence is saved.
    interval_mut = 10
    
    #the total number of saved time points 
    nbr_points = int(np.sum(numberflips)/interval_mut)
    
    #defines the MC time at which sequences are saved and corresponding temperature
    list_tpoints = []
    newl_mtpoints = [0]
    for idxm,m in enumerate(numberflips):
    
        ltmp = list(temperature_list[idxm]*np.ones(int(m/interval_mut)))    
        list_tpoints = list_tpoints + ltmp
        newl_mtpoints = newl_mtpoints+list(np.arange(newl_mtpoints[-1]+1,newl_mtpoints[-1]+1+int(m/interval_mut)))

    
    date = today.strftime("%Y_%m_%d_")
    hour = startTime.strftime('%H_%M_%S_')
    path = './data/sequences/telegraph/T1_{}_T2_{}_tau{}/'.format(t1,t2,tau)
    if not(os.path.exists(path)):
        os.mkdir(path)
    filename = path+date+hour+'Nspins{}_probagraph0{}_flips{}_Nchains{}_seed_{}_filenbr{}.h5'.format(number_spins, int(proba_graph*1000), np.sum(numberflips), number_sequences,sd,filenbr)
    file = h5py.File(filename, 'w')
    
    file.close()
    
    #When starting from sequences at equilibrium
    EQSTART = True
    FLIPS_PRIOR_START = 3000
    T_PRIOR_START = 1
    
    if EQSTART:
        para.extend([FLIPS_PRIOR_START,T_PRIOR_START])
        idxs_eq = np.arange(number_sequences)
        random.shuffle(idxs_eq)
        

    
    file = h5py.File(filename, 'r+')
    file.create_dataset('Parameters', data = np.array(para))
    file.create_dataset('Matrix_contact', data = matcontact)
    file.create_dataset('Temperatures', data = np.array(temperature_list))
    file.create_dataset('Mutations', data = np.array(numberflips))
    file.create_dataset('TemperatureList', data = np.array(list_tpoints))
    file.create_dataset('MutationList', data = np.array(newl_mtpoints))
    file.close()
    

    #Generate and evolve sequences
    matrix_chains = np.zeros((number_sequences,len(list_tpoints)+1,number_spins))
    
    for seq in tqdm(range(0,number_sequences)):
        matchains = EvolvutionVaryingTemperature(list_tpoints, interval_mut, number_spins, EQSTART,idxs_eq[seq],T_PRIOR_START,matcontact)
        matrix_chains[seq,:,:] = matchains

        
        
    file = h5py.File(filename, 'r+')

    file.create_dataset('Chains', data = matrix_chains, dtype = np.int8,compression='gzip', compression_opts=9)
    file.close()
     

if __name__ == '__main__':  
    if mp.get_start_method() == "spawn":
        import sys
        sys.exit(0)
    
    backend = 'multiprocessing'
    Parallel(n_jobs=1)(delayed(Run_DataGeneration)(filenbr+2,filenbr) for filenbr in range(0,2))
        