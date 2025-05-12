#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic sequences with binary variables, cluster algorithm
"""

import numpy as np
import random
from math import exp, expm1
from datetime import datetime, date
import h5py
import time
from numba.typed import List
from numba import njit,prange,jit
from tqdm import tqdm

startTime = datetime.now()
today = date.today()

def GenerateChain(nbrspin):
    # generate a chain of spins whose value is randomly set between 1 & -1
    
    tmp_chain = np.zeros(nbrspin)
    
    for j in range(0,nbrspin):
        tmp_chain[j] = random.randrange(-1,2,2)
        
    return tmp_chain

def GenerateMatcontact(nbrspin, p):
    """
    Generates a random erdos-renyi graoh

    Parameters
    ----------
    nbrspin : int
        number of nodes in the graph
    p : float
        probability of connecting two nodes

    Returns
    -------
    tmp_matcontact : array
        returns the adjacency matrix of the graph

    """
    tmp_matcontact = np.zeros([nbrspin, nbrspin])
    
    for k in range(0, nbrspin):
        for j in range(k+1, nbrspin):
            randvalue = random.uniform(0,1)
            if  randvalue < p:
                tmp_matcontact[k,j] = 1
                
    return tmp_matcontact

def FindNeighbors(edge_site, tmp_matcontact, nbrspin):
    """
    Computes every neighbouring sites of the edge site

    Parameters
    ----------
    edge_site : int
        site number in graph
    tmp_matcontact : 2darray
        adjacency matrix of the graph
    nbrspin : int
        number of sites/nodes in graph

    Returns
    -------
    list_neighbors_site : List
        provides a list of ints corresponding to the neighbouring sites

    """

    
    list_neighbors_site = []
    
    for l in range(0,edge_site):
        if tmp_matcontact[l, edge_site] == 1:
            list_neighbors_site.append(l)
            
    for k in range(edge_site+1, nbrspin):
        if tmp_matcontact[edge_site,k] == 1:
            list_neighbors_site.append(k)
            
    return list_neighbors_site

  
@jit(nopython = True)
def ClusterIsing(nbrspin, list_neighbours, tmp_chain, p):
    """
    Algorithm to make one flip in a sequence

    Parameters
    ----------
    nbrspin : int
        number of sites/nodes in graph.
    list_neighbours : List
        List of neighbours, for each key (= one site) it stores a list of neighbours
    tmp_chain : 1d array
        starting sequence .
    p : float
        probability accepting a site in cluster.

    Returns
    -------
    tmp_chain : 1d array
        sequence after cluster flip.

    """
    # select randomly the starting site from which the cluster is built
    start_site = random.randint(0, nbrspin-1)
    # create the list which contain all sites in the cluster
    cluster = []
    # create the list which contains the sites newly added to explore the 
    # neighbors of them to expand the cluster
    pocket = []
    
    # contains all the sites forming the cluster
    cluster.append(start_site)
    
    # contains all the sites which are on the edge of the cluster
    pocket.append(start_site)
    
    # define a variable containing the spin orientation of the cluster 1 or -1
    spin_orientation = tmp_chain[start_site] 
 
    # as long as the cluster is not empty the loop continues
    while pocket:

        # choose one element in the pocket list containing edge sites
        pocket_site = np.random.choice(np.array(pocket))
        # pocket_site = random.shuffle(pocket)[0]
        # look into the dictionary containing all the neighbors of pocket_site
        for site in list_neighbours[pocket_site]:
            
            # test if the site is not already in the cluster and also 
            # if the site has the same spin orientation than the 

            if (site not in cluster) and (tmp_chain[site] == spin_orientation) and (random.uniform(0.0,1.0)<p):
                 

                    # added a new site, so explore the neighbors
                    # so add it to pocket
                    pocket.append(site)
                    # cluster is growing, added a new site
                    cluster.append(site)

        pocket.remove(pocket_site)

    for elem in cluster:
            tmp_chain[elem] = tmp_chain[elem]*-1
    
    # print(tmp_chain, tmp_chain_final)
    
    # return tmp_chain_final
    return tmp_chain

# @njit(parallel = True)
def FctToParalllel(Temperature,number_spin, list_neighbours,number_chains,number_flips):
    """
    parallelise the flips of the cluster of sites in the sequence

    Parameters
    ----------
    Temperature : float
        Temperature of the MC algorithm, "selection strength".
    number_spin : int
        number of sites/nodes.
    list_neighbours : List
        List of neighbouring sites for each site.
    number_chains : int
        number of sequences to generate.
    number_flips : int
        Number of cluster flips to do, chosen such that sequences are at eq.

    Returns
    -------
    final_chains : 3d arry
        matrix where for each temperature (axis 0) there are number_chains
        (axis 1) sequences of length number_spin (axis 2)
        at equilibrium that are stored

    """
    final_chains = np.zeros([len(Temperature),number_chains,number_spin], dtype = np.int8)
    
    for index_temp, T in enumerate(Temperature):

        beta = 1/T
        probability = 1.0 - exp(-2.0 *beta)
        

        
        for i in prange(number_chains):
            # generate a new chain
            chain = GenerateChain(number_spin)
            

            
            tau = 0
            # start= time.time()
            while(tau < number_flips):

                chain = ClusterIsing(number_spin, list_neighbours, chain.copy(), probability)

                tau = tau+1

            final_chains[index_temp,i,:] = chain.copy()

    
    return final_chains


@njit(parallel = True)
def FctToParalllel_savetimepoints(Temperature,number_spin, list_neighbours,number_chains,number_flips):
    """
    parallelise the flips of the cluster of sites in the sequence

    Parameters
    ----------
    Temperature : float
        Temperature of the MC algorithm, "selection strength".
    number_spin : int
        number of sites/nodes.
    list_neighbours : List
        List of neighbouring sites for each site.
    number_chains : int
        number of sequences to generate.
    number_flips : int
        Number of cluster flips to do, chosen such that sequences are at eq.

    Returns
    -------
    final_chains : 3d arry
        matrix where for each temperature (axis 0) there are number_chains
        (axis 1) sequences of length number_spin (axis 2)
        at equilibrium that are stored

    """
    final_chains = np.zeros([len(Temperature),number_chains,number_flips+1,number_spin], dtype = np.int8)
    
    for index_temp, T in enumerate(Temperature):

        beta = 1/T
        probability = 1.0 - exp(-2.0 *beta)
        

        
        for i in prange(number_chains):
            # generate a new chain
            chain = GenerateChain(number_spin)
            

            
            tau = 0
            # start= time.time()
            final_chains[index_temp,i,tau,:] = chain.copy()

            while(tau < number_flips):

                chain = ClusterIsing(number_spin, list_neighbours, chain.copy(), probability)

                tau = tau+1

                final_chains[index_temp,i,tau,:] = chain.copy()

    
    return final_chains

def ConvertDictList(Neighbors_dict):
    """
    function to convert the neighbouring dict into a list in order to use numba
    """
    list_neighbors = List()
    
    for key in Neighbors_dict.keys():
        tmplist = List([-1]+Neighbors_dict[key])
        list_neighbors.append(tmplist[1:])
    return list_neighbors




sd = 76
random.seed(sd)
#number of nodes in graph
number_spin = 200
#probability of connecting nodes in the graph
prob_graph = 0.02

#to generate a new graph uncomment the following line 
# matcontact = GenerateMatcontact(number_spin, prob_graph)
matcontact = np.load('./data/contactmap/N200_p002.npy')

# create a dictionary where each site contains a list of its neighbors
Neighbors_dict = {site: FindNeighbors(site, matcontact, number_spin) for site in range(0,number_spin)}
# number of different chains to take then average.
number_chains = 2048
# number of cluster flips, for each flip a new cluster is built. (not all of the same size)
number_flips = 300
# temperature list
# Temperature = list(np.linspace(1,2,2))
Temperature = [1,2,3,4,5,15]

number_averages = 100

list_neighbours = ConvertDictList(Neighbors_dict)
Temperature_temp = np.array(Temperature)

bl_savetimepts = False

for nbf in tqdm(range(0,number_averages)):

    startTime = datetime.now()
    today = date.today()
    # create an hdf file to save the data 
    date1 = today.strftime("%Y_%m_%d_")
    hour = startTime.strftime('%H_%M_%S_')
    path_save = './data/sequences/equilibrium/T_1_2_3_4_5_15_300flips_2048chains_eq/'
    filename = path_save+date1+hour+'Nspins{}_probagraph{}_flips{}_Nchains{}_seed_{}_filenbr{}.h5'.format(number_spin,str(prob_graph).replace('.','_'), number_flips, number_chains, sd, nbf)
    file = h5py.File(filename, 'w')
    file.close()
    
    
    parameters = 'Parameters: Number of spins = {}, Number of chains = {}, Number of flips = {}, probability graph = {}, Temperature(s) = {}, seed = {}'.format(number_spin, number_chains,number_flips, prob_graph,Temperature, sd)
    para = [number_spin, prob_graph, number_chains, number_flips, number_averages]
    
    file = h5py.File(filename, 'r+')
    
    para_arr = np.array(para)
    
    file.create_dataset('Parameters', data = para_arr[:])
    file.create_dataset('Matrix_contact', data = matcontact)
    file.create_dataset('Temperatures', data = Temperature_temp)
    
    file.close()
    
    if bl_savetimepts:  
        fchains = FctToParalllel_savetimepoints(Temperature,number_spin, list_neighbours,number_chains,number_flips)
    else:
        fchains = FctToParalllel(Temperature,number_spin, list_neighbours,number_chains,number_flips)
   
    file = h5py.File(filename, 'r+')
    file.create_dataset('Chains', data = fchains, dtype = np.int8,compression='gzip', compression_opts=9)

    file.close()



