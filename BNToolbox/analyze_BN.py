#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 09:25:40 2025
Last Edited on Thu Aug 14 2025

@author: Claus Kadelka, Benjamin Coberly
"""

##Imports

import numpy as np
import itertools
import networkx as nx
import random
import math
from collections import defaultdict

import utils
import analyze_BF
from boolean_network import BooleanNetwork

##Key functions: compute/simulate network dynamics
def update_single_node(f, states_regulators):
    """
    Update the state of a single node.

    The new state is obtained by applying the Boolean function f to the states of its regulators.
    The regulator states are converted to a decimal index using utils.bin2dec.

    Parameters:
        f (list or np.array): Boolean function (truth table) for the node.
        states_regulators (list or np.array): Binary vector representing the states of the node's regulators.

    Returns:
        int: Updated state of the node (0 or 1).
    """
    return f[utils.bin2dec(states_regulators)]


def update_network_synchronously(BN, X):
    """
    Perform a synchronous update of a Boolean network.

    Each node's new state is determined by applying its Boolean function to the current states of its regulators.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        X (list or np.array): Current state vector of the network.

    Returns:
        np.array: New state vector after the update.
    """
    return BN.update_network_synchronously(X)


def update_network_synchronously_many_times(BN, X, n_steps):
    """
    Update the state of a Boolean network sychronously multiple time steps.

    Starting from the initial state, the network is updated synchronously n_steps times using the update_network_synchronously function.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        X (list or np.array): Initial state vector of the network.
        n_steps (int): Number of update iterations to perform.

    Returns:
        np.array: Final state vector after n_steps updates.
    """
    return BN.update_network_synchronously_many_times(X, n_steps)


def update_network_SDDS(BN, X, P):
    """
    Perform a stochastic update (SDDS) on a Boolean network.

    For each node, the next state is computed as nextstep = F[i] evaluated on the current states of its regulators.
    If nextstep > X[i], the node is activated with probability P[i,0]; if nextstep < X[i],
    the node is degraded with probability P[i,1]. Otherwise, the state remains unchanged.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        X (list or np.array): Current state vector.
        P (np.array): A len(F)×2 array of probabilities; for each node i, P[i,0] is the activation probability,
                      and P[i,1] is the degradation probability.

    Returns:
        np.array: Updated state vector after applying the stochastic update.
        
    References:
        
    """
    return BN.update_network_SDDS(X, P)


def get_steady_states_asynchronous(BN, nsim=500, EXACT=False, initial_sample_points=[], search_depth=50, SEED=-1, DEBUG=False):
    """
    Compute the steady states of a Boolean network under asynchronous updates.

    This function simulates asynchronous updates of a Boolean network (with N nodes)
    for a given number of initial conditions (nsim). For each initial state, the network
    is updated asynchronously until a steady state (or attractor) is reached or until a maximum
    search depth is exceeded. The simulation can be performed either approximately (by sampling nsim
    random initial conditions) or exactly (by iterating over the entire state space when EXACT=True).

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        nsim (int, optional): Number of initial conditions to simulate (default is 500).
        EXACT (bool, optional): If True, iterate over the entire state space and guarantee finding all steady states (2^N initial conditions);
                                otherwise, use nsim random initial conditions. (Default is False.)
        initial_sample_points (list, optional): List of initial states (as binary vectors) to use.
                                                  If provided and EXACT is False, these override random sampling.
        search_depth (int, optional): Maximum number of asynchronous update iterations to attempt per simulation.
        SEED (int, optional): Random seed. If SEED is -1, a random seed is generated.
        DEBUG (bool, optional): If True, print debugging information during simulation.

    Returns:
        tuple: A tuple containing:
            - steady_states (list): List of steady state values (in decimal form) found.
            - number_of_steady_states (int): Total number of unique steady states.
            - basin_sizes (list): List of counts showing how many initial conditions converged to each steady state.
            - steady_state_dict (dict): Dictionary mapping a steady state (in decimal) to its index in the steady_states list.
            - dictF (dict): Dictionary caching state transitions. Keys are tuples (xdec, i) and values are the updated state.
            - SEED (int): The random seed used for the simulation.
            - initial_sample_points (list): The list of initial sample points used (if provided) or those generated during simulation.
    """
    return BN.get_stead_states_asynchronous(nsim, EXACT, initial_sample_points, search_depth, SEED, DEBUG)


def get_steady_states_asynchronous_given_one_initial_condition(BN, nsim=500, stochastic_weights=[], initial_condition=0, search_depth=50, SEED=-1, DEBUG=False):
    """
    Determine the steady states reachable from one initial condition using weighted asynchronous updates.

    This function is similar to steady_states_asynchronous_given_one_IC but allows the update order
    to be influenced by provided stochastic weights (one per node). A weight vector (of length N) may be provided,
    and if given, it is normalized and used to bias the random permutation of node update order.
    
    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        nsim (int, optional): Number of simulation runs (default is 500).
        stochastic_weights (list, optional): List of stochastic weights (one per node) used to bias update order.
                                               If empty, uniform random order is used.
        initial_condition (int or list/np.array, optional): The initial state for all simulations. If an integer, 
                                                           it is converted to a binary vector. Default is 0.
        search_depth (int, optional): Maximum number of asynchronous update iterations per simulation (default is 50).
        SEED (int, optional): Random seed. If -1, a random seed is generated (default is -1).
        DEBUG (bool, optional): If True, print debugging information (default is False).

    Returns:
        tuple: A tuple containing:
            - steady_states (list): List of steady state values (in decimal form) reached.
            - number_of_steady_states (int): Total number of unique steady states.
            - basin_sizes (list): List of counts of how many simulations reached each steady state.
            - transient_times (list): List of lists with transient times (number of updates) for each steady state.
            - steady_state_dict (dict): Dictionary mapping a steady state (in decimal) to its index.
            - dictF (dict): Dictionary caching computed state transitions.
            - SEED (int): The random seed used.
            - queues (list): List of state update queues (the sequence of states encountered) for each simulation.
    """
    return BN.get_steady_states_aysnchronous_given_one_initial_condition(nsim, stochastic_weights, initial_condition, search_depth, SEED, DEBUG)


def get_attractors_synchronous(BN, nsim=500, initial_sample_points=[], n_steps_timeout=100000,
                               INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS=True):
    """
    Compute the number of attractors in a Boolean network using an alternative (v2) approach.

    This version is optimized for networks with longer average path lengths. For each of nb initial conditions,
    the network is updated synchronously until an attractor is reached or until n_steps_timeout is exceeded.
    The function returns the attractors found, their basin sizes, a mapping of states to attractors,
    the set of initial sample points used, the explored state space, and the number of simulations that timed out.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        nsim (int, optional): Number of initial conditions to simulate (default is 500).
        initial_sample_points (list, optional): List of initial states (in decimal) to use.
        n_steps_timeout (int, optional): Maximum number of update steps allowed per simulation (default 100000).
        INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS (bool, optional): If True, initial_sample_points are provided as binary vectors;
                                                                  if False, they are given as decimal numbers. Default is True.

    Returns:
        tuple: A tuple containing:
            - attractors (list): List of attractors (each as a list of states in the attractor cycle).
            - number_of_attractors (int): Total number of unique attractors found.
            - basin_sizes (list): List of counts for each attractor.
            - attr_dict (dict): Dictionary mapping states (in decimal) to the index of their attractor.
            - initial_sample_points (list): The initial sample points used (if provided, they are returned; otherwise, the generated points).
            - state_space (list): List of states (in decimal) encountered after one update from initial_sample_points.
            - n_timeout (int): Number of simulations that timed out before reaching an attractor.
    """
    return BN.get_attractors_synchronous(nsim, initial_sample_points, n_steps_timeout, INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS)


def get_attractors_synchronous_exact(BN, RETURN_DICTF=False):
    """
    Compute the exact number of attractors in a Boolean network using a fast, vectorized approach.

    This function computes the state of each node for all 2^N states by constructing the network's state space,
    then maps each state to its corresponding successor state via the Boolean functions F.
    Attractors and their basin sizes are then determined by iterating over the entire state space.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        RETURN_DICTF (bool, optional): If True, the state space is returned as a dictionary, 
                                        in which each state is associated by its decimal representation.

    Returns:
        tuple: A tuple containing:
            - attractors (list): List of attractors (each attractor is represented as a list of states forming the cycle).
            - number_of_attractors (int): Total number of unique attractors.
            - basin_sizes (list): List of counts for each attractor.
            - attractor_dict (dict): Dictionary mapping each state (in decimal) to its attractor index.
            - state_space (np.array): The constructed state space matrix (of shape (2^N, N)).
            - dictF (dict, only returned if RETURN_DICTF==True): State space as dictionary.
    """
    return BN.get_attractors_synchronous_exact(RETURN_DICTF)


## Transform Boolean networks
def get_essential_network(BN):
    """
    Determine the essential components of a Boolean network.

    For each node in a Boolean network, represented by its Boolean function and its regulators,
    this function extracts the “essential” part of the function by removing non-essential regulators.
    The resulting network contains, for each node, a reduced truth table (with only the essential inputs)
    and a corresponding list of essential regulators.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.

    Returns:
        tuple: (F_essential, I_essential) where:
            - F_essential is a list of N Boolean functions (truth tables) of length 2^(m_i), with m_i ≤ n_i,
              representing the functions restricted to the essential regulators.
            - I_essential is a list of N lists containing the indices of the essential regulators for each node.
    """
    return BN.get_essential_network()


def get_edge_controlled_network(BN, control_target, control_source, type_of_edge_control=0, left_side_of_truth_table=[]):
    """
    Generate a perturbed Boolean network by removing the influence of a specified regulator on a specified target.

    The function modifies the Boolean function for a target node by restricting it to those entries in its truth table
    where the input from a given regulator equals the specified type_of_control. The regulator is then removed from
    the wiring diagram for that node.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        control_target (int): Index of the target node to be perturbed.
        control_source (int): Index of the regulator whose influence is to be removed.
        type_of_edge_control (int, optional): Source value in regulation after control. Default is 0.
        left_side_of_truth_table (optional): Precomputed truth table (array of tuples) for the target node with len(I[control_target]) inputs.
                                              If not provided, it is computed.

    Returns:
        BooleanNetwork object where:
            - F is the updated list of Boolean functions after perturbation.
            - I is the updated wiring diagram after removing the control regulator from the target node.
            - ns is the updated list of in-degrees for each node.
    """
    return BN.get_edge_controlled_network(control_target, control_source, type_of_edge_control, left_side_of_truth_table)


def get_BN_with_fixed_source_nodes(F,I,n_variables,n_source_nodes,values_source_nodes):
    
    #NOTE: F, I must be arranged so that the source nodes appear last
    
    assert len(F) == len(I)
    F_new = [np.array(el.f) for el in F[:n_variables]]
    I_new = [np.array(el) for el in I[:n_variables]]
    
    for source_node,value in zip(list(range(n_variables,n_variables+n_source_nodes)),values_source_nodes):
        for i in range(n_variables):
            try:
                index = list(I[i]).index(source_node) #check if the constant is part of regulators
            except ValueError:
                continue
            truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=len(I_new[i]))))))
            indices_to_keep = np.where(truth_table[:,index]==value)[0]
            F_new[i] = F_new[i][indices_to_keep]
            I_new[i] = I_new[i][~np.in1d(I_new[i], source_node)]
    return F_new,I_new


def get_external_inputs(BN):
    """
    Identify external inputs in a Boolean network.

    A node is considered an external input if it has exactly one regulator and that regulator is the node itself.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.

    Returns:
        np.array: Array of node indices that are external inputs.
    """
    return BN.get_external_inputs()


## Robustness measures: synchronous Derrida value, entropy of basin size distribution, coherence, fragility
def get_derrida_value(BN, nsim=1000, EXACT = False):
    """
    Estimate the Derrida value for a Boolean network.

    The Derrida value is computed by perturbing a single node in a randomly chosen state and measuring
    the average Hamming distance between the resulting updated states of the original and perturbed networks.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        nsim (int, optional): Number of simulations to perform. Default is 1000.

    Returns:
        float: The average Hamming distance (Derrida value) over nsim simulations.
    """
    return BN.get_derrida_value(nsim, EXACT)


def get_relative_size_of_largest_basin(basin_sizes):
    """
    Compute the size of the largest basin relative to the total size of the state space.

    This function calculates the ratio of the largest basin size to the size of the state space.
    This metric is useful for assessing the dominance of a particular basin of attraction in a Boolean network.

    Parameters:
        basin_sizes (list or array-like): A list where each element represents the size of a basin
                                           (i.e., the number of initial conditions that converge to a specific attractor).

    Returns:
        float: The proportion of the largest basin size (largest basin size divided by the total sum of basin sizes).
    """
    return max(basin_sizes) * 1.0 / sum(basin_sizes)


def get_entropy_of_basin_size_distribution(basin_sizes):
    """
    Compute the Shannon entropy of the basin size distribution.

    This function calculates the Shannon entropy of a probability distribution derived from the basin sizes.
    First, the basin sizes are normalized to form a probability distribution, and then the entropy is computed
    using the formula: H = - sum(p_i * log(p_i)), where p_i is the proportion of the basin size i.

    Parameters:
        basin_sizes (list or array-like): A list where each element represents the size of a basin,
                                           i.e., the number of initial conditions that converge to a particular attractor.

    Returns:
        float: The Shannon entropy of the basin size distribution.
    """
    total = sum(basin_sizes)
    probabilities = [size * 1.0 / total for size in basin_sizes]
    return sum([-np.log(p) * p for p in probabilities])


def get_coherence_from_attractor_dict_exact(attractor_dict, left_side_of_truth_table=None):
    """
    Compute the coherence of a Boolean network based on its attractor dictionary.

    This function computes the proportion of neighbors in the Boolean hypercube that, following a synchronous update,
    transition to the same attractor. For each state in the fully sampled state space (left_side_of_truth_table),
    it examines all N neighbors (each obtained by flipping one bit) and counts how many have the same attractor
    as the current state. The robustness is then given as the fraction of such edges over the total number of edges
    in the hypercube (which is 2^(N-1) * N).

    Parameters:
        attractor_dict (dict): A dictionary mapping each state (in decimal representation) to its attractor index.
                                 This dictionary must be computed from a fully sampled state space, 
                                 e.g., the output of get_attractors_synchronous_exact.
        left_side_of_truth_table (list or array): The full truth table of the network states, where each entry is a numpy array
                                                    representing one state (of length N). If None, it is generated

    Returns:
        float: The robustness measure (i.e., the proportion of neighboring states that transition to the same attractor).
    """
    n_attractors = len(set(attractor_dict.values()))
    if n_attractors == 1:
        return 1
    
    N = int(np.log2(len(attractor_dict)))

    if left_side_of_truth_table is None:
        left_side_of_truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=N)))))

    powers_of_2 = np.array([2**i for i in range(N)])[::-1]
    count_of_neighbors_who_transition_to_same_attractor = 0
    for xdec, x in enumerate(left_side_of_truth_table):
        for i in range(N):
            if x[i] == 0:
                ydec = xdec + powers_of_2[i]
            else:
                continue
            if attractor_dict[xdec] == attractor_dict[ydec]:
                count_of_neighbors_who_transition_to_same_attractor += 1
    return count_of_neighbors_who_transition_to_same_attractor / (2**(N-1) * N)

def get_attractors_and_robustness_measures_synchronous_exact(BN):
    """
    Compute the attractors and several robustness measures of a Boolean network.

    This function computes the exact attractors and robustness (coherence and fragility) of each basin of attraction and of each attractor.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.

    Returns:
        tuple: A tuple containing:
            - attractors (list): List of attractors (each attractor is represented as a list of state decimal numbers).
            - exact_number_of_attractors (int): The exact number of network attractors.
            - exact_basin_sizes (list): List of basin sizes for each attractor.
            - attractor_dict (dict): Dictionary mapping each state (in decimal) to its attractor index.
            - state_space (np.array): The constructed state space matrix (of shape (2^N, N)).
            - exact_basin_coherence (list): coherence of each basin.
            - exact_basin_fragility (list): fragility of each basin.
            - attractor_coherence (list): attractor coherence of each basin (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
            - attractor_fragility (list): attractor fragility of each basin  (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
            - coherence (float): overall network coherence
            - fragility (float): overall network fragility
    """
    return BN.get_attractors_and_robustness_measures_synchronous_exact()


def get_attractors_and_robustness_measures_synchronous(BN, number_different_IC=500, RETURN_ATTRACTOR_COHERENCE = False):
    """
    Approximate global robustness measures and attractors.

    This function samples the attractor landscape by simulating the network from a number of different initial
    conditions. It computes:
      1. The coherence: the proportion of neighboring states (in the Boolean hypercube) that, after synchronous
         update, transition to the same attractor.
      2. The fragility: a measure of how much the attractor state changes (assumed under synchronous update) in response
         to perturbations.
      3. The final time-step Hamming distance between perturbed trajectories.
    
    In addition, it collects several details about each attractor (such as basin sizes, coherence of each basin, etc.).

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        number_different_IC (int, optional): Number of different initial conditions to sample (default is 500).
        RETURN_ATTRACTOR_COHERENCE (bool, optional): Determines whether the attractor coherence should also be computed (default is No, i.e., False).

    Returns:
        tuple: A tuple containing:
            - attractors (list): List of attractors (each attractor is represented as a list of state decimal numbers).
            - lower_bound_number_of_attractors (int): The lower bound on the number of attractors found.
            - approximate_basin_sizes (list): List of basin sizes for each attractor.
            - approximate_coherence (float): The approximate overall network coherence.
            - approximate_fragility (float): The approximate overall network fragility.
            - final_hamming_distance_approximation (float): The approximated final Hamming distance measure.
            - approximate_basin_coherence (list): The approximate coherence of each basin.
            - approximate_basin_fragility (list): The approximate fragility of each basin.
            - attractor_coherence (list): The exact coherence of each attractor (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
            - attractor_fragility (list): The exact fragility of each attractor (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
    """
    return BN.get_attractors_and_robustness_measures_synchronous(number_different_IC, RETURN_ATTRACTOR_COHERENCE)
    

def get_attractors_synchronous_exact_with_external_inputs(F, I, input_patterns = [],left_side_of_truth_table=None):
    """
    Compute the exact number of attractors in a Boolean network using a fast, vectorized approach.

    This function computes the state of each node for all 2^N states by constructing the network's state space,
    then maps each state to its corresponding successor state via the Boolean functions F.
    Attractors and their basin sizes are then determined by iterating over the entire state space.

    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of lists, where I[i] contains the indices of the regulators for node i.
        left_side_of_truth_table (np.array, optional): Precomputed array of all 2^N states (each row is a state).
                                                        If None, it is generated.

    Returns:
        tuple: A tuple containing:
            - attractors (list): List of attractors (each attractor is represented as a list of states forming the cycle).
            - number_of_attractors (int): Total number of unique attractors.
            - basin_sizes (list): List of counts for each attractor.
            - attractor_dict (dict): Dictionary mapping each state (in decimal) to its attractor index.
            - state_space (np.array): The constructed state space matrix (of shape (2^N, N)).
            
            
    c1 = c3 AND a2 AND b1
    c2 = b3 OR c4 OR c1
    c3 = c1
    c4 = c3 OR c2
    a2
    b1
    b3
    
    F = [
        [0,0,0,0, 0,0,0,1],
        [0,1,1,1, 1,1,1,1],
        [0,1],
        [0,1,1,1]
        ]
    
    I = [
        [2,4,5],
        [0,3,6],
        [0],
        [1,2]
        ]
    
    F = [np.array(el) for el in F]
    I = [np.array(el) for el in I]
    
    n_var = 4
    n_const = 3
    
    values_constants = [1,0,0]
    
    input_patterns = [[1],[0],[0]]
    input_patterns = [[0,1],[0],[0]]
    input_patterns = [[0,1,0],[1,0],[0,1]]
    
    
    
   
    
    """
    

        
    

    n_var = len(F)
    n_const = len(input_patterns)
    
    length_input_patterns = list(map(len,input_patterns))
    lcm = math.lcm(*length_input_patterns)
    periodic_pattern_of_external_inputs = np.zeros((lcm,n_const),dtype=int)
    for i,pattern in enumerate(input_patterns):
        for j in range(int(lcm/len(pattern))):
            periodic_pattern_of_external_inputs[len(pattern)*j:len(pattern)*(j+1),i] = pattern


    n_initial_values = len(periodic_pattern_of_external_inputs)
    
    Fs_with_fixed_sources = []
    Is_with_fixed_sources = []
    degrees_with_fixed_sources = []
    for input_values in periodic_pattern_of_external_inputs:
        dummy = get_BN_with_fixed_source_nodes(F,I,n_var,n_const,input_values)
        Fs_with_fixed_sources.append(dummy[0])
        Is_with_fixed_sources.append(dummy[1])
        degrees_with_fixed_sources.append(list(map(len, dummy[1])))
    
    N = len(F)
    
    if left_side_of_truth_table is None:
        left_side_of_truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=N)))))
    
    powers_of_2 = np.array([2**i for i in range(N)])[::-1]
    
    state_space_with_fixed_sources = []
    dictF_with_fixed_source = []
    for iii in range(n_initial_values):
        state_space = np.zeros((2**N, N), dtype=int)
        for i in range(N):
            for j, x in enumerate(itertools.product([0, 1], repeat=degrees_with_fixed_sources[iii][i])):
                if Fs_with_fixed_sources[iii][i][j]==1:
                    # For rows in left_side_of_truth_table where the columns I[i] equal x, set state_space accordingly.
                    state_space[np.all(left_side_of_truth_table[:, Is_with_fixed_sources[iii][i]] == np.array(x), axis=1), i] = 1
        state_space_with_fixed_sources.append(state_space)
        dictF_with_fixed_source.append(dict(zip(list(range(2**N)), np.dot(state_space, powers_of_2))))
        
    attractors = []
    basin_sizes = []
    attractor_dict = dict()
    iii_start = 0
    for iii_start in range(2):
        for xdec in range(2**N):
            iii = iii_start
            queue = [xdec]
            while True:
                fxdec = dictF_with_fixed_source[iii % n_initial_values][xdec]
                try:
                    index_attr = attractor_dict[(iii % n_initial_values,fxdec)]
                    basin_sizes[index_attr] += 1
                    attractor_dict.update(list(zip(zip(np.arange(iii_start,len(queue)+iii_start)%n_initial_values,queue), [index_attr] * len(queue))))
                    break
                except KeyError:
                    try: 
                        index = queue[-n_initial_values::-n_initial_values].index(fxdec) 
                        dummy = np.arange(iii_start,len(queue)+iii_start)%n_initial_values
                        attractor_dict.update(list(zip(zip(dummy,queue), [len(attractors)] * len(queue))))
                        attractors.append(list(zip(dummy[-n_initial_values*(index+1):],queue[-n_initial_values*(index+1):])))
                        basin_sizes.append(1)
                        break
                    except ValueError:
                        pass
                queue.append(fxdec)
                xdec = fxdec
                iii += 1
    return (attractors, len(attractors), basin_sizes, attractor_dict, state_space)


def get_strongly_connected_components(BN):
    """
    Determine the strongly connected components of a wiring diagram.

    The wiring diagram is provided as a list of lists I, where I[i] contains the indices of regulators for node i.
    The function constructs a directed graph from these edges and returns its strongly connected components.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.

    Returns:
        list: A list of sets, each representing a strongly connected component.
    """
    return BN.get_strongly_connected_components()

def get_modular_structure(BN):
    return BN.get_modular_structure()

def topological_sort_dag(dag,num_sccs):
    visited_component = set()
    topological_order = []

    def dfs(component):
        visited_component.add(component)
        for neighbor in dag[component]:
            if neighbor not in visited_component:
                dfs(neighbor)
        topological_order.append(component)

    for component in range (num_sccs):
        if component not in visited_component:
            dfs(component)

    return topological_order[::-1]



def load_modular_BN_ex1():
    variables = ['a1','a2','b1','b2','b3','c1','c2','c3','d1','d2']
    F = [[0,1],
         [0,1],
         [0,1,1,1],
         [0,1],
         [1,0],
         [0,0,0,0,0,1,1,1],
         [1,0,0,0],
         [0,1,1,1],
         [0,0,0,1],
         [0,1]]
    I  = [[1],
          [0],
          [3,4],
          [2],
          [3],
          [1,4,7],
          [4,5],
          [5,6],
          [7,9],
          [8]]
    
    F = [np.array(el) for el in F]
    I = [np.array(el) for el in I]    
    return BooleanNetwork(F,I,[len(I[i]) for i in range(len(I))]),variables

def get_attractors_synchronous_exact_exploiting_modularity(F, I):
    sccs = get_strongly_connected_components(I)
    module_per_node = {}
    for i,scc in enumerate(sccs):
        for el in scc:
            module_per_node.update({el:i})
    n_modules = len(sccs)
    dag = get_modular_structure(I)
    G = nx.from_edgelist(dag,nx.DiGraph)
    modules_sorted = list(nx.topological_sort(G))
    nodes_in_each_module = [np.sort(list(sccs[module_id])) for module_id in range(n_modules)]
    pos_in_each_module = [dict(zip(nodes_in_each_module[module_id],range(len(nodes_in_each_module[module_id])))) for module_id in range(n_modules)]
    inputs_per_module = [[] for i in range(n_modules)]
    for (a,b) in dag:
        inputs_per_module[b].append(a)
    
    module_attractors_binary = [[] for i in range(n_modules)]
    for module_id in modules_sorted:
        
        if len(inputs_per_module[module_id])==0:
            F1 = [F[j] for j in list(sccs[module_id])]
            I1 = [I[j] for j in list(sccs[module_id])] 
            #reindex I1 to keep track of which nodes are in the module
            nodes_in_module = nodes_in_each_module[module_id]
            pos_in_module = pos_in_each_module[module_id]
            for i in range(len(nodes_in_module)):
                for j in range(len(I1[i])):
                    I1[i][j] = pos_in_module[I1[i][j]]
            module_attractors = get_attractors_synchronous_exact(F1, I1)[0]
            module_attractors_binary[module_id] = [np.array(list(map(lambda x: utils.dec2bin(x,len(nodes_in_module)),module_attractors[i]))) for i in range(len(module_attractors))]
      
        
        else:  
            downstream_module_attractors = []
            nodes_in_module = nodes_in_each_module[module_id]
            pos_in_module = pos_in_each_module[module_id]
            n_nodes_in_module = len(pos_in_module)
           
            #Getting all the nodes for every input module
            input_nodes = {}
            count = 0
            for node in nodes_in_module:
                for regulator in I[node]:
                    try:
                        pos_in_module[regulator]
                    except KeyError:
                        try:
                            input_nodes[regulator]
                        except KeyError:
                            input_nodes.update({regulator:n_nodes_in_module+count})
                            count+=1
            
            pos_in_module.update(input_nodes)
            
            
            pos_in_module_per_input_nodes = {}
            for input_node in input_nodes.keys():
                pos_in_module_per_input_nodes.update({input_node: pos_in_each_module[module_per_node[input_node]][input_node]})
            
            nodes_per_upstream_module_we_care_about = {}
            for input_node in input_nodes.keys():
                upstream_module_id = module_per_node[input_node]
                try:
                    nodes_per_upstream_module_we_care_about[upstream_module_id].add(input_node)
                except KeyError:
                    nodes_per_upstream_module_we_care_about.update({upstream_module_id:set([input_node])})
                    
            
            
            #Getting the Upstream Attractors
            upstream_attractors = []
            for upstream_module in inputs_per_module[module_id]:
                upstream_attractors.append(module_attractors_binary[upstream_module])
                
            all_input_patterns = []
            for attractor_of_each_upstream_module in itertools.product(*upstream_attractors):
                all_input_patterns.append([])
                for attractor,upstream_module_id in zip(attractor_of_each_upstream_module,inputs_per_module[module_id]):
                    all_input_patterns[-1].append(utils.flatten(attractor[:,np.array(list(map(lambda x: pos_in_module_per_input_nodes[x],nodes_per_upstream_module_we_care_about[upstream_module_id])))]))

            F2 = [F[j] for j in nodes_in_module]
            I2 = [list(map(lambda x: pos_in_module[x],I[j])) for j in nodes_in_module]                
                
            module_attractors = get_attractors_synchronous_exact_with_external_inputs(F2, I2, input_patterns = all_input_patterns[0])[0]
            
            #next line needs to be checked
            module_attractors_binary[module_id] = [np.array(list(map(lambda x: utils.dec2bin(x,len(nodes_in_module)),module_attractors[i]))) for i in range(len(module_attractors))]
            
   
    return module_attractors_binary


def adjacency_matrix(BN, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
    """
    Construct the (binary) adjacency matrix from the wiring diagram.

    Given the wiring diagram I (a list of regulator lists for each node) and a list of constants,
    this function builds an adjacency matrix where each entry m[j, i] is 1 if node j regulates node i.
    Self-loops can be optionally ignored, and constant nodes can be excluded.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        constants (list, optional): List of constant nodes.
        IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
        IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded from the matrix.

    Returns:
        np.array: The binary adjacency matrix.
    """
    return BN.get_adjacency_matrix(constants, IGNORE_SELFLOOPS, IGNORE_CONSTANTS)


def get_signed_adjacency_matrix(BN, type_of_each_regulation, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
    """
    Construct the signed adjacency matrix of a Boolean network.

    The signed adjacency matrix assigns +1 for increasing (activating) regulations,
    -1 for decreasing (inhibiting) regulations, and NaN for any other type.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        type_of_each_regulation (list): List of lists corresponding to the type of regulation ('increasing' or 'decreasing')
                                        for each edge in I.
        constants (list, optional): List of constant nodes.
        IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
        IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded.

    Returns:
        np.array: The signed adjacency matrix.
    """
    return BN.get_signed_adjacency_matrix(type_of_each_regulation, constants, IGNORE_SELFLOOPS, IGNORE_CONSTANTS)


def get_signed_effective_graph(BN, type_of_each_regulation, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
    """
    Construct the signed effective graph of a Boolean network.

    This function computes an effective graph in which each edge is weighted by its effectiveness.
    Effectiveness is obtained via get_edge_effectiveness on the corresponding Boolean function.
    Edges are signed according to the type of regulation ('increasing' or 'decreasing').

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        type_of_each_regulation (list): List of lists specifying the type of regulation for each edge.
        constants (list, optional): List of constant nodes.
        IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
        IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded.

    Returns:
        np.array: The signed effective graph as a matrix of edge effectiveness values.
    """
    return BN.get_signed_effective_graph(type_of_each_regulation, constants, IGNORE_SELFLOOPS, IGNORE_CONSTANTS)


def get_ffls(BN):
    """
    Identify feed-forward loops (FFLs) in a Boolean network and optionally determine their types.

    A feed-forward loop (FFL) is a three-node motif where node i regulates node k both directly and indirectly via node j.
    If F is provided, the function also computes the monotonicity of each regulation in the FFL.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.

    Returns:
        If F is None:
            list: A list of FFLs, each represented as a list [i, k, j] (or similar ordering).
        Otherwise:
            tuple: A tuple (ffls, types), where ffls is a list of FFLs and types is a list of corresponding monotonicity types.
    """
    return BN.get_ffls()


def get_ffls_from_I(BN, types_I=None):
    """
    Identify feed-forward loops (FFLs) in a Boolean network based solely on the wiring diagram.

    The function uses the inverted wiring diagram to identify common targets and returns the FFLs found.
    If types_I (the type of each regulation) is provided, it also returns the corresponding regulation types.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        types_I (list, optional): List of lists specifying the type (e.g., 'increasing' or 'decreasing') for each regulation.

    Returns:
        If types_I is provided:
            tuple: (ffls, types) where ffls is a list of identified FFLs (each as a list [i, j, k]),
                   and types is a list of corresponding regulation type triplets.
        Otherwise:
            list: A list of identified FFLs.
    """
    return BN.get_ffls_from_I(types_I)


def get_ffl_type_number(types_vector):
    """
    Compute a numeric type for a feed-forward loop (FFL) based on its regulation types.

    For a given FFL, this function converts the types (strings 'increasing' or 'decreasing') into a numeric code.
    If any type is not in the set {'decreasing', 'increasing'} and 'not essential' is not present, -1 is returned;
    if 'not essential' is present, -2 is returned.

    Parameters:
        types_vector (list): List of regulation type strings for the FFL.

    Returns:
        int: A numeric type representing the FFL, or -1/-2 if types are not as expected.
    """
    if not set(types_vector).issubset(set(['decreasing', 'increasing'])):
        return -1 if 'not essential' not in types_vector else -2
    else:
        dummy = np.array([1 if el == 'increasing' else 0 for el in types_vector])
        nr_type = int(np.dot(dummy, 2**np.arange(len(types_vector))))
    return nr_type


def is_ffl_coherent(types_vector):
    """
    Determine whether a feed-forward loop (FFL) is coherent.

    A coherent FFL is defined (in this context) such that the number of 'increasing' regulations in the FFL is odd.
    If the types are not exclusively 'decreasing' or 'increasing', NaN is returned.

    Parameters:
        types_vector (list): List of regulation type strings for the FFL.

    Returns:
        bool or float: True if the FFL is coherent, False otherwise, or NaN if types are ambiguous.
    """
    if not set(types_vector).issubset(set(['decreasing', 'increasing'])):
        return np.nan
    else:
        dummy = np.array([1 if el == 'increasing' else 0 for el in types_vector])
        COHERENT = (sum(dummy) % 2) == 1
    return COHERENT


def generate_networkx_graph(BN, constants, variables):
    """
    Generate a NetworkX directed graph from a wiring diagram.

    Nodes are labeled with variable names (from variables) and constant names (from constants). Edges are added
    from each regulator to its target based on the wiring diagram I.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        constants (list): List of constant names.
        variables (list): List of variable names.

    Returns:
        networkx.DiGraph: The noderated directed graph.
    """
    return BN.generate_networkx_graph()


def generate_networkx_graph_from_edges(BN, n_variables):
    """
    Generate a NetworkX directed graph from an edge list derived from the wiring diagram.

    Only edges among the first n_variables (excluding constant self-loops) are included.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        n_variables (int): Number of variable nodes (constants are excluded).

    Returns:
        networkx.DiGraph: The generated directed graph.
    """
    return BN.generate_networkx_graph_from_edges(n_variables)


def simple_cycles(G, max_len=4):
    """
    Generate simple cycles in a directed graph using a variant of Johnson's algorithm.

    This function finds simple cycles (elementary circuits) in graph G with a maximum length of max_len.
    It first yields self-cycles (if any), then iterates through the strongly connected components of G,
    recursively unblocking nodes to yield cycles.

    Parameters:
        G (networkx.DiGraph): A directed graph.
        max_len (int, optional): Maximum length of cycles to consider (default is 4).

    Yields:
        list: A list of nodes representing a simple cycle.
    """

    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()

    subG = nx.DiGraph(G.edges())
    sccs = [scc for scc in nx.strongly_connected_components(subG) if len(scc) > 1]
    
    # Yield self-cycles and remove them.
    for v in subG:
        if subG.has_edge(v, v):
            yield [v]
            subG.remove_edge(v, v)
    
    while sccs:
        scc = sccs.pop()
        sccG = subG.subgraph(scc)
        startnode = scc.pop()
        path = [startnode]
        len_path = 1
        blocked = set()
        closed = set()
        blocked.add(startnode)
        B = defaultdict(set)
        stack = [(startnode, list(sccG[startnode]))]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs and len_path <= max_len:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                elif nextnode not in blocked:
                    path.append(nextnode)
                    len_path += 1
                    stack.append((nextnode, list(sccG[nextnode])))
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            if not nbrs or len_path > max_len:
                if thisnode in closed:
                    _unblock(thisnode, blocked, B)
                else:
                    for nbr in sccG[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
                path.pop()
                len_path -= 1
        H = subG.subgraph(scc)
        sccs.extend(scc for scc in nx.strongly_connected_components(H) if len(scc) > 1)


def get_type_of_loop(BN, loop):
    """
    Determine the regulation types along a feedback loop.

    For a given loop (a list of node indices), this function returns a list containing
    the type (e.g., 'increasing' or 'decreasing') of each regulation along the loop.
    The loop is assumed to be ordered such that the first node is repeated at the end.

    Parameters:
        BN (BooleanNetwork): Boolean Network object.
        loop (list): List of node indices representing the loop.

    Returns:
        list: A list of regulation types corresponding to each edge in the loop.
    """
    return BN.get_type_of_loop(loop)


def get_loop_type_number(types_vector):
    """
    Compute a numeric code for a loop based on its regulation types.

    For the given list of regulation types in a loop, this function returns the number
    of 'decreasing' regulations. If the types are not a subset of {'decreasing', 'increasing'},
    it returns -1 (or -2 if 'not essential' is present).

    Parameters:
        types_vector (list): List of regulation type strings.

    Returns:
        int: A numeric code representing the loop type.
    """
    if not set(types_vector).issubset(set(['decreasing', 'increasing'])):
        return -1 if 'not essential' not in types_vector else -2
    else:
        nr_type = int(np.sum([1 if el == 'decreasing' else 0 for el in types_vector]))
    return nr_type


def is_pos_loop(types_vector):
    """
    Determine whether a loop is positive based on its regulation types.

    A positive loop is defined such that the total number of 'decreasing' regulations is even.
    If the types vector contains values other than 'decreasing' or 'increasing', NaN is returned.

    Parameters:
        types_vector (list): List of regulation type strings.

    Returns:
        bool or float: True if the loop is positive, False if negative, or NaN if undefined.
    """
    if not set(types_vector).issubset(set(['decreasing', 'increasing'])):
        return np.nan
    else:
        POSITIVE = (np.sum([1 if el == 'decreasing' else 0 for el in types_vector]) % 2) == 0
    return POSITIVE