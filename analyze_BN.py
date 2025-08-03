#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 09:25:40 2025

@author: Claus Kadelka
"""

##Imports

import numpy as np
import itertools
import networkx as nx
import random
from collections import defaultdict

import utils
import analyze_BF



def get_strongly_connected_components(I):
    """
    Determine the strongly connected components of a wiring diagram.

    The wiring diagram is provided as a list of lists I, where I[i] contains the indices of regulators for node i.
    The function constructs a directed graph from these edges and returns its strongly connected components.

    Parameters:
        I (list): A list of lists, where each inner list contains the regulators (source nodes) for the corresponding target node.

    Returns:
        list: A list of sets, each representing a strongly connected component.
    """
    edges_wiring_diagram = []
    for target, regulators in enumerate(I):
        for regulator in regulators:
            edges_wiring_diagram.append((regulator, target))
    subG = nx.from_edgelist(edges_wiring_diagram, create_using=nx.MultiDiGraph())
    return [scc for scc in nx.strongly_connected_components(subG)]

def get_modular_structure(I):
    sccs = get_strongly_connected_components(I)
    scc_dict = {}
    for j,s in enumerate(sccs):
        for el in s:
            scc_dict.update({el:j})
    dag = set()
    for target,regulators in enumerate(I):
        for regulator in regulators:
            edge = (scc_dict[regulator],scc_dict[target])
            if edge[0]!=edge[1]:
                dag.add(edge)   
    return dag


def get_essential_network(F, I):
    """
    Determine the essential components of a Boolean network.

    For each gene in a Boolean network, represented by its Boolean function and its regulators,
    this function extracts the “essential” part of the function by removing non-essential regulators.
    The resulting network contains, for each gene, a reduced truth table (with only the essential inputs)
    and a corresponding list of essential regulators.

    Parameters:
        F (list): A list of N Boolean functions (truth tables). For gene i, the Boolean function is given as a list
                  of length 2^(n_i), where n_i is the number of regulators for that gene.
        I (list): A list of N lists. For gene i, I[i] is a list of regulator indices (typically 0, 1, ..., n_i-1)
                  corresponding to the wiring diagram of the Boolean network.

    Returns:
        tuple: (F_essential, I_essential) where:
            - F_essential is a list of N Boolean functions (truth tables) of length 2^(m_i), with m_i ≤ n_i,
              representing the functions restricted to the essential regulators.
            - I_essential is a list of N lists containing the indices of the essential regulators for each gene.
    """
    import itertools
    F_essential = []
    I_essential = []
    for f, regulators in zip(F, I):
        if len(f) == 0:  # happens if the actual degree of f was too large for it to be loaded
            F_essential.append(f)
            I_essential.append(regulators)
            continue
        elif sum(f) == 0:
            F_essential.append(np.array([0]))
            I_essential.append(np.array([], dtype=int))
            continue
        elif sum(f) == len(f):
            F_essential.append(np.array([1]))
            I_essential.append(np.array([], dtype=int))
            continue
        essential_variables = np.array(analyze_BF.get_essential_variables(f))
        n = len(regulators)
        non_essential_variables = np.array(list(set(list(range(n))) - set(essential_variables)))
        if len(non_essential_variables) == 0:
            F_essential.append(f)
            I_essential.append(regulators)
        else:
            left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=n)))
            F_essential.append(np.array(f)[np.sum(left_side_of_truth_table[:, non_essential_variables], 1) == 0])
            I_essential.append(np.array(regulators)[essential_variables])
    return F_essential, I_essential


def get_perturbed_network(F, I, ns, control_target, control_source, type_of_control=0, left_side_of_truth_table=[]):
    """
    Generate a perturbed Boolean network by removing the influence of a specified regulator.

    The function modifies the Boolean function for a target gene by restricting it to those entries in its truth table
    where the input from a given regulator equals the specified type_of_control. The regulator is then removed from
    the wiring diagram for that gene.

    Parameters:
        F (list): List of Boolean functions (truth tables) for each gene.
        I (list): Wiring diagram for the network; each entry I[i] is a list of regulator indices for gene i.
        ns (list or np.array): List of in-degrees (number of regulators) for each gene.
        control_target (int): Index of the target gene to be perturbed.
        control_source (int): Index of the regulator whose influence is to be removed.
        type_of_control (int, optional): The regulator value (0 or 1) for which the perturbation is applied. Default is 0.
        left_side_of_truth_table (optional): Precomputed truth table (array of tuples) for the target gene with ns[control_target] inputs.
                                              If not provided, it is computed.

    Returns:
        tuple: (F_new, I_new, ns) where:
            - F_new is the updated list of Boolean functions after perturbation.
            - I_new is the updated wiring diagram after removing the control regulator from the target gene.
            - ns is the updated list of in-degrees for each gene.
    """
    F_new = [f for f in F]
    I_new = [i for i in I]

    if left_side_of_truth_table == []:
        left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=ns[control_target])))

    try:
        index = list(I[control_target]).index(control_source)
        F_new[control_target] = F_new[control_target][left_side_of_truth_table[:, index] == type_of_control]
        dummy = list(I_new[control_target])
        dummy.remove(control_source)
        I_new[control_target] = np.array(dummy)
    except ValueError:
        print('source not regulating target')

    ns = list(map(len, I_new))
    return F_new, I_new, ns

def get_constant_nodes(I, degree, N):
    """
    Identify constant nodes in a Boolean network.

    A node is considered constant if it has exactly one regulator and that regulator is the node itself.

    Parameters:
        I (list): A list where I[i] is a list of regulator indices for node i.
        degree (list or array): A list where degree[i] is the number of regulators for node i.
        N (int): Total number of nodes in the network.

    Returns:
        np.array: Array of node indices that are constant.
    """
    return np.array([i for i in range(N) if degree[i] == 1 and I[i][0] == i])

## 5) Analysis methods
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

def update_network_synchronously(F, I, X):
    """
    Perform a synchronous update of a Boolean network.

    Each node's new state is determined by applying its Boolean function to the current states of its regulators.
    The conversion from the regulator states (a binary vector) to a truth table index is done via utils.bin2dec.

    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of regulator indices for each node.
        X (list or np.array): Current state vector of the network.

    Returns:
        np.array: New state vector after the update.
    """
    if type(X)==list:
        X = np.array(X)
    Fx = np.zeros(len(F), dtype=int)
    for i in range(len(F)):
        Fx[i] = update_single_node(f = F[i], states_regulators = X[I[i]])
    return Fx


def update_network_SDDS(F, I, X, P):
    """
    Perform a stochastic update (SDDS) on a Boolean network.

    For each node, the next state is computed as nextstep = F[i] evaluated on the current states of its regulators.
    If nextstep > X[i], the node is activated with probability P[i,0]; if nextstep < X[i],
    the node is degraded with probability P[i,1]. Otherwise, the state remains unchanged.

    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of regulator indices for each node.
        X (list or np.array): Current state vector.
        P (np.array): A len(F)×2 array of probabilities; for each node i, P[i,0] is the activation probability,
                      and P[i,1] is the degradation probability.

    Returns:
        np.array: Updated state vector after applying the stochastic update.
    """
    if type(X)==list:
        X = np.array(X)
    Fx = X.copy()
    for i in range(len(F)):
        nextstep = update_single_node(f = F[i], states_regulators = X[I[i]])
        if nextstep > X[i] and random.random() < P[i, 0]:  # activation
            Fx[i] = nextstep
        elif nextstep < X[i] and random.random() < P[i, 1]:  # degradation
            Fx[i] = nextstep
    return Fx


def update_network_many_times(F, I, initial_state, n_steps):
    """
    Update the state of a Boolean network sychronously multiple time steps.

    Starting from the initial state, the network is updated synchronously n_steps times using the update_network_synchronously function.

    Parameters:
        F (list): List of Boolean functions for each node.
        I (list): List of regulator indices for each node.
        initial_state (list or np.array): Initial state vector of the network.
        n_steps (int): Number of update iterations to perform.

    Returns:
        np.array: Final state vector after n_steps updates.
    """
    N = len(F)
    for i in range(n_steps):
        initial_state = update_network_synchronously(F, I, N, initial_state)
    return initial_state


def get_derrida_value(F, I, nsim):
    """
    Estimate the Derrida value for a Boolean network.

    The Derrida value is computed by perturbing a single node in a randomly chosen state and measuring
    the average Hamming distance between the resulting updated states of the original and perturbed networks.

    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of regulator indices for each node.
        nsim (int): Number of simulations to perform.

    Returns:
        float: The average Hamming distance (Derrida value) over nsim simulations.
    """
    N = len(F)
    hamming_distances = []
    for i in range(nsim):
        X = np.random.randint(0, 2, N)
        Y = X.copy()
        index = np.random.randint(N)
        Y[index] = 1 - Y[index]
        FX = update_network_synchronously(F, I, X)
        FY = update_network_synchronously(F, I, Y)
        hamming_distances.append(sum(FX != FY))
    return np.mean(hamming_distances)



def get_steady_states_asynchronous(F, I, N, nsim=500, EXACT=False, left_side_of_truth_table=[], 
                                   initial_sample_points=[], search_depth=50, SEED=-1, DEBUG=True):
    """
    Compute the steady states of a Boolean network under asynchronous updates.

    This function simulates asynchronous updates of a Boolean network (with N nodes)
    for a given number of initial conditions (nsim). For each initial state, the network
    is updated asynchronously until a steady state (or attractor) is reached or until a maximum
    search depth is exceeded. The simulation can be performed either approximately (by sampling nsim
    random initial conditions) or exactly (by iterating over the entire state space when EXACT=True).

    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
                 Each function is defined over 2^(# of regulators) entries.
        I (list): List of lists, where I[i] contains the indices of the regulators for node i.
        N (int): Total number of nodes in the network.
        nsim (int, optional): Number of initial conditions to simulate (default is 500).
        EXACT (bool, optional): If True, iterate over the entire state space (2^N initial conditions);
                                otherwise, use nsim random initial conditions. (Default is False.)
        left_side_of_truth_table (list, optional): Precomputed truth table (list of tuples) for N inputs.
                                                     Used only if EXACT is True.
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
    if EXACT and left_side_of_truth_table == []:
        left_side_of_truth_table = list(map(np.array, list(itertools.product([0, 1], repeat=N))))

    sampled_points = []
    
    assert initial_sample_points == [] or not EXACT, (
        "Warning: sample points were provided but, with option EXACT==True, the entire state space is computed "
        "(and initial sample points ignored)"
    )
    
    if SEED == -1:
        SEED = int(random.random() * 2**31)
    
    np.random.seed(SEED)
    
    dictF = dict()
    steady_states = []
    basin_sizes = []
    steady_state_dict = dict()   
    
    for iteration in range(nsim if not EXACT else 2**N):
        if EXACT:
            x = left_side_of_truth_table[iteration]
            xdec = iteration
        else:
            if initial_sample_points == []:  # generate random initial states on the fly
                x = np.random.randint(2, size=N)
                xdec = utils.bin2dec(x)
                sampled_points.append(xdec)
            else:                
                x = initial_sample_points[iteration]
                xdec = utils.bin2dec(x)
        
        if DEBUG:
            print(iteration, -1, -1, False, xdec, x)
        for jj in range(search_depth):  # update until a steady state is reached or search_depth is exceeded
            FOUND_NEW_STATE = False
            try:
                # Check if this state is already recognized as a steady state.
                index_ss = steady_state_dict[xdec]
            except KeyError:
                # Asynchronously update the state until a new state is found.
                update_order_to_try = np.random.permutation(N)
                for i in update_order_to_try:
                    try:
                        fxdec = dictF[(xdec, i)]
                        if fxdec != xdec:
                            FOUND_NEW_STATE = True
                            x[i] = 1 - x[i]
                    except KeyError:
                        fx_i = update_single_node(F[i], x[I[i]])
                        if fx_i > x[i]:
                            fxdec = xdec + 2**(N - 1 - i)
                            x[i] = 1
                            FOUND_NEW_STATE = True
                        elif fx_i < x[i]:
                            fxdec = xdec - 2**(N - 1 - i)
                            x[i] = 0
                            FOUND_NEW_STATE = True
                        else:
                            fxdec = xdec
                        dictF.update({(xdec, i): fxdec})
                    if FOUND_NEW_STATE:
                        xdec = fxdec
                        break
                if DEBUG:
                    print(iteration, jj, i, FOUND_NEW_STATE, xdec, x)
            if FOUND_NEW_STATE == False:  # steady state reached
                try:
                    index_ss = steady_state_dict[xdec]
                    basin_sizes[index_ss] += 1
                    break
                except KeyError:
                    steady_state_dict.update({xdec: len(steady_states)})
                    steady_states.append(xdec)
                    basin_sizes.append(1)
                    break
        if DEBUG:
            print()
    if sum(basin_sizes) < (nsim if not EXACT else 2**N):
        print('Warning: only %i of the %i tested initial conditions eventually reached a steady state. Try increasing the search depth. '
              'It may however also be the case that your asynchronous state space contains a limit cycle.' %
              (sum(basin_sizes), nsim if not EXACT else 2**N))
    return (steady_states, len(steady_states), basin_sizes, steady_state_dict, dictF, SEED,
            initial_sample_points if initial_sample_points != [] else sampled_points)


def get_steady_states_asynchronous_given_one_initial_condition(F, I, nsim=500, stochastic_weights=[], initial_condition=0, search_depth=50, SEED=-1, DEBUG=False):
    """
    Determine the steady states reachable from one initial condition using weighted asynchronous updates.

    This function is similar to steady_states_asynchronous_given_one_IC but allows the update order
    to be influenced by provided stochastic weights (one per node). A weight vector (of length N) may be provided,
    and if given, it is normalized and used to bias the random permutation of node update order.
    
    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of regulator indices for each node.
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
    if SEED == -1:
        SEED = int(random.random() * 2**31)
    np.random.seed(SEED)    
    
    N = len(F)
    
    if type(initial_condition) == int:
        initial_condition = np.array(utils.dec2bin(initial_condition, N))
        initial_condition_bin = utils.bin2dec(initial_condition)
    else:
        initial_condition = np.array(initial_condition, dtype=int)
        initial_condition_bin = utils.bin2dec(initial_condition)
    
    assert stochastic_weights == [] or len(stochastic_weights) == N, "one stochastic weight per node is required"    
    if stochastic_weights != []:
        stochastic_weights = np.array(stochastic_weights) / sum(stochastic_weights)
    
    dictF = dict()
    steady_states = []
    basin_sizes = []
    transient_times = []
    steady_state_dict = dict()   
    queues = []
    for iteration in range(nsim):
        x = initial_condition.copy()
        xdec = initial_condition_bin
        queue = [xdec]
        for jj in range(search_depth):  # update until a steady state is reached or search_depth is exceeded
            FOUND_NEW_STATE = False
            try:
                index_ss = steady_state_dict[xdec]
            except KeyError:
                if stochastic_weights != []:
                    update_order_to_try = np.random.choice(N, size=N, replace=False, p=stochastic_weights)
                else:
                    update_order_to_try = np.random.permutation(N)
                for i in update_order_to_try:
                    try:
                        fxdec = dictF[(xdec, i)]
                        if fxdec != xdec:
                            FOUND_NEW_STATE = True
                            x[i] = 1 - x[i]
                    except KeyError:
                        fx_i = update_single_node(F[i], x[I[i]])
                        if fx_i > x[i]:
                            fxdec = xdec + 2**(N - 1 - i)
                            x[i] = 1
                            FOUND_NEW_STATE = True
                        elif fx_i < x[i]:
                            fxdec = xdec - 2**(N - 1 - i)
                            x[i] = 0
                            FOUND_NEW_STATE = True
                        else:
                            fxdec = xdec
                        dictF.update({(xdec, i): fxdec})
                    if FOUND_NEW_STATE:
                        xdec = fxdec
                        queue.append(xdec)
                        break
                if DEBUG:
                    print(iteration, jj, i, FOUND_NEW_STATE, xdec, x)
            if not FOUND_NEW_STATE:  # steady state reached
                queues.append(queue[:])
                try:
                    index_ss = steady_state_dict[xdec]
                    basin_sizes[index_ss] += 1
                    transient_times[index_ss].append(jj)
                    break
                except KeyError:
                    steady_state_dict.update({xdec: len(steady_states)})
                    steady_states.append(xdec)
                    basin_sizes.append(1)
                    transient_times.append([jj])
                    break
        if FOUND_NEW_STATE:
            print(jj)
            break
        if DEBUG:
            print()
    if sum(basin_sizes) < nsim:
        print('Warning: only %i of the %i tested initial conditions eventually reached a steady state. '
              'Try increasing the search depth. It may also be that your asynchronous state space contains a limit cycle.' % (sum(basin_sizes), nsim))
    return (steady_states, len(steady_states), basin_sizes, transient_times, steady_state_dict, dictF, SEED, queues)


def get_attractors_synchronous(F, I, nsim=500, initial_sample_points=[], n_steps_timeout=1000000000000,
                               INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS=True):
    """
    Compute the number of attractors in a Boolean network using an alternative (v2) approach.

    This version is optimized for networks with longer average path lengths. For each of nb initial conditions,
    the network is updated synchronously until an attractor is reached or until n_steps_timeout is exceeded.
    The function returns the attractors found, their basin sizes, a mapping of states to attractors,
    the set of initial sample points used, the explored state space, and the number of simulations that timed out.

    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of lists, where I[i] contains the indices of the regulators for node i.
        nsim (int, optional): Number of initial conditions to simulate (default is 500).
        initial_sample_points (list, optional): List of initial states (in decimal) to use.
        n_steps_timeout (int, optional): Maximum number of update steps allowed per simulation (default is a very large number).
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
            - n_timeout (int): Number of simulations that reached the step timeout.
    """
    dictF = dict()
    attractors = []
    basin_sizes = []
    attr_dict = dict()
    state_space = []
    
    sampled_points = []
    n_timeout = 0
    
    N = len(F)
    
    INITIAL_SAMPLE_POINTS_EMPTY = utils.check_if_empty(initial_sample_points)
    if not INITIAL_SAMPLE_POINTS_EMPTY:
        nsim = len(initial_sample_points)
    
    for i in range(nsim):
        if INITIAL_SAMPLE_POINTS_EMPTY:
            x = np.random.randint(2, size=N)
            xdec = utils.bin2dec(x)
            sampled_points.append(xdec)
        else:
            if INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS:
                x = initial_sample_points[i]
                xdec = utils.bin2dec(x)
            else:
                xdec = initial_sample_points[i]
                x = np.array(utils.dec2bin(xdec, N))
        queue = [xdec]
        count = 0
        while count < n_steps_timeout:
            try:
                fxdec = dictF[xdec]
            except KeyError:
                fx = update_network_synchronously(F, I, x)
                fxdec = utils.bin2dec(fx)
                dictF.update({xdec: fxdec})
                x = fx
            if count == 0:
                state_space.append(fxdec)
            try:
                index_attr = attr_dict[fxdec]
                basin_sizes[index_attr] += 1
                attr_dict.update(list(zip(queue, [index_attr] * len(queue))))
                break
            except KeyError:
                try:
                    index = queue.index(fxdec)
                    attr_dict.update(list(zip(queue[index:], [len(attractors)] * (len(queue) - index))))
                    attractors.append(queue[index:])
                    basin_sizes.append(1)
                    break
                except ValueError:
                    pass
            queue.append(fxdec)
            xdec = fxdec
            count += 1
            if count == n_steps_timeout:
                n_timeout += 1            
    return (attractors, len(attractors), basin_sizes, attr_dict,
            sampled_points if INITIAL_SAMPLE_POINTS_EMPTY else initial_sample_points,
            state_space, n_timeout)


def get_attractors_synchronous_exact(F, I, left_side_of_truth_table=None,RETURN_DICTF=False):
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
    """
    
    N = len(F)
    
    if left_side_of_truth_table is None:
        left_side_of_truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=N)))))
    
    utils.bin2dec = np.array([2**i for i in range(N)])[::-1]
    degrees = list(map(len, I))
    
    state_space = np.zeros((2**N, N), dtype=int)
    for i in range(N):
        for j, x in enumerate(itertools.product([0, 1], repeat=degrees[i])):
            if F[i][j]:
                # For rows in left_side_of_truth_table where the columns I[i] equal x, set state_space accordingly.
                state_space[np.all(left_side_of_truth_table[:, I[i]] == np.array(x), axis=1), i] = 1
    dictF = dict(zip(list(range(2**N)), np.dot(state_space, utils.bin2dec)))
    
    attractors = []
    basin_sizes = []
    attractor_dict = dict()
    for xdec in range(2**N):
        queue = [xdec]
        while True:
            fxdec = dictF[xdec]
            try:
                index_attr = attractor_dict[fxdec]
                basin_sizes[index_attr] += 1
                attractor_dict.update(list(zip(queue, [index_attr] * len(queue))))
                break
            except KeyError:
                try:
                    index = queue.index(fxdec)
                    attractor_dict.update(list(zip(queue, [len(attractors)] * len(queue))))
                    attractors.append(queue[index:])
                    basin_sizes.append(1)
                    break
                except ValueError:
                    pass
            queue.append(fxdec)
            xdec = fxdec
    if RETURN_DICTF:
        return (attractors, len(attractors), basin_sizes, attractor_dict, state_space, dictF)        
    else:
        return (attractors, len(attractors), basin_sizes, attractor_dict, state_space)

# def get_specific_coherence_values(attractors,attractor_dict,dictF,left_side_of_truth_table=None):
#     N = int(np.log2(len(dictF)))
#     if left_side_of_truth_table==None:
#         left_side_of_truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=N)))))
#     #attractors, n_attractors, basin_sizes, attractor_dict, state_space, dictF = get_attractors_synchronous_exact(F, I, left_side_of_truth_table,True)
#     attracting_states = flatten(attractors)
#     height = dict(zip(attracting_states,[0]*len(attracting_states)))
#     remaining_states = set(dictF.keys()) - set(attracting_states)
#     while len(remaining_states)>0:
#         for key in list(remaining_states):
#             try:
#                 height_key = height[dictF[key]] + 1
#                 height.update({key:height_key})
#                 remaining_states.discard(key)
#             except KeyError:
#                 pass
            
#     flowthrough = 
            
#     coherence_per_state = np.zeros(2**N)
#     utils.bin2dec = np.array([2**i for i in range(N)])[::-1]
#     for xdec, x in enumerate(left_side_of_truth_table): #iterate over each edge of the n-dim Hypercube once
#         for i in range(N):
#             if x[i] == 0:
#                 ydec = xdec + utils.bin2dec[i]
#             else: #to ensure we are not double-counting each edge
#                 continue
#             index_attr_x = attractor_dict[xdec]
#             index_attr_y = attractor_dict[ydec]
#             if index_attr_x == index_attr_y:
#                 coherence_per_state[xdec] += 1
#                 coherence_per_state[ydec] += 1
#     coherence_per_state /= N



def get_proportion_of_largest_basin_size(basin_sizes):
    """
    Compute the proportion of the largest basin size relative to the total basin sizes.

    This function calculates the ratio of the largest basin size to the sum of all basin sizes.
    This metric is useful for assessing the dominance of a particular attractor’s basin in a Boolean network.

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


def get_robustness_from_attractor_dict_exact(attractor_dict, N, n_attractors, left_side_of_truth_table):
    """
    Compute the robustness (coherence) of a Boolean network based on its attractor dictionary.

    This function computes the proportion of neighbors in the Boolean hypercube that, following a synchronous update,
    transition to the same attractor. For each state in the fully sampled state space (left_side_of_truth_table),
    it examines all N neighbors (each obtained by flipping one bit) and counts how many have the same attractor
    as the current state. The robustness is then given as the fraction of such edges over the total number of edges
    in the hypercube (which is 2^(N-1) * N).

    Parameters:
        attractor_dict (dict): A dictionary mapping each state (in decimal representation) to its attractor index.
                                 This dictionary must be computed from a fully sampled state space.
        N (int): The number of nodes in the network.
        n_attractors (int): The total number of attractors (not directly used in computation, but provided for context).
        left_side_of_truth_table (list or array): The full truth table of the network states, where each entry is a numpy array
                                                    representing one state (of length N).

    Returns:
        float: The robustness measure (i.e., the proportion of neighboring states that transition to the same attractor).
    """
    if n_attractors == 1:
        return 1
    utils.bin2dec = np.array([2**i for i in range(N)])[::-1]
    count_of_neighbors_who_transition_to_same_attractor = 0
    for xdec, x in enumerate(left_side_of_truth_table):
        for i in range(N):
            if x[i] == 0:
                ydec = xdec + utils.bin2dec[i]
            else:
                continue
            if attractor_dict[xdec] == attractor_dict[ydec]:
                count_of_neighbors_who_transition_to_same_attractor += 1
    return count_of_neighbors_who_transition_to_same_attractor / (2**(N-1) * N)

def get_robustness_measures_and_attractors_synchronous_exact(F, I, left_side_of_truth_table=None):
    """
    Compute the robustness (coherence) of a Boolean network based on its attractor dictionary.

    This function computes the exact attractors and robustness (coherence and fragility) of each basin of attraction and of each attractor.

    Parameters:
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of lists, where I[i] contains the indices of the regulators for node i.
        left_side_of_truth_table (np.array, optional): Precomputed array of all 2^N states (each row is a state).
                                                        If None, it is generated.

    Returns:
        tuple: A tuple containing:
            - attractors (list): List of attractors (each attractor is represented as a list of state decimal numbers).
            - exact_number_of_attractors (int): The lower bound on the number of attractors found.
            - exact_basin_sizes (list): List of basin sizes for each attractor.
            - attractor_dict (dict): Dictionary mapping each state (in decimal) to its attractor index.
            - state_space (np.array): The constructed state space matrix (of shape (2^N, N)).
            - exact_basin_coherence (list): coherence of each basin.
            - exact_basin_fragility (list): fragility of each basin.
            - attractor_coherence (list): attractor coherence of each basin (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
            - attractor_fragility (list): attractor fragility of each basin  (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
    """
    N = len(F)
    
    if left_side_of_truth_table is None:
        left_side_of_truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=N)))))
    
    attractors, n_attractors, basin_sizes, attractor_dict, state_space = get_attractors_synchronous_exact(F, I, left_side_of_truth_table=left_side_of_truth_table)
    
    len_attractors = list(map(len,attractors))
    
    
    if n_attractors == 1:
        return (attractors, n_attractors, np.array(basin_sizes)/2**N, attractor_dict, state_space, np.ones(1), np.zeros(1), np.ones(1), np.zeros(1))
    
    mean_states_attractors = []
    is_attr_dict = dict()
    for i in range(n_attractors):
        if len_attractors[i] == 1:
            mean_states_attractors.append(np.array(utils.dec2bin(attractors[i][0], N)))
        else:
            states_attractors = np.array([utils.dec2bin(state, N) for state in attractors[i]])
            mean_states_attractors.append(states_attractors.mean(0))
        for state in attractors[i]:
            is_attr_dict.update({state:i})
        
    distance_between_attractors = np.zeros((n_attractors,n_attractors),dtype=int)
    for i in range(n_attractors):
        for j in range(i+1,n_attractors):
            distance_between_attractors[i,j] = np.sum(np.abs(mean_states_attractors[i] - mean_states_attractors[j]))
            distance_between_attractors[j,i] = distance_between_attractors[i,j]
    distance_between_attractors = distance_between_attractors/N
    
    basin_coherences = np.zeros(n_attractors)
    basin_fragility = np.zeros(n_attractors)
    attractor_coherences = np.zeros(n_attractors)
    attractor_fragility = np.zeros(n_attractors)
    
    utils.bin2dec = np.array([2**i for i in range(N)])[::-1]
    for xdec, x in enumerate(left_side_of_truth_table): #iterate over each edge of the n-dim Hypercube once
        for i in range(N):
            if x[i] == 0:
                ydec = xdec + utils.bin2dec[i]
            else: #to ensure we are not double-counting each edge
                continue
            index_attr_x = attractor_dict[xdec]
            index_attr_y = attractor_dict[ydec]
            if index_attr_x == index_attr_y:
                basin_coherences[index_attr_x] += 1
                basin_coherences[index_attr_y] += 1
                try:
                    is_attr_dict[xdec]
                    attractor_coherences[index_attr_x] += 1
                except KeyError:
                    pass
                try:
                    is_attr_dict[ydec]
                    attractor_coherences[index_attr_y] += 1
                except KeyError:
                    pass
            else:
                normalized_Hamming_distance = distance_between_attractors[index_attr_x,index_attr_y]
                basin_fragility[index_attr_x] += normalized_Hamming_distance
                basin_fragility[index_attr_y] += normalized_Hamming_distance
                try:
                    is_attr_dict[xdec]
                    attractor_fragility[index_attr_x] += normalized_Hamming_distance
                except KeyError:
                    pass
                try:
                    is_attr_dict[ydec]
                    attractor_fragility[index_attr_y] += normalized_Hamming_distance
                except KeyError:
                    pass
                
    #normalizations
    for i,(basin_size,length_attractor) in enumerate(zip(basin_sizes,len_attractors)):
        basin_coherences[i] = basin_coherences[i] / basin_size / N
        basin_fragility[i] = basin_fragility[i] / basin_size / N
        attractor_coherences[i] = attractor_coherences[i] / length_attractor / N
        attractor_fragility[i] = attractor_fragility[i] / length_attractor / N
    basin_sizes = np.array(basin_sizes)/2**N
    
    return (attractors, n_attractors, basin_sizes, attractor_dict, state_space, basin_coherences, basin_fragility, attractor_coherences, attractor_fragility)


def get_robustness_measures_and_attractors(F, I, number_different_IC=500, RETURN_ATTRACTOR_COHERENCE = False):
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
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of lists, where I[i] contains the indices of the regulators for node i.
        number_different_IC (int, optional): Number of different initial conditions to sample (default is 500).
        RETURN_ATTRACTOR_COHERENCE (bool, optional): Determines whether the attractor coherence should also be computed (default is No, i.e., False).

    Returns:
        tuple: A tuple containing:
            - attractors (list): List of attractors (each attractor is represented as a list of state decimal numbers).
            - lower_bound_number_of_attractors (int): The lower bound on the number of attractors found.
            - approximate_basin_sizes (list): List of basin sizes for each attractor.
            - approximate_coherence (float): The approximated global robustness (coherence) measure.
            - approximate_fragility (float): The approximated fragility measure (averaged per node).
            - final_hamming_distance_approximation (float): The approximated final Hamming distance measure.
            - approximate_basin_coherence (list): coherence of each basin.
            - approximate_basin_fragility (list): fragility of each basin.
            - attractor_coherence (list): attractor coherence of each basin (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
            - attractor_fragility (list): attractor fragility of each basin  (only computed and returned if RETURN_ATTRACTOR_COHERENCE == True).
    """
    import math
    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b)
    
    N = len(F)
    
    dictF = dict()
    attractors = []
    ICs_per_attractor_state = []
    basin_sizes = []
    attractor_dict = dict()
    attractor_state_dict = []
    distance_from_attractor_state_dict = []
    counter_phase_shifts = []
    
    height = []
    degrees = list(map(len, I))
    
    utils.bin2decs = [np.array([2**i for i in range(NN)])[::-1] for NN in range(max(degrees)+1)]
    if N<64:
        utils.bin2dec = np.array([2**i for i in range(N)])[::-1]
    
    robustness_approximation = 0
    fragility_sum = 0
    basin_robustness = defaultdict(float)
    basin_fragility = defaultdict(float)
    final_hamming_distance_approximation = 0
    mean_states_attractors = []
    states_attractors = []
    
    for i in range(number_different_IC):
        index_attractors = []
        index_of_state_within_attractor_reached = []
        distance_from_attractor = []
        for j in range(2):
            if j == 0:
                x = np.random.randint(2, size=N)
                if N<64:
                    xdec = np.dot(x, utils.bin2dec)
                else: #out of range of np.int64
                    xdec = ''.join(str(bit) for bit in x)
                x_old = x.copy()
            else:
                x = x_old
                random_flipped_bit = np.random.choice(N)
                x[random_flipped_bit] = 1 - x[random_flipped_bit]
                if N<64:
                    xdec = np.dot(x, utils.bin2dec)
                else: #out of range of np.int64
                    xdec = ''.join(str(bit) for bit in x)               
            queue = [xdec]
            try:
                index_attr = attractor_dict[xdec]
            except KeyError:
                while True:
                    try: #check if we already know F(xdec)
                        fxdec = dictF[xdec]
                    except KeyError: #if not, then compute the F(xdec)
                        fx = []
                        for jj in range(N):
                            if degrees[jj]>0:
                                fx.append(F[jj][np.dot(x[I[jj]], utils.bin2decs[degrees[jj]])])
                            else:#constant functions whose regulators were all fixed to a specific value
                                fx.append(F[jj][0])
                        if N<64:
                            fxdec = np.dot(fx, utils.bin2dec)
                        else:
                            fxdec = ''.join(str(bit) for bit in fx)               
                        dictF.update({xdec: fxdec})
                    try: #check if we already know the attractor of F(xdec) 
                        index_attr = attractor_dict[fxdec]
                        dummy_index_within_attractor_reached = attractor_state_dict[index_attr][fxdec]
                        dummy_distance_from_attractor = distance_from_attractor_state_dict[index_attr][fxdec]
                        attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                        attractor_state_dict[index_attr].update(list(zip(queue, [dummy_index_within_attractor_reached]*len(queue))))
                        distance_from_attractor_state_dict[index_attr].update(
                            list(zip(queue, list(range(len(queue) + dummy_distance_from_attractor, dummy_distance_from_attractor, -1))))
                        )
                        break
                    except KeyError: 
                        try: #if not, then check if F(xdec) is already in the queue, i.e., if F(xdec) is part of an attractor itself
                            index = queue.index(fxdec)
                            index_attr = len(attractors)
                            attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                            attractors.append(queue[index:])
                            basin_sizes.append(1)
                            attractor_state_dict.append(dict(zip(queue, [0]*index + list(range(len(attractors[-1])))))
                            )
                            distance_from_attractor_state_dict.append(
                                dict(zip(queue, list(range(index, 0, -1)) + [0]*len(attractors[-1])))
                            )
                            ICs_per_attractor_state.append([0] * len(attractors[-1]))
                            counter_phase_shifts.append([0] * len(attractors[-1]))

                            if len(attractors[-1]) == 1:
                                if N<64:
                                    fixed_point = np.array(utils.dec2bin(queue[index], N))
                                else:
                                    fixed_point = np.array(list(queue[index]), dtype=int)
                                states_attractors.append(fixed_point.reshape((1, N)))
                                mean_states_attractors.append(fixed_point)
                            else:
                                if N<64:
                                    limit_cycle = np.array([utils.dec2bin(state, N) for state in queue[index:]])
                                else:
                                    limit_cycle = np.array([np.array(list(state), dtype=int) for state in queue[index:]])          
                                states_attractors.append(limit_cycle)
                                mean_states_attractors.append(limit_cycle.mean(0))
                            break
                        except ValueError: #if not, proceed by setting x = F(x)
                            x = np.array(fx)
                    queue.append(fxdec)
                    xdec = fxdec

            index_attractors.append(index_attr)
            index_of_state_within_attractor_reached.append(attractor_state_dict[index_attr][xdec])
            distance_from_attractor.append(distance_from_attractor_state_dict[index_attr][xdec])
            basin_sizes[index_attr] += 1
            ICs_per_attractor_state[index_attr][attractor_state_dict[index_attr][xdec]] += 1
        if index_attractors[0] == index_attractors[1]:
            robustness_approximation += 1
            basin_robustness[index_attractors[0]] += 1
            length_phaseshift = max(index_of_state_within_attractor_reached) - min(index_of_state_within_attractor_reached)
            counter_phase_shifts[index_attr][length_phaseshift] += 1
        else:
            fragility_sum += np.sum(np.abs(mean_states_attractors[index_attractors[0]] - mean_states_attractors[index_attractors[1]]))
            basin_fragility[index_attractors[0]] += np.sum(np.abs(mean_states_attractors[index_attractors[0]] - mean_states_attractors[index_attractors[1]]))
            required_n_states = lcm(len(attractors[index_attractors[0]]), len(attractors[index_attractors[1]]))
            index_j0 = index_of_state_within_attractor_reached[0]
            periodic_states_j0 = np.tile(states_attractors[index_attractors[0]], 
                                         (required_n_states // len(attractors[index_attractors[0]]) + 1, 1))[index_j0:(index_j0 + required_n_states), :]
            index_j1 = index_of_state_within_attractor_reached[1]
            periodic_states_j1 = np.tile(states_attractors[index_attractors[1]], 
                                         (required_n_states // len(attractors[index_attractors[1]]) + 1, 1))[index_j1:(index_j1 + required_n_states), :]
            final_hamming_distance_approximation += np.mean(periodic_states_j1 == periodic_states_j0)
            
        height.extend(distance_from_attractor)
    
    lower_bound_number_of_attractors = len(attractors)
    approximate_basin_sizes = np.array(basin_sizes)
    approximate_coherence = robustness_approximation * 1.0 / number_different_IC
    approximate_fragility = fragility_sum * 1.0 / number_different_IC / N
    
    approximate_basin_coherence = np.array([basin_robustness[index_att] * 2.0 / basin_sizes[index_att] for index_att in range(len(attractors))])
    approximate_basin_fragility = np.array([basin_fragility[index_att] * 2.0 / basin_sizes[index_att] / N for index_att in range(len(attractors))])
    
    for index_attr in range(len(attractors)):
        periodic_states_two_periods = np.tile(states_attractors[index_attr], (2, 1))
        for length_phaseshift, num_IC_with_that_phaseshift in enumerate(counter_phase_shifts[index_attr]):
            if num_IC_with_that_phaseshift > 0 and length_phaseshift > 0:
                final_hamming_distance_approximation += num_IC_with_that_phaseshift * np.mean(
                    states_attractors[index_attr] ==
                    periodic_states_two_periods[length_phaseshift:(length_phaseshift + len(attractors[index_attr])), :]
                )
                
    final_hamming_distance_approximation = final_hamming_distance_approximation / number_different_IC
    
    #fixing the results here because the subsequent attractor coherence computation could in theory identify additional attractors, 
    #which would screw things up because the attractor regions of the state space have then been oversampled
    results = [attractors, lower_bound_number_of_attractors, approximate_basin_sizes/2./number_different_IC, 
               approximate_coherence, approximate_fragility, final_hamming_distance_approximation,
               approximate_basin_coherence, approximate_basin_fragility]
    if RETURN_ATTRACTOR_COHERENCE == False:
        return tuple(results)
    else:
        attractor_coherence = np.zeros(lower_bound_number_of_attractors)
        attractor_fragility = np.zeros(lower_bound_number_of_attractors)
        attractors_original = attractors[:] #needed because new attractors may be found
        for index_attr_original,attractor in enumerate(attractors_original):
            for attractor_state in attractor: #perturb each attractor state
                for i in range(N):
                    if N<64:
                        x = np.array(utils.dec2bin(attractor_state,N))
                    else:
                        x = np.array(list(attractor_state), dtype=int)
                    x[i] = 1 - x[i]
                    if N<64:
                        xdec = np.dot(x, utils.bin2dec)
                    else:
                        xdec = ''.join(str(bit) for bit in x)
                    queue = [xdec]
                    try:
                        index_attr = attractor_dict[xdec]
                    except KeyError:
                        while True:
                            try: #check if we already know F(xdec)
                                fxdec = dictF[xdec]
                            except KeyError: #if not, then compute the F(xdec)
                                fx = []
                                for jj in range(N):
                                    if degrees[jj]>0:
                                        fx.append(F[jj][np.dot(x[I[jj]], utils.bin2decs[degrees[jj]])])
                                    else:#constant functions whose regulators were all fixed to a specific value
                                        fx.append(F[jj][0])
                                if N<64:
                                    fxdec = np.dot(fx, utils.bin2dec)
                                else:
                                    fxdec = ''.join(str(bit) for bit in fx)               
                                dictF.update({xdec: fxdec})
                            try: #check if we already know the attractor of F(xdec) 
                                index_attr = attractor_dict[fxdec]
                                dummy_index_within_attractor_reached = attractor_state_dict[index_attr][fxdec]
                                dummy_distance_from_attractor = distance_from_attractor_state_dict[index_attr][fxdec]
                                attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                attractor_state_dict[index_attr].update(list(zip(queue, [dummy_index_within_attractor_reached]*len(queue))))
                                distance_from_attractor_state_dict[index_attr].update(
                                    list(zip(queue, list(range(len(queue) + dummy_distance_from_attractor, dummy_distance_from_attractor, -1))))
                                )
                                break
                            except KeyError: 
                                try: #if not, then check if F(xdec) is already in the queue, i.e., if F(xdec) is part of an attractor itself
                                    index = queue.index(fxdec)
                                    index_attr = len(attractors)
                                    attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                    attractors.append(queue[index:])
                                    #basin_sizes.append(1)
                                    attractor_state_dict.append(dict(zip(queue, [0]*index + list(range(len(attractors[-1])))))
                                    )
                                    distance_from_attractor_state_dict.append(
                                        dict(zip(queue, list(range(index, 0, -1)) + [0]*len(attractors[-1])))
                                    )
                                    ICs_per_attractor_state.append([0] * len(attractors[-1]))
                                    counter_phase_shifts.append([0] * len(attractors[-1]))
        
                                    if len(attractors[-1]) == 1:
                                        if N<64:
                                            fixed_point = np.array(utils.dec2bin(queue[index], N))
                                        else:
                                            fixed_point = np.array(list(queue[index]), dtype=int)
                                        states_attractors.append(fixed_point.reshape((1, N)))
                                        mean_states_attractors.append(fixed_point)
                                    else:
                                        if N<64:
                                            limit_cycle = np.array([utils.dec2bin(state, N) for state in queue[index:]])
                                        else:
                                            limit_cycle = np.array([np.array(list(state), dtype=int) for state in queue[index:]])          
                                        states_attractors.append(limit_cycle)
                                        mean_states_attractors.append(limit_cycle.mean(0))
                                    break
                                except ValueError: #if not, proceed by setting x = F(x)
                                    x = np.array(fx)
                            queue.append(fxdec)
                            xdec = fxdec
                    if index_attr_original == index_attr:
                        attractor_coherence[index_attr_original] += 1
                    else:
                        attractor_fragility[index_attr_original] += np.sum(np.abs(mean_states_attractors[index_attr_original] - mean_states_attractors[index_attr]))
        attractor_coherence = np.array([s/N/size_attr for s,size_attr in zip(attractor_coherence,map(len,attractors_original))])
        attractor_fragility = np.array([s/N**2/size_attr for s,size_attr in zip(attractor_fragility,map(len,attractors_original))]) #something is wrong with attractor fragility, it returns values > 1 for small basins
        results[0] = attractors_original
        return tuple(results + [attractor_coherence,attractor_fragility])
def flatten(l):
    return [item for sublist in l for item in sublist]

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
    def get_BN_with_fixed_source_nodes_fast(F,I,n_var,n_const,values_constants):
        #don't require a truth table look-up as long as constants are removed in reverse. 
        #Reason: I[i] is sorted and constants are the last entries in F
        assert len(F) == len(I)
        F_new = F[:n_var]
        I_new = I[:n_var]
        
        for constant,value in zip(list(range(n_var+n_const-1,n_var-1,-1)),values_constants):
            for i in range(n_var):
                if constant in I[i]:
                    if value==0:
                        F_new[i] = F_new[i][0::2]
                    else:
                        F_new[i] = F_new[i][1::2]                
                    I_new[i] = I_new[i][:-1]
        return F_new,I_new
    
    def get_BN_with_fixed_source_nodes(F,I,n_var,n_const,values_constants):
        assert len(F) == len(I)
        F_new = [np.array(el) for el in F[:n_var]]
        I_new = [np.array(el) for el in I[:n_var]]
        
        for constant,value in zip(list(range(n_var,n_var+n_const)),values_constants):
            for i in range(n_var):
                try:
                    index = list(I[i]).index(constant) #check if the constant is part of regulators
                except ValueError:
                    continue
                truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=len(I_new[i]))))))
                indices_to_keep = np.where(truth_table[:,index]==value)[0]
                F_new[i] = F_new[i][indices_to_keep]
                I_new[i] = I_new[i][~np.in1d(I_new[i], constant)]
        return F_new,I_new
    
    import math
    
    

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
    
    utils.bin2dec = np.array([2**i for i in range(N)])[::-1]
    
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
        dictF_with_fixed_source.append(dict(zip(list(range(2**N)), np.dot(state_space, utils.bin2dec))))
        
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
    return F,I,variables

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
                    all_input_patterns[-1].append(flatten(attractor[:,np.array(list(map(lambda x: pos_in_module_per_input_nodes[x],nodes_per_upstream_module_we_care_about[upstream_module_id])))]))

            F2 = [F[j] for j in nodes_in_module]
            I2 = [list(map(lambda x: pos_in_module[x],I[j])) for j in nodes_in_module]                
                
            module_attractors = get_attractors_synchronous_exact_with_external_inputs(F2, I2, input_patterns = all_input_patterns[0])[0]
            
            #next line needs to be checked
            module_attractors_binary[module_id] = [np.array(list(map(lambda x: utils.dec2bin(x,len(nodes_in_module)),module_attractors[i]))) for i in range(len(module_attractors))]
            
   
    return module_attractors_binary


def adjacency_matrix(I, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
    """
    Construct the (binary) adjacency matrix from the wiring diagram.

    Given the wiring diagram I (a list of regulator lists for each gene) and a list of constants,
    this function builds an adjacency matrix where each entry m[j, i] is 1 if gene j regulates gene i.
    Self-loops can be optionally ignored, and constant nodes can be excluded.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        constants (list, optional): List of constant nodes.
        IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
        IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded from the matrix.

    Returns:
        np.array: The binary adjacency matrix.
    """
    n = len(I)
    n_constants = len(constants)
    if IGNORE_CONSTANTS:
        m = np.zeros((n - n_constants, n - n_constants), dtype=int)
        for i in range(len(I)):
            for j in I[i]:
                if j < n - n_constants and (not IGNORE_SELFLOOPS or i != j):
                    m[j, i] = 1
        return m
    else:
        return adjacency_matrix(I, [], IGNORE_CONSTANTS=True)


def get_signed_adjacency_matrix(I, type_of_each_regulation, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
    """
    Construct the signed adjacency matrix of a Boolean network.

    The signed adjacency matrix assigns +1 for increasing (activating) regulations,
    -1 for decreasing (inhibiting) regulations, and NaN for any other type.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        type_of_each_regulation (list): List of lists corresponding to the type of regulation ('increasing' or 'decreasing')
                                        for each edge in I.
        constants (list, optional): List of constant nodes.
        IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
        IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded.

    Returns:
        np.array: The signed adjacency matrix.
    """
    n = len(I)
    n_constants = len(constants)
    if IGNORE_CONSTANTS:
        m = np.zeros((n - n_constants, n - n_constants), dtype=int)
        for i, (regulators, type_of_regulation) in enumerate(zip(I, type_of_each_regulation)):
            for j, t in zip(regulators, type_of_regulation):
                if j < n - n_constants and (not IGNORE_SELFLOOPS or i != j):
                    if t == 'increasing':
                        m[j, i] = 1 
                    elif t == 'decreasing':
                        m[j, i] = -1 
                    else:
                        m[j, i] = np.nan
        return m
    else:
        return get_signed_adjacency_matrix(I, type_of_each_regulation, [], IGNORE_CONSTANTS=True)


def get_signed_effective_graph(I, type_of_each_regulation, F, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
    """
    Construct the signed effective graph of a Boolean network.

    This function computes an effective graph in which each edge is weighted by its effectiveness.
    Effectiveness is obtained via get_edge_effectiveness on the corresponding Boolean function.
    Edges are signed according to the type of regulation ('increasing' or 'decreasing').

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        type_of_each_regulation (list): List of lists specifying the type of regulation for each edge.
        F (list): List of Boolean functions (truth tables) for each node.
        constants (list, optional): List of constant nodes.
        IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
        IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded.

    Returns:
        np.array: The signed effective graph as a matrix of edge effectiveness values.
    """
    n = len(I)
    n_constants = len(constants)
    if IGNORE_CONSTANTS:
        m = np.zeros((n - n_constants, n - n_constants), dtype=float)
        for i, (regulators, type_of_regulation) in enumerate(zip(I, type_of_each_regulation)):
            effectivenesses = analyze_BF.get_edge_effectiveness(F[i])
            for j, t, e in zip(regulators, type_of_regulation, effectivenesses):
                if j < n - n_constants and (not IGNORE_SELFLOOPS or i != j):
                    if t == 'increasing':
                        m[j, i] = e
                    elif t == 'decreasing':
                        m[j, i] = -e
                    else:
                        m[j, i] = np.nan
        return m
    else:
        return get_signed_effective_graph(I, type_of_each_regulation, F, [], IGNORE_CONSTANTS=True)


def get_ffls(I, F=None):
    """
    Identify feed-forward loops (FFLs) in a Boolean network and optionally determine their types.

    A feed-forward loop (FFL) is a three-node motif where node i regulates node k both directly and indirectly via node j.
    If F is provided, the function also computes the monotonicity of each regulation in the FFL.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        F (list, optional): List of Boolean functions (truth tables) for each node.
                             If provided along with F, the types (monotonicities) of the regulations are computed.

    Returns:
        If F is None:
            list: A list of FFLs, each represented as a list [i, k, j] (or similar ordering).
        Otherwise:
            tuple: A tuple (ffls, types), where ffls is a list of FFLs and types is a list of corresponding monotonicity types.
    """
    ffls = []
    types = []
    for i in range(len(I)):
        for j in range(i + 1, len(I)):
            for k in range(len(I)):
                if i == k or j == k:
                    continue
                # Check if there is an FFL: i regulates k and j regulates both i and k.
                if i in I[k] and i in I[j] and j in I[k]:
                    ffls.append([i, j, k])
                    if F is not None:
                        # Compute types if F is provided.
                        # (This example assumes a helper function is_monotonic exists and that I is ordered.)
                        #monotonic_i = is_monotonic(F[i], True)[1]
                        monotonic_j = analyze_BF.is_monotonic(F[j], True)[1]
                        monotonic_k = analyze_BF.is_monotonic(F[k], True)[1]
                        direct = monotonic_k[I[k].index(i)]
                        indirect1 = monotonic_j[I[j].index(i)]
                        indirect2 = monotonic_k[I[k].index(j)]
                        types.append([direct, indirect1, indirect2])
    if F is not None:
        return (ffls, types)
    else:
        return ffls


def get_ffls_from_I(I, types_I=None):
    """
    Identify feed-forward loops (FFLs) in a Boolean network based solely on the wiring diagram.

    The function uses the inverted wiring diagram to identify common targets and returns the FFLs found.
    If types_I (the type of each regulation) is provided, it also returns the corresponding regulation types.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        types_I (list, optional): List of lists specifying the type (e.g., 'increasing' or 'decreasing') for each regulation.

    Returns:
        If types_I is provided:
            tuple: (ffls, types) where ffls is a list of identified FFLs (each as a list [i, j, k]),
                   and types is a list of corresponding regulation type triplets.
        Otherwise:
            list: A list of identified FFLs.
    """
    all_tfs = list(range(len(I)))
    n_tfs = len(all_tfs)
    all_tfs_dict = dict(zip(all_tfs, list(range(n_tfs))))
    I_inv = [[] for _ in all_tfs]
    for target, el in enumerate(I):
        for regulator in el:
            I_inv[all_tfs_dict[regulator]].append(target)
    ffls = []
    types = []
    for i in range(n_tfs):  # master regulators
        for j in range(n_tfs):
            if i == j or all_tfs[j] not in I_inv[i]:
                continue
            common_targets = list(set(I_inv[i]) & set(I_inv[j]))
            for k in common_targets:
                if all_tfs[j] == k or all_tfs[i] == k:
                    continue
                ffls.append([i, j, k])
                if types_I is not None:
                    direct = types_I[k][I[k].index(all_tfs[i])]
                    indirect1 = types_I[all_tfs[j]][I[all_tfs[j]].index(all_tfs[i])]
                    indirect2 = types_I[k][I[k].index(all_tfs[j])]
                    types.append([direct, indirect1, indirect2])
    if types_I is not None:
        return (ffls, types)
    else:
        return ffls


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


def generate_networkx_graph(I, constants, variables):
    """
    Generate a NetworkX directed graph from a wiring diagram.

    Nodes are labeled with variable names (from variables) and constant names (from constants). Edges are added
    from each regulator to its target based on the wiring diagram I.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        constants (list): List of constant names.
        variables (list): List of variable names.

    Returns:
        networkx.DiGraph: The generated directed graph.
    """
    names = list(variables) + list(constants)
    G = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_edges_from([(names[I[i][j]], names[i]) for i in range(len(variables)) for j in range(len(I[i]))])
    return G


def generate_networkx_graph_from_edges(I, n_variables):
    """
    Generate a NetworkX directed graph from an edge list derived from the wiring diagram.

    Only edges among the first n_variables (excluding constant self-loops) are included.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        n_variables (int): Number of variable nodes (constants are excluded).

    Returns:
        networkx.DiGraph: The generated directed graph.
    """
    edges = []
    for j, regulators in enumerate(I):
        if j >= n_variables:  # Exclude constant self-loops
            break
        for i in regulators:
            edges.append((i, j))
    return nx.DiGraph(edges)


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
    from collections import defaultdict

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


def get_type_of_loop(loop, F, I):
    """
    Determine the regulation types along a feedback loop.

    For a given loop (a list of node indices), this function returns a list containing
    the type (e.g., 'increasing' or 'decreasing') of each regulation along the loop.
    The loop is assumed to be ordered such that the first node is repeated at the end.

    Parameters:
        loop (list): List of node indices representing the loop.
        F (list): List of Boolean functions (truth tables) for each node.
        I (list): List of regulator indices for each node.

    Returns:
        list: A list of regulation types corresponding to each edge in the loop.
    """
    n = len(loop)
    dummy = loop[:]
    dummy.append(loop[0])
    res = []
    for i in range(n):
        # Assumes is_monotonic returns a tuple with the monotonicity information.
        res.append(analyze_BF.is_monotonic(F[dummy[i+1]], True)[1][list(I[dummy[i+1]]).index(dummy[i])])
    return res


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


def generate_networkx_graph(I, constants, variables):
    """
    Generate a NetworkX directed graph from a wiring diagram.

    Nodes are labeled using the provided variable and constant names.
    Edges are added based on the wiring diagram I.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        constants (list): List of constant names.
        variables (list): List of variable names.

    Returns:
        networkx.DiGraph: The generated directed graph.
    """
    names = list(variables) + list(constants)
    G = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_edges_from([(names[I[i][j]], names[i]) for i in range(len(variables)) for j in range(len(I[i]))])
    return G


def generate_networkx_graph_from_edges(I, n_variables):
    """
    Generate a NetworkX directed graph from an edge list derived from the wiring diagram.

    Only edges among the first n_variables (i.e., non-constant nodes) are included.

    Parameters:
        I (list): List of lists, where I[i] contains the indices of regulators for gene i.
        n_variables (int): Number of variable nodes (constants are excluded).

    Returns:
        networkx.DiGraph: The resulting directed graph.
    """
    edges = []
    for j, regulators in enumerate(I):
        if j >= n_variables:  # Exclude constant self-loops.
            break
        for i in regulators:
            edges.append((i, j))
    return nx.DiGraph(edges)


















