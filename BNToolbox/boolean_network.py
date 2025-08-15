#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 11:08:44 2025
Last Edited on Thu Aug 14 2025

@author: Benjamin Coberly
"""

import numpy as np
import itertools
import networkx as nx
import random
import math
from collections import defaultdict

import utils
from boolean_function import BooleanFunction as BF

try:
    import cana.boolean_network
    __LOADED_CANA__=True
except ModuleNotFoundError:
    print('The module cana cannot be found. Ensure it is installed to use all functionality of this toolbox.')
    __LOADED_CANA__=False

def from_cana_BooleanNetwork(BooleanNetwork):
    return BooleanNetwork(F = xxx, I = xxx)#TODO: figure out what exactly to pass

def pyboolnet_bnet_to_BooleanNetwork(bnet):
    """
    Compatability method: Transforms a bnet object from the pyboolnet library to an instance of the class BooleanNetwork, used in this toolbox.

    Returns:
        An instance of BooleanFunction
    """
    variables = []
    functions = []
    for line in bnet.split('\n'):
        try:
            functions.append(line.split(',')[1].strip())
            variables.append(line.split(',')[0])
        except IndexError:
            continue
    dict_variables = dict(zip(variables,range(len(variables))))
    F = []
    I = []
    for function in functions:
        f,var = utils.f_from_expression(function)
        F.append(f)
        I.append([dict_variables[v] for v in var])        
    return BooleanNetwork(F=F,I=I,variables=variables)


class BooleanNetwork:
    def __init__(self, F, I, variables=None, left_side_of_truth_table = None):
        assert type(F) in [ list, np.array, np.ndarray ], "F must be an array"
        assert type(I) in [ list, np.array, np.ndarray ], "I must be an array"
        #assert (len(I[i]) == ns[i] for i in range(len(ns))), "Malformed wiring diagram I"
        assert variables is None or len(F)==len(variables), "len(F)==len(variables) required if variable names are provided"
        assert len(F)==len(I), "len(F)==len(I) required"
        
        self.F = []
        for f in F:
            if type(f) in [ list, np.array ]:
                self.F.append(BF(f))
            elif type(f) == BF:
                self.F.append(f)
            else:
                raise TypeError(f"F holds invalid data type {type(f)} : Expected either list, numpy array, or BooleanFunction")
                
        self.N = len(F)
        if variables is None:
            self.variables = np.array(['x'+str(i) for i in range(len(F))])
        else:
            self.variables = np.array(variables)
        self.I = [np.array(regulators,dtype=int) for regulators in I]
        self.lstt = left_side_of_truth_table #TODO: pass as argument in the two functions that use it
    
    def to_cana_BooleanNetwork(self):
        return cana.boolean_network.BooleanNetwork(xxxxx)#TODO: figure out what exactly to pass

    def to_pyboolnet_bnet(self):
        lines = []
        for bf,regulators,variable in zip(self.F,self.I,self.variables):
            polynomial = utils.bool_to_poly(bf.f,variables=self.variables[regulators])
            lines.append(f'{variable},    {polynomial}')
        return '\n'.join(lines)
        
    def update_single_node(self, index, states_regulators):
        """
        Update the state of a single node.

        The new state is obtained by applying the Boolean function f to the states of its regulators.
        The regulator states are converted to a decimal index using utils.bin2dec.

        Parameters:
            bf (BooleanFunction): Boolean function object.
            states_regulators (list or np.array): Binary vector representing the states of the node's regulators.

        Returns:
            int: Updated state of the node (0 or 1).
        """
        return self.F[index].f[utils.bin2dec(states_regulators)]


    def update_network_synchronously(self, X):
        """
        Perform a synchronous update of a Boolean network.

        Each node's new state is determined by applying its Boolean function to the current states of its regulators.

        Parameters:
            X (list or np.array): Current state vector of the network.

        Returns:
            np.array: New state vector after the update.
        """
        if type(X)==list:
            X = np.array(X)
        Fx = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            Fx[i] = self.update_single_node(index = i, states_regulators = X[self.I[i]])
        return Fx


    def update_network_synchronously_many_times(self, X, n_steps):
        """
        Update the state of a Boolean network sychronously multiple time steps.

        Starting from the initial state, the network is updated synchronously n_steps times using the update_network_synchronously function.

        Parameters:
            X (list or np.array): Initial state vector of the network.
            n_steps (int): Number of update iterations to perform.

        Returns:
            np.array: Final state vector after n_steps updates.
        """
        for i in range(n_steps):
            X = self.update_network_synchronously(X)
        return X


    def update_network_SDDS(self, X, P):
        """
        Perform a stochastic update (SDDS) on a Boolean network.

        For each node, the next state is computed as nextstep = F[i] evaluated on the current states of its regulators.
        If nextstep > X[i], the node is activated with probability P[i,0]; if nextstep < X[i],
        the node is degraded with probability P[i,1]. Otherwise, the state remains unchanged.

        Parameters:
            X (list or np.array): Current state vector.
            P (np.array): A len(F)×2 array of probabilities; for each node i, P[i,0] is the activation probability,
                          and P[i,1] is the degradation probability.

        Returns:
            np.array: Updated state vector after applying the stochastic update.
            
        References:
            
        """
        if type(X)==list:
            X = np.array(X)
        Fx = X.copy()
        for i in range(self.N):
            nextstep = self.update_single_node(index = i, states_regulators = X[self.I[i]])
            if nextstep > X[i] and random.random() < P[i, 0]:  # activation
                Fx[i] = nextstep
            elif nextstep < X[i] and random.random() < P[i, 1]:  # degradation
                Fx[i] = nextstep
        return Fx


    def get_steady_states_asynchronous(self, nsim=500, EXACT=False, 
                                       initial_sample_points=[], search_depth=50, SEED=-1, DEBUG=False):
        """
        Compute the steady states of a Boolean network under asynchronous updates.

        This function simulates asynchronous updates of a Boolean network (with N nodes)
        for a given number of initial conditions (nsim). For each initial state, the network
        is updated asynchronously until a steady state (or attractor) is reached or until a maximum
        search depth is exceeded. The simulation can be performed either approximately (by sampling nsim
        random initial conditions) or exactly (by iterating over the entire state space when EXACT=True).

        Parameters:
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
        if EXACT and self.lstt is None:
            self.lstt = list(map(np.array, list(itertools.product([0, 1], repeat=self.N))))

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
        
        for iteration in range(nsim if not EXACT else 2**self.N):
            if EXACT:
                x = self.lstt[iteration]
                xdec = iteration
            else:
                if initial_sample_points == []:  # generate random initial states on the fly
                    x = np.random.randint(2, size=self.N)
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
                    update_order_to_try = np.random.permutation(self.N)
                    for i in update_order_to_try:
                        try:
                            fxdec = dictF[(xdec, i)]
                            if fxdec != xdec:
                                FOUND_NEW_STATE = True
                                x[i] = 1 - x[i]
                        except KeyError:
                            fx_i = self.update_single_node(i, x[self.I[i]])
                            if fx_i > x[i]:
                                fxdec = xdec + 2**(self.N - 1 - i)
                                x[i] = 1
                                FOUND_NEW_STATE = True
                            elif fx_i < x[i]:
                                fxdec = xdec - 2**(self.N - 1 - i)
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
        if sum(basin_sizes) < (nsim if not EXACT else 2**self.N):
            print('Warning: only %i of the %i tested initial conditions eventually reached a steady state. Try increasing the search depth. '
                  'It may however also be the case that your asynchronous state space contains a limit cycle.' %
                  (sum(basin_sizes), nsim if not EXACT else 2**self.N))
        return (steady_states, len(steady_states), basin_sizes, steady_state_dict, dictF, SEED,
                initial_sample_points if initial_sample_points != [] else sampled_points)


    def get_steady_states_asynchronous_given_one_initial_condition(self, nsim=500, stochastic_weights=[], initial_condition=0, search_depth=50, SEED=-1, DEBUG=False):
        """
        Determine the steady states reachable from one initial condition using weighted asynchronous updates.

        This function is similar to steady_states_asynchronous_given_one_IC but allows the update order
        to be influenced by provided stochastic weights (one per node). A weight vector (of length N) may be provided,
        and if given, it is normalized and used to bias the random permutation of node update order.
        
        Parameters:
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
        
        if type(initial_condition) == int:
            initial_condition = np.array(utils.dec2bin(initial_condition, self.N))
            initial_condition_bin = utils.bin2dec(initial_condition)
        else:
            initial_condition = np.array(initial_condition, dtype=int)
            initial_condition_bin = utils.bin2dec(initial_condition)
        
        assert stochastic_weights == [] or len(stochastic_weights) == self.N, "one stochastic weight per node is required"    
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
                        update_order_to_try = np.random.choice(self.N, size=self.N, replace=False, p=stochastic_weights)
                    else:
                        update_order_to_try = np.random.permutation(self.N)
                    for i in update_order_to_try:
                        try:
                            fxdec = dictF[(xdec, i)]
                            if fxdec != xdec:
                                FOUND_NEW_STATE = True
                                x[i] = 1 - x[i]
                        except KeyError:
                            fx_i = self.update_single_node(i, x[self.I[i]])
                            if fx_i > x[i]:
                                fxdec = xdec + 2**(self.N - 1 - i)
                                x[i] = 1
                                FOUND_NEW_STATE = True
                            elif fx_i < x[i]:
                                fxdec = xdec - 2**(self.N - 1 - i)
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


    def get_attractors_synchronous(self, nsim=500, initial_sample_points=[], n_steps_timeout=100000,
                                   INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS=True):
        """
        Compute the number of attractors in a Boolean network using an alternative (v2) approach.

        This version is optimized for networks with longer average path lengths. For each of nb initial conditions,
        the network is updated synchronously until an attractor is reached or until n_steps_timeout is exceeded.
        The function returns the attractors found, their basin sizes, a mapping of states to attractors,
        the set of initial sample points used, the explored state space, and the number of simulations that timed out.

        Parameters:
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
        dictF = dict()
        attractors = []
        basin_sizes = []
        attr_dict = dict()
        state_space = []
        
        sampled_points = []
        n_timeout = 0
        
        INITIAL_SAMPLE_POINTS_EMPTY = utils.check_if_empty(initial_sample_points)
        if not INITIAL_SAMPLE_POINTS_EMPTY:
            nsim = len(initial_sample_points)
        
        for i in range(nsim):
            if INITIAL_SAMPLE_POINTS_EMPTY:
                x = np.random.randint(2, size=self.N)
                xdec = utils.bin2dec(x)
                sampled_points.append(xdec)
            else:
                if INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS:
                    x = initial_sample_points[i]
                    xdec = utils.bin2dec(x)
                else:
                    xdec = initial_sample_points[i]
                    x = np.array(utils.dec2bin(xdec, self.N))
            queue = [xdec]
            count = 0
            while count < n_steps_timeout:
                try:
                    fxdec = dictF[xdec]
                except KeyError:
                    fx = self.update_network_synchronously(x)
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


    def get_attractors_synchronous_exact(self, RETURN_DICTF=False):
        """
        Compute the exact number of attractors in a Boolean network using a fast, vectorized approach.

        This function computes the state of each node for all 2^N states by constructing the network's state space,
        then maps each state to its corresponding successor state via the Boolean functions F.
        Attractors and their basin sizes are then determined by iterating over the entire state space.

        Parameters:
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
        if self.lstt is None:
            self.lstt = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=self.N)))))
        
        powers_of_two = np.array([2**i for i in range(self.N)])[::-1]
        degrees = list(map(len, self.I))
        
        state_space = np.zeros((2**self.N, self.N), dtype=int)
        for i in range(self.N):
            for j, x in enumerate(itertools.product([0, 1], repeat=degrees[i])):
                if self.F[i].f[j]:
                    # For rows in left_side_of_truth_table where the columns I[i] equal x, set state_space accordingly.
                    state_space[np.all(self.lstt[:, self.I[i]] == np.array(x), axis=1), i] = 1
        dictF = dict(zip(list(range(2**self.N)), np.dot(state_space, powers_of_two)))
        
        attractors = []
        basin_sizes = []
        attractor_dict = dict()
        for xdec in range(2**self.N):
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


    ## Transform Boolean networks
    def get_essential_network(self):
        """
        Determine the essential components of a Boolean network.

        For each node in a Boolean network, represented by its Boolean function and its regulators,
        this function extracts the “essential” part of the function by removing non-essential regulators.
        The resulting network contains, for each node, a reduced truth table (with only the essential inputs)
        and a corresponding list of essential regulators.

        Returns:
            tuple: (F_essential, I_essential) where:
                - F_essential is a list of N Boolean functions (truth tables) of length 2^(m_i), with m_i ≤ n_i,
                  representing the functions restricted to the essential regulators.
                - I_essential is a list of N lists containing the indices of the essential regulators for each node.
        """
        F_essential = []
        I_essential = []
        for bf, regulators in zip(self.F, self.I):
            if len(bf.f) == 0:  # happens for biological networks if the actual degree of f was too large for it to be loaded
                F_essential.append(bf)
                I_essential.append(regulators) #keep all regulators (unable to determine if all are essential)
                continue
            elif sum(bf.f) == 0: #constant zero function
                F_essential.append(BF(np.array([0])))
                I_essential.append(np.array([], dtype=int))
                continue
            elif sum(bf.f) == len(bf.f): #constant one function
                F_essential.append(BF(np.array([1])))
                I_essential.append(np.array([], dtype=int))
                continue
            essential_variables = np.array(bf.get_essential_variables())
            n = len(regulators)
            non_essential_variables = np.array(list(set(list(range(n))) - set(essential_variables)))
            if len(non_essential_variables) == 0:
                F_essential.append(bf)
                I_essential.append(regulators)
            else:
                lstt = np.array(list(itertools.product([0, 1], repeat=n)))
                F_essential.append(BF(bf.f[np.sum(lstt[:, non_essential_variables], 1) == 0]))
                I_essential.append(np.array(regulators)[essential_variables])
        return F_essential, I_essential #TODO: return an instanc eof BooleanNetwork


    def get_edge_controlled_network(self, control_target, control_source, type_of_edge_control=0, left_side_of_truth_table=[]):
        """
        Generate a perturbed Boolean network by removing the influence of a specified regulator on a specified target.

        The function modifies the Boolean function for a target node by restricting it to those entries in its truth table
        where the input from a given regulator equals the specified type_of_control. The regulator is then removed from
        the wiring diagram for that node.

        Parameters:
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
        assert type_of_edge_control in [0,1], "type_of_edge_control must be 0 or 1."
        assert control_source in self.I[control_target], "control_source=%i does not regulate control_target=%i." % (control_source,control_target)
        
        F_new = [bf.f for bf in self.F]
        I_new = [i for i in self.I]

        if left_side_of_truth_table == []:
            left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=len(self.I[control_target]))))

        index = list(self.I[control_target]).index(control_source)
        F_new[control_target] = F_new[control_target][left_side_of_truth_table[:, index] == type_of_edge_control]
        dummy = list(I_new[control_target])
        dummy.remove(control_source)
        I_new[control_target] = np.array(dummy)
        return BooleanNetwork(F_new, I_new, self.variables)


    def get_external_inputs(self):
        """
        Identify external inputs in a Boolean network.

        A node is considered an external input if it has exactly one regulator and that regulator is the node itself.

        Returns:
            np.array: Array of node indices that are external inputs.
        """
        degrees = [len(el) for el in self.I]
        return np.array([i for i in range(self.N) if degrees[i] == 1 and self.I[i][0] == i])


    ## Robustness measures: synchronous Derrida value, entropy of basin size distribution, coherence, fragility
    def get_derrida_value(self, nsim=1000, EXACT = False):
        """
        Estimate the Derrida value for a Boolean network.

        The Derrida value is computed by perturbing a single node in a randomly chosen state and measuring
        the average Hamming distance between the resulting updated states of the original and perturbed networks.

        Parameters:
            nsim (int, optional): Number of simulations to perform. Default is 1000.

        Returns:
            float: The average Hamming distance (Derrida value) over nsim simulations.
        """
        if EXACT:
            return np.mean([bf.get_average_sensitivity(EXACT=True,NORMALIZED=False) for bf in self.F])
        else:
            hamming_distances = []
            for i in range(nsim):
                X = np.random.randint(0, 2, self.N)
                Y = X.copy()
                index = np.random.randint(self.N)
                Y[index] = 1 - Y[index]
                FX = self.update_network_synchronously(X)
                FY = self.update_network_synchronously(Y)
                hamming_distances.append(sum(FX != FY))
            return np.mean(hamming_distances)


    def get_attractors_and_robustness_measures_synchronous_exact(self):
        """
        Compute the attractors and several robustness measures of a Boolean network.

        This function computes the exact attractors and robustness (coherence and fragility) of each basin of attraction and of each attractor.

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
        if self.lstt is None:
            self.lstt = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=self.N)))))
        
        attractors, n_attractors, basin_sizes, attractor_dict, state_space = self.get_attractors_synchronous_exact()
        
        len_attractors = list(map(len,attractors))
        
        
        if n_attractors == 1:
            return (attractors, n_attractors, np.array(basin_sizes)/2**self.N, attractor_dict, state_space, np.ones(1), np.zeros(1), np.ones(1), np.zeros(1), 1, 0)
        
        mean_states_attractors = []
        is_attr_dict = dict()
        for i in range(n_attractors):
            if len_attractors[i] == 1:
                mean_states_attractors.append(np.array(utils.dec2bin(attractors[i][0], self.N)))
            else:
                states_attractors = np.array([utils.dec2bin(state, self.N) for state in attractors[i]])
                mean_states_attractors.append(states_attractors.mean(0))
            for state in attractors[i]:
                is_attr_dict.update({state:i})
            
        distance_between_attractors = np.zeros((n_attractors,n_attractors),dtype=int)
        for i in range(n_attractors):
            for j in range(i+1,n_attractors):
                distance_between_attractors[i,j] = np.sum(np.abs(mean_states_attractors[i] - mean_states_attractors[j]))
                distance_between_attractors[j,i] = distance_between_attractors[i,j]
        distance_between_attractors = distance_between_attractors/self.N
        
        basin_coherences = np.zeros(n_attractors)
        basin_fragilities = np.zeros(n_attractors)
        attractor_coherences = np.zeros(n_attractors)
        attractor_fragilities = np.zeros(n_attractors)
        
        powers_of_2 = np.array([2**i for i in range(self.N)])[::-1]
        for xdec, x in enumerate(self.lstt): #iterate over each edge of the n-dim Hypercube once
            for i in range(self.N):
                if x[i] == 0:
                    ydec = xdec + powers_of_2[i]
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
                    basin_fragilities[index_attr_x] += normalized_Hamming_distance
                    basin_fragilities[index_attr_y] += normalized_Hamming_distance
                    try:
                        is_attr_dict[xdec]
                        attractor_fragilities[index_attr_x] += normalized_Hamming_distance
                    except KeyError:
                        pass
                    try:
                        is_attr_dict[ydec]
                        attractor_fragilities[index_attr_y] += normalized_Hamming_distance
                    except KeyError:
                        pass
                    
        #normalizations
        for i,(basin_size,length_attractor) in enumerate(zip(basin_sizes,len_attractors)):
            basin_coherences[i] = basin_coherences[i] / basin_size / self.N
            basin_fragilities[i] = basin_fragilities[i] / basin_size / self.N
            attractor_coherences[i] = attractor_coherences[i] / length_attractor / self.N
            attractor_fragilities[i] = attractor_fragilities[i] / length_attractor / self.N
        basin_sizes = np.array(basin_sizes)/2**self.N
        
        coherence = np.dot(basin_sizes,basin_coherences)
        fragility = np.dot(basin_sizes,basin_fragilities)
        
        return (attractors, n_attractors, basin_sizes, 
                attractor_dict, state_space,
                coherence,fragility,
                basin_coherences, basin_fragilities,
                attractor_coherences, attractor_fragilities)


    def get_attractors_and_robustness_measures_synchronous(self, number_different_IC=500, RETURN_ATTRACTOR_COHERENCE = False):
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
        def lcm(a, b):
            return abs(a*b) // math.gcd(a, b)
        
        dictF = dict()
        attractors = []
        ICs_per_attractor_state = []
        basin_sizes = []
        attractor_dict = dict()
        attractor_state_dict = []
        distance_from_attractor_state_dict = []
        counter_phase_shifts = []
        
        height = []
        degrees = list(map(len, self.I))
        
        powers_of_2s = [np.array([2**i for i in range(NN)])[::-1] for NN in range(max(degrees)+1)]
        if self.N<64:
            powers_of_2 = np.array([2**i for i in range(self.N)])[::-1]
        
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
                    x = np.random.randint(2, size=self.N)
                    if self.N<64:
                        xdec = np.dot(x, powers_of_2)
                    else: #out of range of np.int64
                        xdec = ''.join(str(bit) for bit in x)
                    x_old = x.copy()
                else:
                    x = x_old
                    random_flipped_bit = np.random.choice(self.N)
                    x[random_flipped_bit] = 1 - x[random_flipped_bit]
                    if self.N<64:
                        xdec = np.dot(x, powers_of_2)
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
                            for jj in range(self.N):
                                if degrees[jj]>0:
                                    fx.append(self.F[jj].f[np.dot(x[self.I[jj]], powers_of_2s[degrees[jj]])])
                                else:#constant functions whose regulators were all fixed to a specific value
                                    fx.append(self.F[jj].f[0])
                            if self.N<64:
                                fxdec = np.dot(fx, powers_of_2)
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
                                    if self.N<64:
                                        fixed_point = np.array(utils.dec2bin(queue[index], self.N))
                                    else:
                                        fixed_point = np.array(list(queue[index]), dtype=int)
                                    states_attractors.append(fixed_point.reshape((1, self.N)))
                                    mean_states_attractors.append(fixed_point)
                                else:
                                    if self.N<64:
                                        limit_cycle = np.array([utils.dec2bin(state, self.N) for state in queue[index:]])
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
        approximate_fragility = fragility_sum * 1.0 / number_different_IC / self.N
        
        approximate_basin_coherence = np.array([basin_robustness[index_att] * 2.0 / basin_sizes[index_att] for index_att in range(len(attractors))])
        approximate_basin_fragility = np.array([basin_fragility[index_att] * 2.0 / basin_sizes[index_att] / self.N for index_att in range(len(attractors))])
        
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
                    for i in range(self.N):
                        if self.N<64:
                            x = np.array(utils.dec2bin(attractor_state, self.N))
                        else:
                            x = np.array(list(attractor_state), dtype=int)
                        x[i] = 1 - x[i]
                        if self.N<64:
                            xdec = np.dot(x, powers_of_2)
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
                                    for jj in range(self.N):
                                        if degrees[jj]>0:
                                            fx.append(self.F[jj].f[np.dot(x[self.I[jj]], powers_of_2s[degrees[jj]])])
                                        else:#constant functions whose regulators were all fixed to a specific value
                                            fx.append(self.F[jj].f[0])
                                    if self.N<64:
                                        fxdec = np.dot(fx, powers_of_2)
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
                                            if self.N<64:
                                                fixed_point = np.array(utils.dec2bin(queue[index], self.N))
                                            else:
                                                fixed_point = np.array(list(queue[index]), dtype=int)
                                            states_attractors.append(fixed_point.reshape((1, self.N)))
                                            mean_states_attractors.append(fixed_point)
                                        else:
                                            if self.N<64:
                                                limit_cycle = np.array([utils.dec2bin(state, self.N) for state in queue[index:]])
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
            attractor_coherence = np.array([s/self.N/size_attr for s,size_attr in zip(attractor_coherence,map(len,attractors_original))])
            attractor_fragility = np.array([s/self.N**2/size_attr for s,size_attr in zip(attractor_fragility,map(len,attractors_original))]) #something is wrong with attractor fragility, it returns values > 1 for small basins
            results[0] = attractors_original
            return tuple(results + [attractor_coherence,attractor_fragility])
        
    def get_strongly_connected_components(self):
        """
        Determine the strongly connected components of a wiring diagram.

        The wiring diagram is provided as a list of lists I, where I[i] contains the indices of regulators for node i.
        The function constructs a directed graph from these edges and returns its strongly connected components.

        Returns:
            list: A list of sets, each representing a strongly connected component.
        """
        edges_wiring_diagram = []
        for target, regulators in enumerate(self.I):
            for regulator in regulators:
                edges_wiring_diagram.append((regulator, target))
        subG = nx.from_edgelist(edges_wiring_diagram, create_using=nx.MultiDiGraph())
        return [scc for scc in nx.strongly_connected_components(subG)]

    def get_modular_structure(self):
        sccs = self.get_strongly_connected_components()
        scc_dict = {}
        for j,s in enumerate(sccs):
            for el in s:
                scc_dict.update({el:j})
        dag = set()
        for target,regulators in enumerate(self.I):
            for regulator in regulators:
                edge = (scc_dict[regulator],scc_dict[target])
                if edge[0]!=edge[1]:
                    dag.add(edge)   
        return dag

    def adjacency_matrix(self, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
        """
        Construct the (binary) adjacency matrix from the wiring diagram.

        Given the wiring diagram I (a list of regulator lists for each node) and a list of constants,
        this function builds an adjacency matrix where each entry m[j, i] is 1 if node j regulates node i.
        Self-loops can be optionally ignored, and constant nodes can be excluded.

        Parameters:
            constants (list, optional): List of constant nodes.
            IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
            IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded from the matrix.

        Returns:
            np.array: The binary adjacency matrix.
        """
        n = len(self.I)
        n_constants = len(constants)
        if IGNORE_CONSTANTS:
            m = np.zeros((n - n_constants, n - n_constants), dtype=int)
            for i in range(len(self.I)):
                for j in self.I[i]:
                    if j < n - n_constants and (not IGNORE_SELFLOOPS or i != j):
                        m[j, i] = 1
            return m
        else:
            return self.adjacency_matrix([], IGNORE_CONSTANTS=True)


    def get_signed_adjacency_matrix(self, type_of_each_regulation, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
        """
        Construct the signed adjacency matrix of a Boolean network.

        The signed adjacency matrix assigns +1 for increasing (activating) regulations,
        -1 for decreasing (inhibiting) regulations, and NaN for any other type.

        Parameters:
            type_of_each_regulation (list): List of lists corresponding to the type of regulation ('increasing' or 'decreasing')
                                            for each edge in I.
            constants (list, optional): List of constant nodes.
            IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
            IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded.

        Returns:
            np.array: The signed adjacency matrix.
        """
        n = len(self.I)
        n_constants = len(constants)
        if IGNORE_CONSTANTS:
            m = np.zeros((n - n_constants, n - n_constants), dtype=int)
            for i, (regulators, type_of_regulation) in enumerate(zip(self.I, type_of_each_regulation)):
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
            return self.get_signed_adjacency_matrix(type_of_each_regulation, [], IGNORE_CONSTANTS=True)


    def get_signed_effective_graph(self, type_of_each_regulation, constants=[], IGNORE_SELFLOOPS=False, IGNORE_CONSTANTS=True):
        """
        Construct the signed effective graph of a Boolean network.

        This function computes an effective graph in which each edge is weighted by its effectiveness.
        Effectiveness is obtained via get_edge_effectiveness on the corresponding Boolean function.
        Edges are signed according to the type of regulation ('increasing' or 'decreasing').

        Parameters:
            type_of_each_regulation (list): List of lists specifying the type of regulation for each edge.
            constants (list, optional): List of constant nodes.
            IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
            IGNORE_CONSTANTS (bool, optional): If True, constant nodes are excluded.

        Returns:
            np.array: The signed effective graph as a matrix of edge effectiveness values.
        """
        n = len(self.I)
        n_constants = len(constants)
        if IGNORE_CONSTANTS:
            m = np.zeros((n - n_constants, n - n_constants), dtype=float)
            for i, (regulators, type_of_regulation) in enumerate(zip(self.I, type_of_each_regulation)):
                effectivenesses = self.F[i].get_edge_effectiveness()
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
            return self.get_signed_effective_graph(type_of_each_regulation, [], IGNORE_CONSTANTS=True)


    def get_ffls(self):
        """
        Identify feed-forward loops (FFLs) in a Boolean network and optionally determine their types.

        A feed-forward loop (FFL) is a three-node motif where node i regulates node k both directly and indirectly via node j.
        
        Returns:
            tuple: A tuple (ffls, types), where ffls is a list of FFLs and types is a list of corresponding monotonicity types.
        """
        ffls = []
        types = []
        for i in range(len(self.I)):
            for j in range(i + 1, len(self.I)):
                for k in range(len(self.I)):
                    if i == k or j == k:
                        continue
                    # Check if there is an FFL: i regulates k and j regulates both i and k.
                    if i in self.I[k] and i in self.I[j] and j in self.I[k]:
                        ffls.append([i, j, k])
                        # Compute types if F is provided.
                        # (This example assumes a helper function is_monotonic exists and that I is ordered.)
                        #monotonic_i = is_monotonic(F[i], True)[1]
                        monotonic_j = self.F[j].is_monotonic(True)[1]
                        monotonic_k = self.F[k].is_monotonic(True)[1]
                        direct = monotonic_k[self.I[k].index(i)]
                        indirect1 = monotonic_j[self.I[j].index(i)]
                        indirect2 = monotonic_k[self.I[k].index(j)]
                        types.append([direct, indirect1, indirect2])
        return (ffls, types)


    def get_ffls_from_I(self, types_I=None):
        """
        Identify feed-forward loops (FFLs) in a Boolean network based solely on the wiring diagram.

        The function uses the inverted wiring diagram to identify common targets and returns the FFLs found.
        If types_I (the type of each regulation) is provided, it also returns the corresponding regulation types.

        Parameters:
            types_I (list, optional): List of lists specifying the type (e.g., 'increasing' or 'decreasing') for each regulation.

        Returns:
            If types_I is provided:
                tuple: (ffls, types) where ffls is a list of identified FFLs (each as a list [i, j, k]),
                       and types is a list of corresponding regulation type triplets.
            Otherwise:
                list: A list of identified FFLs.
        """
        all_tfs = list(range(len(self.I)))
        n_tfs = len(all_tfs)
        all_tfs_dict = dict(zip(all_tfs, list(range(n_tfs))))
        I_inv = [[] for _ in all_tfs]
        for target, el in enumerate(self.I):
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
                        direct = types_I[k][self.I[k].index(all_tfs[i])]
                        indirect1 = types_I[all_tfs[j]][self.I[all_tfs[j]].index(all_tfs[i])]
                        indirect2 = types_I[k][self.I[k].index(all_tfs[j])]
                        types.append([direct, indirect1, indirect2])
        if types_I is not None:
            return (ffls, types)
        else:
            return ffls

    def generate_networkx_graph(self, constants, variables):
        """
        Generate a NetworkX directed graph from a wiring diagram.

        Nodes are labeled with variable names (from variables) and constant names (from constants). Edges are added
        from each regulator to its target based on the wiring diagram I.

        Parameters:
            constants (list): List of constant names.
            variables (list): List of variable names.

        Returns:
            networkx.DiGraph: The noderated directed graph.
        """
        names = list(variables) + list(constants)
        G = nx.DiGraph()
        G.add_nodes_from(names)
        G.add_edges_from([(names[self.I[i][j]], names[i]) for i in range(len(variables)) for j in range(len(self.I[i]))])
        return G


    def generate_networkx_graph_from_edges(self, n_variables):
        """
        Generate a NetworkX directed graph from an edge list derived from the wiring diagram.

        Only edges among the first n_variables (excluding constant self-loops) are included.

        Parameters:
            n_variables (int): Number of variable nodes (constants are excluded).

        Returns:
            networkx.DiGraph: The generated directed graph.
        """
        edges = []
        for j, regulators in enumerate(self.I):
            if j >= n_variables:  # Exclude constant self-loops
                break
            for i in regulators:
                edges.append((i, j))
        return nx.DiGraph(edges)

    def get_type_of_loop(self, loop):
        """
        Determine the regulation types along a feedback loop.

        For a given loop (a list of node indices), this function returns a list containing
        the type (e.g., 'increasing' or 'decreasing') of each regulation along the loop.
        The loop is assumed to be ordered such that the first node is repeated at the end.

        Parameters:
            loop (list): List of node indices representing the loop.

        Returns:
            list: A list of regulation types corresponding to each edge in the loop.
        """
        n = len(loop)
        dummy = loop[:]
        dummy.append(loop[0])
        res = []
        for i in range(n):
            # Assumes is_monotonic returns a tuple with the monotonicity information.
            res.append(self.F[dummy[i+1]].is_monotonic(True)[1][list(self.I[dummy[i+1]]).index(dummy[i])])
        return res