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

import utils
from boolean_function import BooleanFunction as BF
from boolean_network import BooleanNetwork as BN

def random_function(n, probability_one=0.5):
    """
    Generate a random Boolean function in n variables.

    The Boolean function is represented as a truth table (an array of length 2^n) in which each entry is 0 or 1.
    Each entry is set to 1 with probability `probability_one`.

    Parameters:
        n (int): Number of variables.
        probability_one (float, optional): Probability that a given entry is 1 (default is 0.5).

    Returns:
        BooleanFunction: Boolean function object.
    """
    return BF(np.array(np.random.random(2**n) < probability_one, dtype=int))


def random_linear_function(n):
    """
    Generate a random linear Boolean function in n variables.

    A random linear Boolean function is constructed by randomly choosing whether to include each variable or its negation in a linear sum.
    The resulting expression is then reduced modulo 2.

    Parameters:
        n (int): Number of variables.

    Returns:
        BooleanFunction: Boolean function object.
    """
    expr = '(%s) %% 2 == 1' % (' + '.join(['x%i' % i if random.random() > 0.5 else '(1 + x%i)' % i for i in range(n)]))
    return BF(utils.f_from_expression(expr)[0])


def random_non_degenerated_function(n, probability_one=0.5):
    """
    Generate a random non-degenerated Boolean function in n variables.

    A non-degenerated Boolean function is one in which every variable is essential (i.e. the output depends on every input).
    The function is repeatedly generated with the specified bias until a non-degenerated function is found.

    Parameters:
        n (int): Number of variables.
        probability_one (float, optional): Bias of the Boolean function (probability of a 1; default is 0.5).

    Returns:
        BooleanFunction: Boolean function object.
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    while True:  # works well because most Boolean functions are non-degenerated
        f = random_function(n, probability_one)
        if not f.is_degenerated():
            return f


def random_degenerated_function(n, probability_one=0.5):
    """
    Generate a random degenerated Boolean function in n variables.

    A degenerated Boolean function is one in which at least one variable is non‐essential (its value never affects the output).
    The function is generated repeatedly until a degenerated function is found.

    Parameters:
        n (int): Number of variables.
        probability_one (float, optional): Bias of the Boolean function (default is 0.5, i.e., unbiased).

    Returns:
        BooleanFunction: Boolean function object that is degenerated in the first input (and possibly others).
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    
    f_original = random_function(n-1, probability_one)
    index_non_essential_variable = int(random.random()*n)
    f = np.zeros(2**n, dtype=int)
    indices = (np.arange(2**n)//(2**index_non_essential_variable))%2==1
    f[indices] = f_original.f
    f[~indices] = f_original.f
    return BF(f)


def random_non_canalizing_function(n, probability_one=0.5):
    """
    Generate a random non-canalizing Boolean function in n (>1) variables.

    A Boolean function is canalizing if there exists at least one variable whose fixed value forces the output.
    This function returns one that is not canalizing.

    Parameters:
        n (int): Number of variables (n > 1).
        probability_one (float, optional): Bias of the Boolean function (default is 0.5, i.e., unbiased).

    Returns:
        BooleanFunction: Boolean function object.
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    assert type(n)==int and n > 1, "n must be an integer > 1"
    while True:  # works because most functions are non-canalizing
        f = BF(np.array(np.random.random(2**n) < probability_one, dtype=int))
        if not f.is_canalizing():
            return f


def random_non_canalizing_non_degenerated_function(n, probability_one=0.5):
    """
    Generate a random Boolean function in n (>1) variables that is both non-canalizing and non-degenerated.

    Such a function has every variable essential and is not canalizing.

    Parameters:
        n (int): Number of variables (n > 1).
        probability_one (float, optional): Bias of the Boolean function (default is 0.5, i.e., unbiased).

    Returns:
        BooleanFunction: Boolean function object.
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    assert type(n)==int and n > 1, "n must be an integer > 1"
    while True:  # works because most functions are non-canalizing and non-degenerated
        f = BF(np.array(np.random.random(2**n) < probability_one, dtype=int))
        if not f.is_canalizing() and not f.is_degenerated():
            return f


def random_k_canalizing(n, k, EXACT_DEPTH=False, left_side_of_truth_table=None):
    """
    Generate a random k-canalizing Boolean function in n variables.

    A Boolean function is k-canalizing if it has at least k conditionally canalizing variables.
    If EXACT_DEPTH is True, the function will have exactly k canalizing variables; otherwise, its canalizing depth may exceed k.

    Parameters:
        n (int): Total number of variables.
        k (int): Number of canalizing variables. Set k==n to generate a random nested canalizing function.
        EXACT_DEPTH (bool, optional): If True, enforce that the canalizing depth is exactly k (default is False).
        left_side_of_truth_table (optional): Precomputed left-hand side of the truth table for speed-up. Default is None.

    Returns:
        BooleanFunction: Boolean function object.
    
    References:
        [1] He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth. 
            Physica D: Nonlinear Phenomena, 314, 1-8.
        [2] Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions: 
            Algorithms and applications. Automatica, 146, 110630.
    """
    assert n - k != 1 or EXACT_DEPTH == False,'There are no functions of exact canalizing depth n-1.\nEither set EXACT_DEPTH=False or ensure k != n-1'
    assert 0 <= k and k <= n,'Error:\nEnsure 0 <= k <= n.'
        
    if left_side_of_truth_table is None:  # to reduce run time, this should be computed once and then passed as argument
        left_side_of_truth_table = list(itertools.product([0, 1], repeat=n))
    num_values = 2**n
    aas = np.random.randint(2, size=k)  # canalizing inputs
    bbs = np.random.randint(2, size=k)  # canalized outputs

    # The activator_or_inhibitor parameter is currently not used.
    can_vars = np.random.choice(n, k, replace=False)
    f = np.zeros(num_values, dtype=int)
    if k < n:
        if EXACT_DEPTH:
            core_polynomial = random_non_canalizing_non_degenerated_function(n - k).f
        else:
            core_polynomial = random_non_degenerated_function(n - k).f
    else:
        core_polynomial = [1 - bbs[-1]]
    counter_non_canalized_positions = 0
    for i in range(num_values):
        for j in range(k):
            if left_side_of_truth_table[i][can_vars[j]] == aas[j]:
                f[i] = bbs[j]
                break
        else:
            f[i] = core_polynomial[counter_non_canalized_positions]
            counter_non_canalized_positions += 1
    return BF(f)


def random_k_canalizing_with_specific_layer_structure(n, layer_structure, EXACT_DEPTH=False, left_side_of_truth_table=None):
    """
    Generate a random Boolean function in n variables with a specified canalizing layer structure.

    The layer structure is given as a list [k_1, ..., k_r], where each k_i indicates the number of canalizing variables 
    in that layer. If the function is fully canalizing (i.e. sum(layer_structure) == n and n > 1), the last layer must have at least 2 variables.

    Parameters:
        n (int): Total number of variables.
        layer_structure (list): List [k_1, ..., k_r] describing the canalizing layer structure.
                               Each k_i ≥ 1, and if sum(layer_structure) == n and n > 1, then layer_structure[-1] ≥ 2.
                               Set sum(layer_structure)==n to generate a random nested canalizing function.
        EXACT_DEPTH (bool, optional): If True, the canalizing depth is exactly sum(layer_structure) (default is False).
        left_side_of_truth_table (optional): Precomputed left-hand side of the truth table for speed-up. Default is None.

    Returns:
        BooleanFunction: Boolean function object.
    
    References:
        [1] He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        [2] Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
            of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    k = sum(layer_structure)  # canalizing depth
    if k == 0:
        layer_structure = [0]
        
    assert n - k != 1 or EXACT_DEPTH == False,'Error:\nThere are no functions of exact canalizing depth n-1.\nEither set EXACT_DEPTH=False or ensure k=sum(layer_structure)!=n-1.'
    assert 0 <= k and k <= n,'Error:\nEnsure 0 <= k = sum(layer_structure) <= n.'
    assert k < n or layer_structure[-1] > 1 or n == 1,'Error:\nThe last layer of an NCF (i.e., an n-canalizing function) has to have size >= 2 whenever n > 1.\nIf k=sum(layer_structure)=n, ensure that layer_structure[-1]>=2.'
    assert min(layer_structure) >= 1,'Error:\nEach layer must have at least one variable (each element of layer_structure must be >= 1).'
    
    if left_side_of_truth_table is None:  # to decrease run time, this should be computed once and then passed as argument
        left_side_of_truth_table = list(itertools.product([0, 1], repeat=n))
    num_values = 2**n
    aas = np.random.randint(2, size=k)  # canalizing inputs
    b0 = np.random.randint(2)
    bbs = [b0] * layer_structure[0]  # canalized outputs for first layer
    for i in range(1, len(layer_structure)):
        if i % 2 == 0:
            bbs.extend([b0] * layer_structure[i])
        else:
            bbs.extend([1 - b0] * layer_structure[i])
    can_vars = np.random.choice(n, k, replace=False)
    f = np.zeros(num_values, dtype=int)
    if k < n:
        if EXACT_DEPTH:
            core_polynomial = random_non_canalizing_non_degenerated_function(n - k).f
        else:
            core_polynomial = random_non_degenerated_function(n - k).f
    else:
        core_polynomial = [1 - bbs[-1]]
    counter_non_canalized_positions = 0
    for i in range(num_values):
        for j in range(k):
            if left_side_of_truth_table[i][can_vars[j]] == aas[j]:
                f[i] = bbs[j]
                break
        else:
            f[i] = core_polynomial[counter_non_canalized_positions]
            counter_non_canalized_positions += 1
    return BF(f)


def random_NCF(n,layer_structure=None,left_side_of_truth_table=None):
    '''
    Generate a random nested canalizing Boolean function in n variables 
    with a specified canalizing layer structure (if provided).

    The layer structure is given as a list [k_1, ..., k_r], where each k_i indicates the number of canalizing variables 
    in that layer. If the function is fully canalizing (i.e. sum(layer_structure) == n and n > 1), the last layer must have at least 2 variables.

    Parameters:
        n (int): Total number of variables.
        layer_structure (list,optional): List [k_1, ..., k_r] describing the canalizing layer structure.
                               Each k_i ≥ 1, and if sum(layer_structure) == n and n > 1, then layer_structure[-1] ≥ 2.
                               Set sum(layer_structure)==n to generate a random nested canalizing function.
        left_side_of_truth_table (optional): Precomputed left-hand side of the truth table for speed-up. Default is None.

    Returns:
        BooleanFunction: Boolean function object.
    
    References:
        [1] He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        [2] Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
            of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    '''    
    if layer_structure is None:
        return random_k_canalizing(n,n,EXACT_DEPTH=False,left_side_of_truth_table=left_side_of_truth_table)
    else:
        assert sum(layer_structure) == n,'Error:\nEnsure sum(layer_structure) == n.'
        assert layer_structure[-1] > 1 or n == 1,'Error:\nThe last layer of an NCF has to have size >= 2 whenever n > 1.\nEnsure that layer_structure[-1]>=2.'
        return random_k_canalizing_with_specific_layer_structure(n,layer_structure,EXACT_DEPTH=False, left_side_of_truth_table=left_side_of_truth_table)


def get_layer_structure_of_an_NCF_given_its_Hamming_weight(n, w):
    """
    Compute the canalizing layer structure of a nested canalizing function (NCF) given its Hamming weight.

    There exists a bijection between the Hamming weight (with w equivalent to 2^n - w) and the canalizing layer structure of an NCF.
    The layer structure is represented as [k_1, ..., k_r], where each k_i ≥ 1 and, if n > 1, for the last layer k_r ≥ 2.

    Parameters:
        n (int): Number of inputs (variables) of the NCF.
        w (int): Odd Hamming weight of the NCF, i.e., the number of 1s in the 2^n-vector representation of the function.

    Returns:
        tuple: A tuple (r, layer_structure_NCF), where:
            - r (int): The number of canalizing layers.
            - layer_structure_NCF (list): A list [k_1, ..., k_r] describing the number of variables in each layer.

    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness of Boolean networks.
        Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    if w == 1:
        r = 1
        layer_structure_NCF = [n]
    else:
        assert type(w) == int or type(w) == np.int64, 'Hamming weight must be an integer'
        assert 1 <= w <= 2**n - 1, 'Hamming weight w must satisfy 1 <= w <= 2^n - 1'
        assert w % 2 == 1, 'Hamming weight must be an odd integer since all NCFs have an odd Hamming weight.'
        w_bin = utils.dec2bin(w, n)
        current_el = w_bin[0]
        layer_structure_NCF = [1]
        for el in w_bin[1:-1]:
            if el == current_el:
                layer_structure_NCF[-1] += 1
            else:
                layer_structure_NCF.append(1)
                current_el = el
        layer_structure_NCF[-1] += 1
        r = len(layer_structure_NCF)
    return (r, layer_structure_NCF)


def random_adjacency_matrix(N, indegrees, NO_SELF_REGULATION=True, STRONGLY_CONNECTED=False):
    """
    Generate a random adjacency matrix for a network of N nodes.

    Each node i is assigned indegrees[i] outgoing edges (regulators) chosen at random.
    Optionally, self-regulation (an edge from a node to itself) can be disallowed,
    and the generated network can be forced to be strongly connected.

    Parameters:
        N (int): Number of nodes.
        indegrees (list or array-like): List of length N specifying the number of outgoing edges for each node.
        NO_SELF_REGULATION (bool, optional): If True, self-regulation is disallowed (default is True).
        STRONGLY_CONNECTED (bool, optional): If True, the generated network is forced to be strongly connected (default is False).

    Returns:
        tuple: (matrix, indices) where:
            - matrix (np.array): An N x N adjacency matrix with entries 0 or 1.
            - indices (list): A list of length N, where each element is an array of selected target indices for the corresponding node.
    """
    matrix = np.zeros((N, N), dtype=int)
    indices = []
    for i in range(N):
        if NO_SELF_REGULATION:
            indexes = np.random.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
        else:
            indexes = np.random.choice(np.arange(N), indegrees[i], replace=False)
        indexes = np.sort(indexes)
        indices.append(indexes)
        for index in indexes:
            matrix[i][index] = 1
    if STRONGLY_CONNECTED:
        G = nx.from_numpy_array(matrix, parallel_edges=False, create_using=nx.MultiDiGraph())
        if not nx.is_strongly_connected(G):
            return random_adjacency_matrix(N, indegrees, NO_SELF_REGULATION, STRONGLY_CONNECTED)
    return (matrix, indices)

    
def random_edge_list(N, indegrees, NO_SELF_REGULATION, AT_LEAST_ONE_REGULATOR_PER_NODE=False):
    """
    Generate a random edge list for a network of N nodes with optional constraints.

    Each node i receives indegrees[i] incoming edges chosen at random.
    Optionally, the function can ensure that every node regulates at least one other node.

    Parameters:
        N (int): Number of nodes.
        indegrees (list or array-like): List of length N specifying the number of regulators for each node.
        NO_SELF_REGULATION (bool): If True, disallow self-regulation.
        AT_LEAST_ONE_REGULATOR_PER_NODE (bool, optional): If True, ensure that each node has at least one outgoing edge (default is False).

    Returns:
        list: A list of tuples (source, target) representing the edges.
    """
    if AT_LEAST_ONE_REGULATOR_PER_NODE == False:
        edge_list = []
        for i in range(N):
            if NO_SELF_REGULATION:
                indices = np.random.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
            else:
                indices = np.random.choice(np.arange(N), indegrees[i], replace=False)
            edge_list.extend(list(zip(indices, i * np.ones(indegrees[i], dtype=int))))
    else:
        edge_list = []
        outdegree = np.zeros(N, dtype=int)
        sum_indegrees = sum(indegrees)  # total number of regulations
        for i in range(N):
            if NO_SELF_REGULATION:
                indices = np.random.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
            else:
                indices = np.random.choice(np.arange(N), indegrees[i], replace=False)
            outdegree[indices] += 1
            edge_list.extend(list(zip(indices, i * np.ones(indegrees[i], dtype=int))))
        while min(outdegree) == 0:
            index_sink = np.where(outdegree == 0)[0][0]
            index_edge = int(random.random() * sum_indegrees)
            if NO_SELF_REGULATION:
                while edge_list[index_edge][1] == index_sink:
                    index_edge = int(random.random() * sum_indegrees)
            outdegree[index_sink] += 1
            outdegree[edge_list[index_edge][0]] -= 1
            edge_list[index_edge] = (index_sink, edge_list[index_edge][1])
    return edge_list


def random_BN(N, n, k=0, STRONGLY_CONNECTED=False, indegree_distribution='constant',
              left_sides_of_truth_tables=None, layer_structure=None, EXACT_DEPTH=False, NO_SELF_REGULATION=True, LINEAR=False,
              edges_wiring_diagram=None, bias=0.5, n_attempts_to_generate_strongly_connected_network = 1000):
    """
    Generate a random Boolean network (BN).

    This function creates a Boolean network of N nodes by first generating a wiring diagram
    (a set of regulatory interactions) according to a specified in-degree distribution and then assigning
    Boolean functions to each node. The functions can be canalizing with prescribed depth and/or specific layer structure,
    lineear, or random functions with a specified bias.

    Parameters:
        N (int): Number of nodes in the network.
        n (int, list, or np.array; float allowed if indegree_distribution=='poisson'): 
            Determines the in-degree of each node. If an integer, each node has the same number of regulators;
            if a vector, each element gives the number of regulators for the corresponding node.
        k (int, list, or np.array, optional): Specifies the minimal canalizing depth for each node (exact canalizing depth if EXACT_DEPTH==True).
                                              If an integer, the same depth is used for all nodes; if a vector, each node gets its own depth.
                                              Default is 0.
        STRONGLY_CONNECTED (bool, optional): If True, ensures that the generated network is strongly connected. Default is True.
        indegree_distribution (str, optional): In-degree distribution to use. Options include 'constant' (or 'dirac'/'delta'),
                                               'uniform', or 'poisson'. Default is 'constant'.
        left_sides_of_truth_tables (list, optional): Precomputed truth tables (lists of tuples) for different in-degrees, used for speed-up. Default is None and it is computed every time.
        layer_structure (optional): Specifies the canalizing layer structure for the Boolean functions. If provided, the parameter k is ignored.
        EXACT_DEPTH (bool, optional): If True, Boolean functions are generated with exactly the specified canalizing depth;
                                      if False, the functions have at least that depth. Default is False.
        NO_SELF_REGULATION (bool, optional): If True, self-regulation (self-loops) is disallowed. Default is True.
        LINEAR (bool, optional): If True, Boolean functions are generated to be linear. Default is False.
        edges_wiring_diagram (optional): User-defined edge list for the wiring diagram. If provided, the parameters n and indegree_distribution are ignored.
        bias (float, optional): Bias of generated Boolean functions (probability of output 1). Default is 0.5. Ignored unless k==0 and LINEAR==False and layer_structure is None.
        n_attempts_to_generate_strongly_connected_network (integer, optional): Number of attempts to generate a strongly connected wiring diagram before raising an error and quitting.
    
    Returns:
        BooleanNetwork: Boolean network object.
    """
    # Generate the in-degree based on the specified in-degree distribution and the in-degree parameter n. If n is a vector of length N, this is used as in-degree.
    if type(n) in [list, np.array]:
        assert (np.all([type(el) in [int, np.int_] for el in n]) and len(n) == N and min(n) >= 1 and max(n) <= N), 'A vector n was submitted.\nEnsure that n is an N-dimensional vector where each element is an integer between 1 and N representing the upper bound of a uniform degree distribution (lower bound == 1).'
        indegrees = np.array(n[:])
    elif indegree_distribution.lower() in ['constant', 'dirac', 'delta']:
        assert (type(n) in [int, np.int_] and n >= 1 and n <= N), 'n must be a single integer (or N-dimensional vector of integers) between 1 and N when using a constant degree distribution.'
        indegrees = np.ones(N, dtype=int) * n
    elif indegree_distribution.lower() == 'uniform':
        assert (type(n) in [int, np.int_] and n >= 1 and n <= N - int(NO_SELF_REGULATION)), 'n must be a single integer (or N-dimensional vector of integers) between 1 and ' + ('N-1' if NO_SELF_REGULATION else 'N')+' representing the upper bound of a uniform degree distribution (lower bound == 1).'
        indegrees = 1 + np.random.randint(n - 1, size=N)
    elif indegree_distribution.lower() == 'poisson':
        assert (type(n) in [int, np.int_, float, np.float_] and n>= 1 and n<=N), 'n must be a single number (or N-dimensional vector) > 0 representing the Poisson parameter.'
        indegrees = np.maximum(np.minimum(np.random.poisson(lam=n, size=N),N - int(NO_SELF_REGULATION)), 1)
    else:
        raise AssertionError('None of the predefined in-degree distributions were chosen.\nTo use a user-defined in-degree vector, use the input n to submit an N-dimensional vector where each element of n must an integer between 1 and N.')

    # Process the canalizing depth / canalizing layer structure
    if layer_structure is None:
        if type(k) in [int, np.int_]:
            assert k >= 0 and k<=N,'The canalizing depth k must be an integer between 0 and N.'
            max_k = k
        elif type(k) in [list, np.array]:
            max_k = max(k)
            assert (len(k) == N and np.all([type(el) in [int, np.int_] for el in k]) and min(k) >= 0 and max_k <= N),'A vector k was submitted.\nTo use a user-defined vector k, ensure that k is an N-dimensional vector where each element is an integer between 0 and N.'
        else:
            raise AssertionError('Wrong input format for k.\nk must be a single integer (or N-dimensional vector of integers) between 0 and N, specifying the minimal canalizing depth or exact canalizing depth (if EXACT_DEPTH==True).')            
    else:  # layer_structure provided
        if np.all([type(el) in [int, np.int_] for el in layer_structure]):
            max_k = sum(layer_structure)
            assert np.all([type(el) in [int, np.int_] for el in layer_structure]) and np.all([el >= 1 for el in layer_structure]) and max_k <= N, 'The layer structure must be a vector of positive integers with 0 <= k = sum(layer_structure) <= N.'
        elif np.all([type(el) in [list, np.array] for el in layer_structure]):
            max_k = max([sum(el) for el in layer_structure])
            assert len(layer_structure) == N and type(layer_structure[0][0]) in [int, np.int_] and min([min(el) for el in layer_structure]) >= 0 and max_k <= N, 'Ensure that layer_structure is an N-dimensional vector where each element represents a layer structure and is a vector of positive integers with 1 <= k = sum(layer_structure[i]) <= N.'
        else:
            raise AssertionError('Wrong input format for layer_structure.\nlayer_structure must be a single vector (or N-dimensional vector of layer structures) where the sum of each element is between 0 and N.')

    if edges_wiring_diagram is None:
        counter = 0
        while True:  # Keep generating until we have a strongly connected graph
            edges_wiring_diagram = random_edge_list(N, indegrees, NO_SELF_REGULATION)
            if STRONGLY_CONNECTED:#may take a long time if n is small and N is large
                G = nx.from_edgelist(edges_wiring_diagram, create_using=nx.MultiDiGraph())
                if not nx.is_strongly_connected(G):
                    counter+=1
                    if counter>n_attempts_to_generate_strongly_connected_network:
                        raise RuntimeError('Made '+str(n_attempts_to_generate_strongly_connected_network)+' unsuccessful attempts to generate a strongly connected wiring diagram of '+str(N)+' nodes and degrees '+str(indegrees)+'.\nYou may increase the number of attempts by modulating the parameter n_attempts_to_generate_strongly_connected_network.')
                    continue
            break
    else:
        assert len(set(np.array(edges_wiring_diagram).flatten())) == N, "Number of nodes provided in edges_wiring_diagram != N"
        indegrees = np.zeros(N, dtype=int)
        for target in np.array(edges_wiring_diagram)[:, 1]:
            indegrees[target] += 1


    max_n = max(indegrees)
    if max_k > 0 and (left_sides_of_truth_tables is None or len(left_sides_of_truth_tables) < max_n):
        left_sides_of_truth_tables = [[[0], [1]]]
        left_sides_of_truth_tables.extend([list(itertools.product([0, 1], repeat=nn)) for nn in range(2, max_n + 1)])

    F = []
    for i in range(N):
        if LINEAR:
            F.append(random_linear_function(indegrees[i]))
        if k > 0 and layer_structure is None:
            if type(k) in [int, np.int_]:
                F.append(random_k_canalizing(indegrees[i], min(k, indegrees[i]), EXACT_DEPTH=EXACT_DEPTH, left_side_of_truth_table=left_sides_of_truth_tables[indegrees[i]-1]))
            else:
                F.append(random_k_canalizing(indegrees[i], min(k[i], indegrees[i]), EXACT_DEPTH=EXACT_DEPTH, left_side_of_truth_table=left_sides_of_truth_tables[indegrees[i]-1]))
        elif layer_structure is not None:
            if np.all([type(el) in [int, np.int_] for el in layer_structure]):
                F.append(random_k_canalizing_with_specific_layer_structure(indegrees[i], layer_structure, EXACT_DEPTH=EXACT_DEPTH, left_side_of_truth_table=left_sides_of_truth_tables[indegrees[i]-1]))
            else:
                F.append(random_k_canalizing_with_specific_layer_structure(indegrees[i], layer_structure[i], EXACT_DEPTH=EXACT_DEPTH, left_side_of_truth_table=left_sides_of_truth_tables[indegrees[i]-1]))
        else:
            if EXACT_DEPTH is True:
                F.append(random_non_canalizing_non_degenerated_function(indegrees[i], bias))
            else:
                F.append(random_non_degenerated_function(indegrees[i], bias))

    I = [[] for _ in range(N)]
    for edge in edges_wiring_diagram:
        I[edge[1]].append(edge[0])
    for i in range(N):
        I[i] = np.sort(I[i])
    return BN(F, I)

