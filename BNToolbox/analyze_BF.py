#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 09:25:40 2025
Last Edited on Wed Aug 13 2025

@author: Claus Kadelka, Benjamin Coberly
"""

##Imports
import numpy as np
import utils
from scipy.special import binom

from boolean_function import __LOADED_CANA__

## constant functions
def is_constant(bf):
    """
    Check whether a Boolean function is constant.

    Parameters:
        bf (BooleanFunction): Boolean function object.

    Returns:
        bool: True if bf is constant (all outputs are 0 or all are 1), False otherwise.
    """
    return bf.is_constant()


## degenerated functions / essential variables
def is_degenerated(bf):
    """
    Determine if a Boolean function contains non-essential variables.

    A variable is non-essential if the function's output does not depend on it.

    Parameters:
        bf (BooleanFunction): Boolean function object.

    Returns:
        bool: True if f contains at least one non-essential variable, False if all variables are essential.
    """
    return bf.is_degenerated()

def get_essential_variables(bf):
    """
    Determine the indices of essential variables in a Boolean function.

    A variable is essential if changing its value (while holding the others constant) can change the output of f.

    Parameters:
        bf (BooleanFunction): Boolean function object.

    Returns:
        list: List of indices corresponding to the essential variables.
    """
    return bf.get_essential_variables()


def get_number_of_essential_variables(bf):
    """
    Count the number of essential variables in a Boolean function.

    Parameters:
        bf (BooleanFunction): Boolean function object.

    Returns:
        int: The number of essential variables.
    """
    return bf.get_number_of_essential_variables()


## monotonicity, type of inputs
def is_monotonic(bf, GET_DETAILS=False):
    """
    Determine if a Boolean function is monotonic.

    A Boolean function is monotonic if it is monotonic in each variable. 
    That is, if for all i=1,...,n:
        f(x_1,...,x_i=0,...,x_n) >= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n) or 
        f(x_1,...,x_i=0,...,x_n) <= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n)

    Parameters:
        bf (BooleanFunction): Boolean function object.
        GET_DETAILS (bool, optional): If True, the type of each variable (increasing, decreasing, not monotonic, not essential) is returned. 
    Returns:
        bool: True if f contains at least one non-essential variable, False if all variables are essential.
        list: List containing the type of regulation of each variable. Only returned if GET_DETAILS==True.
    """
    return bf.is_monotonic(GET_DETAILS)


## symmetry
def get_symmetry_groups(bf, left_side_of_truth_table=None):
    """
    Determine all symmetry groups of input variables for a Boolean function.

    Two variables are in the same symmetry group if swapping their values does not change the output
    of the function for any input of the other variables.

    Parameters:
        bf (BooleanFunction): Boolean function object.
        left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
            If not provided or if its shape does not match, it will be computed.

    Returns:
        list: A list of lists where each inner list contains indices of variables that form a symmetry group.
    """
    return bf.get_symmetry_groups(left_side_of_truth_table)


## bias
def get_absolute_bias(bf):
    """
    Compute the absolute bias of a Boolean function.

    The absolute bias is defined as |(sum(f) / 2^(n-1)) - 1|, which quantifies how far the function's output distribution
    deviates from being balanced.

    Parameters:
        bf (BooleanFunction): Boolean function object.

    Returns:
        float: The absolute bias of the Boolean function.
    """
    return bf.get_absolute_bias()


## average sensitivity
def get_average_sensitivity(bf, nsim=10000, EXACT=False, NORMALIZED=True):
    """
    Compute the average sensitivity of a Boolean function.

    The average sensitivity is equivalent to the Derrida value D(F,1) when the update rule is sampled
    from the same space. This function can compute the exact sensitivity by exhaustively iterating over all inputs (if EXACT is True)
    or estimate it via Monte Carlo sampling (if EXACT is False). The result can be normalized by the number of inputs.

    Parameters:
        bf (BooleanFunction): Boolean function object.
        nsim (int, optional): Number of random samples (default is 10000, used when EXACT is False).
        EXACT (bool, optional): If True, compute the exact sensitivity by iterating over all inputs; otherwise, use sampling (default).
        NORMALIZED (bool, optional): If True, return the normalized sensitivity (divided by the number of function inputs); otherwise, return the total count.

    Returns:
        float: The (normalized) average sensitivity of the Boolean function.
    """
    return bf.get_average_sensitivity(nsim, EXACT, NORMALIZED)


## canalization, k-canalization, canalizing layer structure, canalizing depth
def is_canalizing(bf):
    """
    Determine if a Boolean function is canalizing.

    A Boolean function f(x_1, ..., x_n) is canalizing if there exists at least one variable x_i and a value a ∈ {0, 1} 
    such that f(x_1, ..., x_i = a, ..., x_n) is constant.

    Parameters:
        bf (BooleanFunction): Boolean function object.

    Returns:
        bool: True if f is canalizing, False otherwise.
    """
    return bf.is_canalizing()


def is_k_canalizing(bf, k):
    """
    Determine if a Boolean function is k-canalizing.

    A Boolean function is k-canalizing if it has at least k conditionally canalizing variables.
    This is checked recursively: after fixing a canalizing variable (with a fixed canalizing input that forces the output),
    the subfunction (core function) must itself be canalizing for the next variable, and so on.

    Parameters:
        bf (BooleanFunction): Boolean function object.
        k (int): The desired canalizing depth (0 ≤ k ≤ n). Note: every function is 0-canalizing.

    Returns:
        bool: True if f is k-canalizing, False otherwise.
    
    References:
        He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    return bf.is_k_canalizing(k)


def is_k_canalizing_return_canalizing_structure(bf, k):
    """
    Determine if a Boolean function is k-canalizing and return associated canalizing data.

    This function recursively checks whether f is k-canalizing and returns:
      - A boolean indicating success.
      - The canalizing input values.
      - The canalized output values.
      - The core function that remains after removing the canalizing variables.

    Parameters:
        bf (BooleanFunction): Boolean function object.
        k (int): The canalizing depth to check.
        can_inputs (np.array, optional): Accumulated canalizing input values (default is an empty array).
        can_outputs (np.array, optional): Accumulated canalized output values (default is an empty array).

    Returns:
        tuple: A tuple containing:
            - bool: True if f is k-canalizing, False otherwise.
            - dict: A dictionary containing:
                - CanalizingInputs (np.array): Array of canalizing input values.
                - CanalizingOutputs (np.array): Array of canalized output values.
                - CoreFunction (np.array): The core function (remaining truth table) after removing canalizing variables.
                - OrderOfVariables (np.array): Array of indices indicating the order of canalizing variables.
    
    References:
        He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    return bf.is_k_canalizing_return_canalizing_structure(k)

def find_layers(bf):
    """
    Determine the canalizing layer structure of a Boolean function.

    This function decomposes a Boolean function into its canalizing layers (standard monomial form)
    by recursively identifying and removing conditionally canalizing variables.
    The output includes the canalizing depth, the number of layers, the canalizing inputs and outputs,
    the core polynomial, and the order of the canalizing variables.

    Parameters:
        bf (BooleanFunction): Boolean function object.
        can_inputs (np.array, optional): Accumulated canalizing input values (for recursion).
        can_outputs (np.array, optional): Accumulated canalized output values (for recursion).
        can_order (np.array, optional): Accumulated indices of canalizing variables (for recursion).
        variables (list, optional): List of variable indices. If empty, defaults to range(n).
        depth (int, optional): Current canalizing depth (for recursion); default is 0.
        number_layers (int, optional): Current number of layers identified (for recursion); default is 0.

    Returns:
        Returns:
            dict: A dictionary containing:
                - Depth (int): Canalizing depth (number of conditionally canalizing variables).
                - NumberOfLayers (int): Number of distinct canalizing layers.
                - CanalizingInputs (np.array): Array of canalizing input values.
                - CanalizingOutputs (np.array): Array of canalized output values.
                - CoreFunction (np.array): The core polynomial (truth table) after removing canalizing variables.
                - OrderOfVariables (np.array): Array of indices representing the order of canalizing variables.
    
    References:
        He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    return bf.find_layers()
    

def get_layerstructure_given_canalizing_outputs_and_corefunction(can_outputs, core_polynomial):
    """
    Compute the canalizing layer structure of a Boolean function given its canalized outputs and core polynomial.

    Two consecutive canalizing variables belong to the same layer if they have the same canalized output, and to different layers otherwise.
    The resulting layer structure is a list [k_1, ..., k_r] indicating the number of variables in each canalizing layer.
    For nested canalizing functions (NCFs) with n > 1, the last layer must have at least two variables.

    Parameters:
        can_outputs (list): List of all canalized output values of the function.
        core_polynomial (list or np.array): Core function (or polynomial) of length 2^(n - depth) after removing canalizing variables.

    Returns:
        list: A list [k_1, ..., k_r] describing the number of variables in each canalizing layer.
              Each k_i ≥ 1, and if the function is an NCF (i.e., sum(k_i) == n), then the last layer k_r ≥ 2 (unless n == 1).

    References:
        [1] Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness of Boolean networks.
            Physica D: Nonlinear Phenomena, 353, 39-47.
        [2] Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    canalizing_depth = len(can_outputs)
    if canalizing_depth == 0:
        return []
    n = int(np.log2(len(core_polynomial))) + canalizing_depth
    if canalizing_depth == n and n > 1:  # For Boolean NCFs, the last layer must have at least two variables.
        can_outputs[-1] = can_outputs[-2]
    elif canalizing_depth == n-1 and len(core_polynomial)==2: #last variable (in core polynomial) must also be canalizing
        canalizing_depth = n
        can_outputs.append(can_outputs[-1])
    elif is_constant(core_polynomial) and canalizing_depth > 1:  # Exceptional case: last layer needs to be size ≥ 2.
        can_outputs[-1] = can_outputs[-2]
    layerstructure = []
    size_of_layer = 1
    for i in range(1, canalizing_depth):
        if can_outputs[i] == can_outputs[i - 1]:
            size_of_layer += 1
        else:
            layerstructure.append(size_of_layer)
            size_of_layer = 1
    layerstructure.append(size_of_layer)
    return layerstructure


def get_layerstructure_of_an_NCF_given_its_Hamming_weight(n, w):
    """
    Compute the canalizing layer structure of a nested canalizing function (NCF) given its Hamming weight.

    There exists a bijection between the Hamming weight (with w equivalent to 2^n - w) and the canalizing layer structure of an NCF.
    The layer structure is represented as [k_1, ..., k_r], where each k_i ≥ 1 and, if n > 1, for the last layer k_r ≥ 2.

    Parameters:
        n (int): Number of inputs (variables) of the NCF.
        w (int): Odd Hamming weight of the NCF, i.e., the number of 1s in the 2^n-vector representation of the function.

    Returns:
        tuple: A tuple (r, layerstructure_NCF), where:
            - r (int): The number of canalizing layers.
            - layerstructure_NCF (list): A list [k_1, ..., k_r] describing the number of variables in each layer.

    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness of Boolean networks.
        Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    if w == 1:
        r = 1
        layerstructure_NCF = [n]
    else:
        assert type(w) == int or type(w) == np.int64, 'Hamming weight must be an integer'
        assert 1 <= w <= 2**n - 1, 'Hamming weight w must satisfy 1 <= w <= 2^n - 1'
        assert w % 2 == 1, 'Hamming weight must be an odd integer since all NCFs have an odd Hamming weight.'
        w_bin = utils.dec2bin(w, n)
        current_el = w_bin[0]
        layerstructure_NCF = [1]
        for el in w_bin[1:-1]:
            if el == current_el:
                layerstructure_NCF[-1] += 1
            else:
                layerstructure_NCF.append(1)
                current_el = el
        layerstructure_NCF[-1] += 1
        r = len(layerstructure_NCF)
    return (r, layerstructure_NCF)


##k-set canalization, canalizing strength
def get_proportion_of_collectively_canalizing_input_sets(bf, k, left_side_of_truth_table=None, verbose=False):
    """
    Compute the proportion of k-set canalizing input sets for a Boolean function.

    For a given k, this function calculates the probability that a randomly chosen set of k inputs canalizes the function,
    i.e., forces the output regardless of the remaining variables.

    Parameters:
        bf (BooleanFunction): Boolean function object.
        k (int): The size of the variable set (0 ≤ k ≤ n).
        left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
            If not provided or if its shape does not match, it will be computed.
        verbose (bool, optional): If True, prints detailed information about canalizing k-sets.

    Returns:
        float: The proportion of k-set canalizing input sets.
    
    References:
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
        Advances in Applied Mathematics, 145, 102475.
    """
    return bf.get_proportion_of_collectively_canalizing_input_sets(k, left_side_of_truth_table, verbose)


def is_kset_canalizing(bf, k, left_side_of_truth_table = None):
    """
    Determine if a Boolean function is k-set canalizing.

    A Boolean function is k-set canalizing if there exists a set of k variables such that setting these variables to specific values
    forces the output of the function, irrespective of the other n - k inputs.

    Parameters:
        bf (BooleanFunction): Boolean function object.
        k (int): The size of the variable set (with 0 ≤ k ≤ n).
        left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
            If not provided or if its shape does not match, it will be computed.
            
    Returns:
        bool: True if f is k-set canalizing, False otherwise.

    References:
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
        Advances in Applied Mathematics, 145, 102475.
    """
    return bf.is_kset_canalizing(k, left_side_of_truth_table)


def get_canalizing_strength(bf, left_side_of_truth_table=None):
    """
    Compute the canalizing strength of a Boolean function via exhaustive enumeration.

    The canalizing strength is defined as a weighted average of the proportions of k-set canalizing inputs for k = 1 to n-1.
    It is 0 for minimally canalizing functions (e.g., Boolean parity functions) and 1 for maximally canalizing functions
    (e.g., nested canalizing functions with one layer).

    Parameters:
        bf (BooleanFunction): Boolean function object.
        left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
            If not provided or if its shape does not match, it will be computed.
            
    Returns:
        tuple:
            - float: The canalizing strength of f.
            - list: A list of the k-set canalizing proportions for k = 1, 2, ..., n-1.
    
    References:
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
        Advances in Applied Mathematics, 145, 102475.
    """
    return bf.get_canalizing_strength(left_side_of_truth_table)


def compute_exact_kset_canalizing_proportion_for_ncf_with_specific_layerstructure(k, layerstructure_NCF):
    """
    Compute the exact k-set canalizing proportion for a nested canalizing function (NCF) with a specific layer structure.

    This function implements Theorem 3.3 from [1] and computes the exact proportion of k-set canalizing inputs for an NCF
    characterized by its layer structure.

    Parameters:
        k (int): The size of the variable set (0 ≤ k ≤ n) for which the canalizing proportion is computed.
        layerstructure_NCF (list): List of integers [k_1, ..., k_r] describing the number of variables in each layer of an NCF.
            Each k_i must be at least 1, and the last layer must have at least 2 variables unless n == 1.

    Returns:
        float: The exact k-set canalizing proportion for the NCF with the provided layer structure.
    
    References:
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
        Advances in Applied Mathematics, 145, 102475.
    """
    r = len(layerstructure_NCF)
    n = sum(layerstructure_NCF)
    assert min(layerstructure_NCF) >= 1 and (layerstructure_NCF[-1] >= 2 or n == 1), \
        "Each layer must contain at least one variable (the last layer at least two unless n == 1)"
    magnitudes = []
    for t in range(r):
        number_of_input_sets = 0
        for c in range(1, min(k - sum(layerstructure_NCF[:t][::-2]), layerstructure_NCF[t]) + 1):
            for d in range(0, min(k - sum(layerstructure_NCF[:t][::-2]) - c, sum(layerstructure_NCF[:max(0, t - 1)][::-2])) + 1):
                binom1 = binom(layerstructure_NCF[t], c)
                binom2 = binom(sum(layerstructure_NCF[:max(0, t - 1)][::-2]), d)
                binom3 = binom(n - sum(layerstructure_NCF[:t + 1]), k - sum(layerstructure_NCF[:t][::-2]) - c - d)
                number_of_inputs_that_canalize_for_selected_variable_set = sum([2**(k - sum(layerstructure_NCF[:t][::-2]) - j - d) for j in range(1, c + 1)])
                number_of_input_sets += binom1 * binom2 * binom3 * number_of_inputs_that_canalize_for_selected_variable_set
        magnitudes.append(number_of_input_sets)
    # For the case where the non-canalizing output value can be reached in the evaluation process, add:
    if k >= sum(layerstructure_NCF[-1::-2]):
        magnitudes.append(binom(n - sum(layerstructure_NCF[-1::-2]), k - sum(layerstructure_NCF[-1::-2])))
    else:
        magnitudes.append(0)
    return sum(magnitudes) / (2**k * binom(n, k))


## input redundancy, edge effectiveness, effective degree
if __LOADED_CANA__:
    def get_input_redundancy(bf):
        """
        Compute the input redundancy of a Boolean function.
    
        The input redundancy quantifies how many inputs are not required to determine the function’s output.
        Constant functions have an input redundancy of 1 (none of the inputs are needed), whereas parity functions have an input redundancy of 0 (all inputs are necessary).
    
        Parameters:
            bf (BooleanFunction): Boolean function object.
    
        Returns:
            float: Normalized input redundancy in the interval [0, 1].
    
        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        return bf.get_input_redundancy()
    
    def get_edge_effectiveness(bf):
        """
        Compute the edge effectiveness for each regulator of a Boolean function.
    
        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.
    
        Parameters:
            bf (BooleanFunction): Boolean function object.
    
        Returns:
            list: A list of n floats in [0, 1] representing the edge effectiveness for each input.
    
        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        return bf.get_edge_effectiveness()
    
    def get_effective_degree(bf):
        """
        Compute the effective degree, i.e., the sum of the edge effectivenesses of each regulator, of a Boolean function.
    
        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.
    
        Parameters:
            bf (BooleanFunction): Boolean function object.
    
        Returns:
            list: A value in [0, 1] representing the effective degree for each input.
    
        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        return bf.get_effective_degree()