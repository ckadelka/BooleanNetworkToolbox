#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 09:25:40 2025

@author: Claus Kadelka
"""

##Imports
import numpy as np
import itertools
import utils
from scipy.special import binom

try:
    import cana.boolean_node
    LOADED_CANA=True
except ModuleNotFoundError:
    print('The module cana cannot be found. Ensure it is installed to use all functionality of this toolbox.')
    LOADED_CANA=False


## constant functions
def is_constant(f):
    """
    Check whether a Boolean function is constant.

    Parameters:
        f (list): Boolean function as a list of length 2^n (truth table), where n is the number of inputs.

    Returns:
        bool: True if f is constant (all outputs are 0 or all are 1), False otherwise.
    """
    return sum(f) in [0, len(f)]


## degenerated functions / essential variables
def is_degenerated(f):
    """
    Determine if a Boolean function contains non-essential variables.

    A variable is non-essential if the function's output does not depend on it.

    Parameters:
        f (list): Boolean function represented as a list of length 2^n (truth table), where n is the number of inputs.

    Returns:
        bool: True if f contains at least one non-essential variable, False if all variables are essential.
    """
    len_f = len(f)
    n = int(np.log2(len_f))
    for i in range(n):
        dummy_add = (2**(n-1-i))
        dummy = np.arange(2**n) % (2**(n-i)) // dummy_add
        depends_on_i = False
        for j in range(2**n):
            if dummy[j] == 1:
                continue
            else:
                if f[j] != f[j + dummy_add]:
                    depends_on_i = True
                    break
        if depends_on_i == False:
            return True
    return False

def get_essential_variables(f):
    """
    Determine the indices of essential variables in a Boolean function.

    A variable is essential if changing its value (while holding the others constant) can change the output of f.

    Parameters:
        f (list): Boolean function as a list of length 2^n (truth table), where n is the number of inputs.

    Returns:
        list: List of indices corresponding to the essential variables.
    """
    if len(f) == 0:
        return []
    len_f = len(f)
    n = int(np.log2(len_f))
    essential_variables = list(range(n))
    for i in range(n):
        dummy_add = (2**(n-1-i))
        dummy = np.arange(2**n) % (2**(n-i)) // dummy_add
        depends_on_i = False
        for j in range(2**n):
            if dummy[j] == 1:
                continue
            else:
                if f[j] != f[j + dummy_add]:
                    depends_on_i = True
                    break
        if depends_on_i == False:
            essential_variables.remove(i)
    return essential_variables 


def get_number_of_essential_variables(f):
    """
    Count the number of essential variables in a Boolean function.

    Parameters:
        f (list): Boolean function as a list of length 2^n (truth table), where n is the number of inputs.

    Returns:
        int: The number of essential variables.
    """
    return len(get_essential_variables(f))


## monotonicity, type of inputs
def is_monotonic(f,GET_DETAILS=False):
    """
    Determine if a Boolean function is monotonic.

    A Boolean function is monotonic if it is monotonic in each variable. 
    That is, if for all i=1,...,n:
        f(x_1,...,x_i=0,...,x_n) >= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n) or 
        f(x_1,...,x_i=0,...,x_n) <= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n)

    Parameters:
        f (list): Boolean function represented as a list of length 2^n (truth table), where n is the number of inputs.
        GET_DETAILS (bool, optional): If True, the type of each variable (increasing, decreasing, not monotonic, not essential) is returned. 
    Returns:
        bool: True if f contains at least one non-essential variable, False if all variables are essential.
        list: List containing the type of regulation of each variable. Only returned if GET_DETAILS==True.
    """
    n=int(np.log2(len(f)))
    f = np.array(f)
    monotonic = []
    for i in range(n):
        dummy_add=(2**(n-1-i))
        dummy=np.arange(2**n)%(2**(n-i))//dummy_add
        diff = f[dummy==1]-f[dummy==0]
        min_diff = min(diff)
        max_diff = max(diff)
        if min_diff==0 and max_diff==0:
            monotonic.append('not essential')
        elif min_diff==-1 and max_diff==1:
            monotonic.append('not monotonic')
        elif min_diff>=0 and max_diff==1:
            monotonic.append('increasing')            
        elif min_diff==-1 and max_diff<=0:
            monotonic.append('decreasing')   
    if GET_DETAILS:
        return ('not essential' not in monotonic,monotonic)
    else:
        return 'not essential' not in monotonic


## symmetry
def get_symmetry_groups(f, left_side_of_truth_table=None):
    """
    Determine all symmetry groups of input variables for a Boolean function.

    Two variables are in the same symmetry group if swapping their values does not change the output
    of the function for any input of the other variables.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
        left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
            If not provided or if its shape does not match, it will be computed.

    Returns:
        list: A list of lists where each inner list contains indices of variables that form a symmetry group.
    """
    len_f = len(f)
    n = int(np.log2(len_f))
    if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len_f:
        left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=n)))
    symmetry_groups = []
    left_to_check = np.ones(n)
    for i in range(n):
        if left_to_check[i] == 0:
            continue
        else:
            symmetry_groups.append([i])
            left_to_check[i] = 0
        for j in range(i + 1, n):
            diff = sum(2**np.arange(n - i - 2, n - j - 2, -1))
            for ii, x in enumerate(left_side_of_truth_table):
                if x[i] != x[j] and x[i] == 0 and f[ii] != f[ii + diff]:
                    break
            else:
                left_to_check[j] = 0
                symmetry_groups[-1].append(j)
    return symmetry_groups


## bias
def get_absolute_bias(f):
    """
    Compute the absolute bias of a Boolean function.

    The absolute bias is defined as |(sum(f) / 2^(n-1)) - 1|, which quantifies how far the function's output distribution
    deviates from being balanced.

    Parameters:
        f (list or np.array): Boolean function (truth table) of length 2^n.

    Returns:
        float: The absolute bias of the Boolean function.
    """
    n = int(np.log2(len(f)))
    return abs(sum(f) * 1.0 / 2**(n - 1) - 1)


## average sensitivity
def get_average_sensitivity(f, nsim=10000, EXACT=False, NORMALIZED=True):
    """
    Compute the average sensitivity of a Boolean function.

    The average sensitivity is equivalent to the Derrida value D(F,1) when the update rule is sampled
    from the same space. This function can compute the exact sensitivity by exhaustively iterating over all inputs (if EXACT is True)
    or estimate it via Monte Carlo sampling (if EXACT is False). The result can be normalized by the number of inputs.

    Parameters:
        f (list or np.array): Boolean function (truth table) of length 2^n, where n is the number of inputs.
        nsim (int, optional): Number of random samples (default is 10000, used when EXACT is False).
        EXACT (bool, optional): If True, compute the exact sensitivity by iterating over all inputs; otherwise, use sampling (default).
        NORMALIZED (bool, optional): If True, return the normalized sensitivity (divided by the number of function inputs); otherwise, return the total count.

    Returns:
        float: The (normalized) average sensitivity of the Boolean function.
    """
    if type(f) == list:
        f = np.array(f)
    n = int(np.log2(len(f)))
    num_values = 2**n
    s = 0
    if EXACT:
        left_side_of_truth_table = list(map(np.array, list(itertools.product([0, 1], repeat=n))))
        for ii, X in enumerate(left_side_of_truth_table):
            for i in range(n):
                Y = X.copy()
                Y[i] = 1 - X[i]
                Ydec = utils.bin2dec(Y)
                s += int(f[ii] != f[Ydec])
        if NORMALIZED:
            return s / (num_values * n)
        else:
            return s / num_values
    else:
        for i in range(nsim):
            xdec = np.random.randint(num_values)
            Y = utils.dec2bin(xdec, n)
            index = np.random.randint(n)
            Y[index] = 1 - Y[index]
            Ybin = utils.bin2dec(Y)
            s += int(f[xdec] != f[Ybin])
        if NORMALIZED:
            return s / nsim
        else:
            return n * s / nsim


## canalization, k-canalization, canalizing layer structure, canalizing depth
def is_canalizing(f):
    """
    Determine if a Boolean function is canalizing.

    A Boolean function f(x_1, ..., x_n) is canalizing if there exists at least one variable x_i and a value a ∈ {0, 1} 
    such that f(x_1, ..., x_i = a, ..., x_n) is constant.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.

    Returns:
        bool: True if f is canalizing, False otherwise.
    """
    if type(f) == list:
        f = np.array(f)
    n = int(np.log2(len(f)))
    desired_value = 2**(n - 1)
    T = np.array(list(itertools.product([0, 1], repeat=n))).T
    A = np.r_[T, 1 - T]
    Atimesf = np.dot(A, f)
    if np.any(Atimesf == desired_value):
        return True
    elif np.any(Atimesf == 0):
        return True
    else:
        return False


def is_k_canalizing(f, k):
    """
    Determine if a Boolean function is k-canalizing.

    A Boolean function is k-canalizing if it has at least k conditionally canalizing variables.
    This is checked recursively: after fixing a canalizing variable (with a fixed canalizing input that forces the output),
    the subfunction (core function) must itself be canalizing for the next variable, and so on.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
        k (int): The desired canalizing depth (0 ≤ k ≤ n). Note: every function is 0-canalizing.

    Returns:
        bool: True if f is k-canalizing, False otherwise.
    
    References:
        He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    n = int(np.log2(len(f)))
    if k > n:
        return False
    if k == 0:
        return True

    w = sum(f)  # Hamming weight of f
    if w == 0 or w == 2**n:  # constant function
        return False
    if type(f) == list:
        f = np.array(f)
    desired_value = 2**(n - 1)
    T = np.array(list(itertools.product([0, 1], repeat=n))).T
    A = np.r_[T, 1 - T]
    try:  # check for canalizing output 1
        index = list(np.dot(A, f)).index(desired_value)
        new_f = f[np.where(A[index] == 0)[0]]
        return is_k_canalizing(new_f, k - 1, n - 1)
    except ValueError:
        try:  # check for canalizing output 0
            index = list(np.dot(A, 1 - f)).index(desired_value)
            new_f = f[np.where(A[index] == 0)[0]]
            return is_k_canalizing(new_f, k - 1, n - 1)
        except ValueError:
            return False


def is_k_canalizing_return_inputs_outputs_corefunction(f, k, can_inputs=np.array([], dtype=int), can_outputs=np.array([], dtype=int)):
    """
    Determine if a Boolean function is k-canalizing and return associated canalizing data.

    This function recursively checks whether f is k-canalizing and returns:
      - A boolean indicating success.
      - The canalizing input values.
      - The canalized output values.
      - The core function that remains after removing the canalizing variables.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
        k (int): The canalizing depth to check.
        can_inputs (np.array, optional): Accumulated canalizing input values (default is an empty array).
        can_outputs (np.array, optional): Accumulated canalized output values (default is an empty array).

    Returns:
        tuple: A tuple containing:
            - bool: True if f is k-canalizing, False otherwise.
            - np.array: Array of canalizing input values.
            - np.array: Array of canalized output values.
            - np.array: The core function (remaining truth table) after canalizing variables are removed.
    
    References:
        He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    if k == 0:
        return (True, can_inputs, can_outputs, f)
    n = int(np.log2(len(f)))
    w = sum(f)
    if w == 0 or w == 2**n:  # constant function
        return (False, can_inputs, can_outputs, f)
    if type(f) == list:
        f = np.array(f)
    desired_value = 2**(n - 1)
    T = np.array(list(itertools.product([0, 1], repeat=n))).T
    A = np.r_[T, 1 - T]
    try:  # check for canalizing output 1
        index = list(np.dot(A, f)).index(desired_value)
        new_f = f[np.where(A[index] == 0)[0]]
        return is_k_canalizing_return_inputs_outputs_corefunction(new_f, k - 1, 
                                                                  np.append(can_inputs, int(index < n)),
                                                                  np.append(can_outputs, 1))
    except ValueError:
        try:  # check for canalizing output 0
            index = list(np.dot(A, 1 - f)).index(desired_value)
            new_f = f[np.where(A[index] == 0)[0]]
            return is_k_canalizing_return_inputs_outputs_corefunction(new_f, k - 1, 
                                                                      np.append(can_inputs, int(index < n)),
                                                                      np.append(can_outputs, 0))
        except ValueError:
            return (False, can_inputs, can_outputs, f)


def is_k_canalizing_return_inputs_outputs_corefunction_order(f, k, can_inputs=np.array([], dtype=int),
                                                            can_outputs=np.array([], dtype=int), can_order=np.array([], dtype=int),
                                                            variables=[]):
    """
    Determine if a Boolean function is k-canalizing and return canalizing data including variable order.

    This function extends the k-canalizing check by additionally returning the order (indices) of the canalizing variables.
    It recursively collects:
      - Canalizing input values.
      - Canalized output values.
      - The core function after removing the canalizing layers.
      - The order of the canalizing variables.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
        k (int): The canalizing depth to check.
        can_inputs (np.array, optional): Accumulated canalizing input values.
        can_outputs (np.array, optional): Accumulated canalized output values.
        can_order (np.array, optional): Accumulated order (indices) of canalizing variables.
        variables (list, optional): List of variable indices. If empty, defaults to range(n).

    Returns:
        tuple: A tuple containing:
            - bool: True if f is k-canalizing, False otherwise.
            - np.array: Array of canalizing input values.
            - np.array: Array of canalized output values.
            - np.array: The core function (remaining truth table) after removing canalizing variables.
            - np.array: Array of indices indicating the order of canalizing variables.
    
    References:
        He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    if k == 0:
        return (True, can_inputs, can_outputs, f, can_order)
    w = sum(f)
    n = int(np.log2(len(f)))
    if w == 0 or w == 2**n:  # constant function
        return (False, can_inputs, can_outputs, f, can_order)
    if type(variables) == np.ndarray:
        variables = list(variables)
    if variables == []:
        variables = list(range(n))
    if type(f) == list:
        f = np.array(f)
    desired_value = 2**(n - 1)
    T = np.array(list(itertools.product([0, 1], repeat=n))).T
    A = np.r_[T, 1 - T]
    try:  # check for canalizing output 0
        index = list(np.dot(A, 1 - f)).index(desired_value)
        newF = f[np.where(A[index] == 0)[0]]
        variable = variables.pop(index % n)
        return is_k_canalizing_return_inputs_outputs_corefunction_order(newF, k - 1, 
                                                                        np.append(can_inputs, int(index < n)),
                                                                        np.append(can_outputs, 0),
                                                                        np.append(can_order, variable),
                                                                        variables)
    except ValueError:
        try:  # check for canalizing output 1
            index = list(np.dot(A, f)).index(desired_value)
            newF = f[np.where(A[index] == 0)[0]]
            variable = variables.pop(index % n)
            return is_k_canalizing_return_inputs_outputs_corefunction_order(newF, k - 1, 
                                                                            np.append(can_inputs, int(index < n)),
                                                                            np.append(can_outputs, 1),
                                                                            np.append(can_order, variable),
                                                                            variables)
        except ValueError:
            return (False, can_inputs, can_outputs, f, can_order)


def find_layers(f, can_inputs=np.array([], dtype=int), can_outputs=np.array([], dtype=int),
                can_order=np.array([], dtype=int), variables=[], depth=0, number_layers=0):
    """
    Determine the canalizing layer structure of a Boolean function.

    This function decomposes a Boolean function into its canalizing layers (standard monomial form)
    by recursively identifying and removing conditionally canalizing variables.
    The output includes the canalizing depth, the number of layers, the canalizing inputs and outputs,
    the core polynomial, and the order of the canalizing variables.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
        can_inputs (np.array, optional): Accumulated canalizing input values (for recursion).
        can_outputs (np.array, optional): Accumulated canalized output values (for recursion).
        can_order (np.array, optional): Accumulated indices of canalizing variables (for recursion).
        variables (list, optional): List of variable indices. If empty, defaults to range(n).
        depth (int, optional): Current canalizing depth (for recursion); default is 0.
        number_layers (int, optional): Current number of layers identified (for recursion); default is 0.

    Returns:
        tuple: A tuple containing:
            - int: Canalizing depth (number of conditionally canalizing variables).
            - int: Number of distinct canalizing layers.
            - np.array: Array of canalizing input values.
            - np.array: Array of canalized output values.
            - np.array: The core polynomial (truth table) after removing canalizing variables.
            - np.array: Array of indices representing the order of canalizing variables.
    
    References:
        He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
            Algorithms and applications. Automatica, 146, 110630.
    """
    n = int(np.log2(len(f)))
    w = sum(f)
    if w == 0 or w == 2**n:  # constant function
        return (depth, number_layers, can_inputs, can_outputs, f, can_order)
    if type(variables) == np.ndarray:
        variables = list(variables)
    if variables == []:
        variables = list(range(n))
    if type(f) == list:
        f = np.array(f)
    desired_value = 2**(n - 1)
    T = np.array(list(itertools.product([0, 1], repeat=n))).T
    A = np.r_[T, 1 - T]

    indices1 = np.where(np.dot(A, f) == desired_value)[0]
    indices0 = np.where(np.dot(A, 1 - f) == desired_value)[0]
    if len(indices1) > 0:
        sorted_order = sorted(range(len(indices1)), key=lambda x: (indices1 % n)[x])
        inputs = (1 - indices1 // n)[np.array(sorted_order)]
        outputs = np.ones(len(indices1), dtype=int)
        new_canalizing_variables = []
        for index in np.sort(indices1 % n)[::-1]:
            new_canalizing_variables.append(variables.pop(index))
        new_canalizing_variables.reverse()
        new_f = f[np.sort(list(set.intersection(*[] + [set(np.where(A[index] == 0)[0]) for index, INPUT in zip(indices1, inputs)])))]
        return find_layers(new_f, np.append(can_inputs, inputs), np.append(can_outputs, outputs),
                           np.append(can_order, new_canalizing_variables), variables, depth + len(new_canalizing_variables),
                           number_layers + 1)
    elif len(indices0):
        sorted_order = sorted(range(len(indices0)), key=lambda x: (indices0 % n)[x])
        inputs = (1 - indices0 // n)[np.array(sorted_order)]
        outputs = np.zeros(len(indices0), dtype=int)
        new_canalizing_variables = []
        for index in np.sort(indices0 % n)[::-1]:
            new_canalizing_variables.append(variables.pop(index))
        new_canalizing_variables.reverse()
        new_f = f[np.sort(list(set.intersection(*[] + [set(np.where(A[index] == 0)[0]) for index, INPUT in zip(indices0, inputs)])))]
        return find_layers(new_f, np.append(can_inputs, inputs), np.append(can_outputs, outputs),
                           np.append(can_order, new_canalizing_variables), variables, depth + len(new_canalizing_variables),
                           number_layers + 1)
    else:
        return (depth, number_layers, can_inputs, can_outputs, f, can_order)
    

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
def get_proportion_of_collectively_canalizing_input_sets(f, k, left_side_of_truth_table=None, verbose=False):
    """
    Compute the proportion of k-set canalizing input sets for a Boolean function.

    For a given k, this function calculates the probability that a randomly chosen set of k inputs canalizes the function,
    i.e., forces the output regardless of the remaining variables.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
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
    if type(f) == list:
        f = np.array(f)
    if k == 0:
        return float(is_constant(f))
    n = int(np.log2(len(f)))
    desired_value = 2**(n - k)
    if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len(f):
        left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=n)))
    T = left_side_of_truth_table.T
    Tk = list(itertools.product([0, 1], repeat=k))
    A = np.r_[T, 1 - T]
    Ak = []
    for indices in itertools.combinations(range(n), k):
        for canalizing_inputs in Tk:
            indices_values = np.array(indices) + n * np.array(canalizing_inputs)
            dummy = np.sum(A[indices_values, :], 0) == k
            if sum(dummy) == desired_value:
                Ak.append(dummy)
                if verbose and np.dot(dummy, f) in [0, desired_value]:
                    print(indices, canalizing_inputs, indices_values, np.dot(dummy, f))
            elif verbose:
                print(indices, canalizing_inputs, sum(dummy), 'a')
    Ak = np.array(Ak)
    is_there_canalization = np.in1d(np.dot(Ak, f), [0, desired_value])
    return sum(is_there_canalization) / len(is_there_canalization)


def is_kset_canalizing(f, k, left_side_of_truth_table = None):
    """
    Determine if a Boolean function is k-set canalizing.

    A Boolean function is k-set canalizing if there exists a set of k variables such that setting these variables to specific values
    forces the output of the function, irrespective of the other n - k inputs.

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
        k (int): The size of the variable set (with 0 ≤ k ≤ n).
        left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
            If not provided or if its shape does not match, it will be computed.
            
    Returns:
        bool: True if f is k-set canalizing, False otherwise.

    References:
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
        Advances in Applied Mathematics, 145, 102475.
    """
    if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len(f):
        n = int(np.log2(len(f)))
        left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=n)))
    return get_proportion_of_collectively_canalizing_input_sets(f,k,left_side_of_truth_table)>0


def get_canalizing_strength(f, left_side_of_truth_table=None):
    """
    Compute the canalizing strength of a Boolean function via exhaustive enumeration.

    The canalizing strength is defined as a weighted average of the proportions of k-set canalizing inputs for k = 1 to n-1.
    It is 0 for minimally canalizing functions (e.g., Boolean parity functions) and 1 for maximally canalizing functions
    (e.g., nested canalizing functions with one layer).

    Parameters:
        f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.
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
    nfloat = np.log2(len(f))
    n = int(nfloat)
    assert abs(n - nfloat) < 1e-10, "f needs to be of length 2^n for some n > 1"
    assert n > 1, "Canalizing strength is only defined for Boolean functions with n > 1 inputs"
    res = []
    if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len(f):
        left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=n)))
    for k in range(1, n):
        res.append(get_proportion_of_collectively_canalizing_input_sets(f, k, left_side_of_truth_table=left_side_of_truth_table))
    return np.mean(np.multiply(res, 2**np.arange(1, n) / (2**np.arange(1, n) - 1))), res


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
if LOADED_CANA:
    def get_input_redundancy(f):
        """
        Compute the input redundancy of a Boolean function.

        The input redundancy quantifies how many inputs are not required to determine the function’s output.
        Constant functions have an input redundancy of 1 (none of the inputs are needed), whereas parity functions have an input redundancy of 0 (all inputs are necessary).

        Parameters:
            f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.

        Returns:
            float: Normalized input redundancy in the interval [0, 1].

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        n = int(np.log2(len(f)))
        return cana.boolean_node.BooleanNode(k=n, outputs=f).input_redundancy()

    def get_edge_effectiveness(f):
        """
        Compute the edge effectiveness for each regulator of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.

        Parameters:
            f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.

        Returns:
            list: A list of n floats in [0, 1] representing the edge effectiveness for each input.

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        n = int(np.log2(len(f)))
        return cana.boolean_node.BooleanNode(k=n, outputs=f).edge_effectiveness()
    
    def get_effective_degree(f):
        """
        Compute the effective degree, i.e., the sum of the edge effectivenesses of each regulator, of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.

        Parameters:
            f (list or np.array): Boolean function of length 2^n (truth table), where n is the number of inputs.

        Returns:
            list: A value in [0, 1] representing the effective degree for each input.

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """       
        return sum(get_edge_effectiveness(f))



