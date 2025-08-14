#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:03:49 2025
Last Edited on Wed Aug 13 2025

@author: Benjamin Coberly
"""

import numpy as np
import itertools

import utils

try:
    import cana.boolean_node
    __LOADED_CANA__=True
except ModuleNotFoundError:
    print('The module cana cannot be found. Ensure it is installed to use all functionality of this toolbox.')
    __LOADED_CANA__=False

def from_cana_BooleanNode(BooleanNode):
    return BooleanFunction(f=BooleanNode.outputs)

class BooleanFunction:
    def __init__(self, f):
        assert type(f) in [ list, np.array], "f must be an array"
        if type(f) == list:
            f = np.array(f)
        self.f = f
        if len(self.f) == 0:
            self.n = 0
        else:
            self.n = int(np.log2(len(f)))
    
    def is_constant(self):
        """
        Check whether a Boolean function is constant.
        
        Returns:
            bool: True if f is constant (all outputs are 0 or all are 1), False otherwise.
        """
        return sum(self.f) in [0, len(self.f)]
    
    def is_degenerated(self):
        """
        Determine if a Boolean function contains non-essential variables.

        A variable is non-essential if the function's output does not depend on it.
        
        Returns:
            bool: True if f contains at least one non-essential variable, False if all variables are essential.
        """
        for i in range(self.n):
            dummy_add = (2**(self.n-1-i))
            dummy = np.arange(2**self.n) % (2**(self.n-i)) // dummy_add
            depends_on_i = False
            for j in range(2**self.n):
                if dummy[j] == 1:
                    continue
                else:
                    if self.f[j] != self.f[j + dummy_add]:
                        depends_on_i = True
                        break
            if depends_on_i == False:
                return True
        return False

    def get_essential_variables(self):
        """
        Determine the indices of essential variables in a Boolean function.

        A variable is essential if changing its value (while holding the others constant) can change the output of f.
        
        Returns:
            list: List of indices corresponding to the essential variables.
        """
        if len(self.f) == 0:
            return []
        essential_variables = list(range(self.n))
        for i in range(self.n):
            dummy_add = (2**(self.n-1-i))
            dummy = np.arange(2**self.n) % (2**(self.n-i)) // dummy_add
            depends_on_i = False
            for j in range(2**self.n):
                if dummy[j] == 1:
                    continue
                else:
                    if self.f[j] != self.f[j + dummy_add]:
                        depends_on_i = True
                        break
            if depends_on_i == False:
                essential_variables.remove(i)
        return essential_variables 

    def get_number_of_essential_variables(self):
        """
        Count the number of essential variables in a Boolean function.
        
        Returns:
            int: The number of essential variables.
        """
        return len(self.get_essential_variables())
    
    def is_monotonic(self, GET_DETAILS=False):
        """
        Determine if a Boolean function is monotonic.

        A Boolean function is monotonic if it is monotonic in each variable. 
        That is, if for all i=1,...,n:
            f(x_1,...,x_i=0,...,x_n) >= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n) or 
            f(x_1,...,x_i=0,...,x_n) <= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n)

        Parameters:
            GET_DETAILS (bool, optional): If True, the type of each variable (increasing, decreasing, not monotonic, not essential) is returned. 
        Returns:
            bool: True if f contains at least one non-essential variable, False if all variables are essential.
            list: List containing the type of regulation of each variable. Only returned if GET_DETAILS==True.
        """
        monotonic = []
        for i in range(self.n):
            dummy_add=(2**(self.n-1-i))
            dummy=np.arange(2**self.n)%(2**(self.n-i))//dummy_add
            diff = self.f[dummy==1]-self.f[dummy==0]
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
    
    def get_symmetry_groups(self, left_side_of_truth_table=None):
        """
        Determine all symmetry groups of input variables for a Boolean function.

        Two variables are in the same symmetry group if swapping their values does not change the output
        of the function for any input of the other variables.

        Parameters:
            left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
                If not provided or if its shape does not match, it will be computed.

        Returns:
            list: A list of lists where each inner list contains indices of variables that form a symmetry group.
        """
        if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len(self.f):
            left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=self.n)))
        symmetry_groups = []
        left_to_check = np.ones(self.n)
        for i in range(self.n):
            if left_to_check[i] == 0:
                continue
            else:
                symmetry_groups.append([i])
                left_to_check[i] = 0
            for j in range(i + 1, self.n):
                diff = sum(2**np.arange(self.n - i - 2, self.n - j - 2, -1))
                for ii, x in enumerate(left_side_of_truth_table):
                    if x[i] != x[j] and x[i] == 0 and self.f[ii] != self.f[ii + diff]:
                        break
                else:
                    left_to_check[j] = 0
                    symmetry_groups[-1].append(j)
        return symmetry_groups
    
    def get_absolute_bias(self):
        """
        Compute the absolute bias of a Boolean function.

        The absolute bias is defined as |(sum(f) / 2^(n-1)) - 1|, which quantifies how far the function's output distribution
        deviates from being balanced.
        
        Returns:
            float: The absolute bias of the Boolean function.
        """
        return abs(sum(self.f) * 1.0 / 2**(self.n - 1) - 1)
    
    def get_average_sensitivity(self, nsim=10000, EXACT=False, NORMALIZED=True):
        """
        Compute the average sensitivity of a Boolean function.

        The average sensitivity is equivalent to the Derrida value D(F,1) when the update rule is sampled
        from the same space. This function can compute the exact sensitivity by exhaustively iterating over all inputs (if EXACT is True)
        or estimate it via Monte Carlo sampling (if EXACT is False). The result can be normalized by the number of inputs.

        Parameters:
            nsim (int, optional): Number of random samples (default is 10000, used when EXACT is False).
            EXACT (bool, optional): If True, compute the exact sensitivity by iterating over all inputs; otherwise, use sampling (default).
            NORMALIZED (bool, optional): If True, return the normalized sensitivity (divided by the number of function inputs); otherwise, return the total count.

        Returns:
            float: The (normalized) average sensitivity of the Boolean function.
        """
        num_values = 2**self.n
        s = 0
        if EXACT:
            left_side_of_truth_table = list(map(np.array, list(itertools.product([0, 1], repeat=self.n))))
            for ii, X in enumerate(left_side_of_truth_table):
                for i in range(self.n):
                    Y = X.copy()
                    Y[i] = 1 - X[i]
                    Ydec = utils.bin2dec(Y)
                    s += int(self.f[ii] != self.f[Ydec])
            if NORMALIZED:
                return s / (num_values * self.n)
            else:
                return s / num_values
        else:
            for i in range(nsim):
                xdec = np.random.randint(num_values)
                Y = utils.dec2bin(xdec, self.n)
                index = np.random.randint(self.n)
                Y[index] = 1 - Y[index]
                Ybin = utils.bin2dec(Y)
                s += int(self.f[xdec] != self.f[Ybin])
            if NORMALIZED:
                return s / nsim
            else:
                return self.n * s / nsim
    
    def is_canalizing(self):
        """
        Determine if a Boolean function is canalizing.

        A Boolean function f(x_1, ..., x_n) is canalizing if there exists at least one variable x_i and a value a ∈ {0, 1} 
        such that f(x_1, ..., x_i = a, ..., x_n) is constant.
        
        Returns:
            bool: True if f is canalizing, False otherwise.
        """
        desired_value = 2**(self.n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=self.n))).T
        A = np.r_[T, 1 - T]
        Atimesf = np.dot(A, self.f)
        if np.any(Atimesf == desired_value):
            return True
        elif np.any(Atimesf == 0):
            return True
        else:
            return False
    
    def is_k_canalizing(self, k, __f__ = None):
        """
        Determine if a Boolean function is k-canalizing.

        A Boolean function is k-canalizing if it has at least k conditionally canalizing variables.
        This is checked recursively: after fixing a canalizing variable (with a fixed canalizing input that forces the output),
        the subfunction (core function) must itself be canalizing for the next variable, and so on.

        Parameters:
            k (int): The desired canalizing depth (0 ≤ k ≤ n). Note: every function is 0-canalizing.

        Returns:
            bool: True if f is k-canalizing, False otherwise.
        
        References:
            He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
                Physica D: Nonlinear Phenomena, 314, 1-8.
            Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
                Algorithms and applications. Automatica, 146, 110630.
        """
        if __f__ == None:
            __f__ = self.f
        n = int(np.log2(len(__f__)))
        if k > n:
            return False
        if k == 0:
            return True

        w = sum(__f__)  # Hamming weight of f
        if w == 0 or w == 2**n:  # constant function
            return False
        if type(__f__) == list:
            __f__ = np.array(__f__)
        desired_value = 2**(n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=n))).T
        A = np.r_[T, 1 - T]
        try:  # check for canalizing output 1
            index = list(np.dot(A, __f__)).index(desired_value)
            new_f = __f__[np.where(A[index] == 0)[0]]
            return self.is_k_canalizing(k - 1, __f__=new_f)
        except ValueError:
            try:  # check for canalizing output 0
                index = list(np.dot(A, 1 - __f__)).index(desired_value)
                new_f = __f__[np.where(A[index] == 0)[0]]
                return self.is_k_canalizing(k - 1, __f__=new_f)
            except ValueError:
                return False


    def _is_k_canalizing_return_inputs_outputs_corefunction(self,k,can_inputs,can_outputs):
        """
        Only for internal use by recursively defined is_k_canalizing_return_inputs_outputs_corefunction.
        """
        if k == 0: #any function is 0-canalizing, can immediately return True
            return (True, can_inputs, can_outputs, self)
        w = sum(self.f)
        if w == 0 or w == 2**self.n: #eventually the recursion will end here (if self.f is a constant function)
            return (False, can_inputs, can_outputs, self)
        desired_value = 2**(self.n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=self.n))).T
        A = np.r_[T, 1 - T]
        if len(can_outputs)==0 or can_outputs[-1]==0:
            CHECK_FOR_CANALIZED_VALUE_0_FIRST = True
        else:
            CHECK_FOR_CANALIZED_VALUE_0_FIRST = False
        try:  # check for canalizing output 1
            index = list(np.dot(A, 1 - self.f if CHECK_FOR_CANALIZED_VALUE_0_FIRST else self.f)).index(desired_value)
            new_f = self.f[np.where(A[index] == 0)[0]]
            new_bf = BooleanFunction(list(new_f))
            return new_bf._is_k_canalizing_return_inputs_outputs_corefunction(k - 1, 
                                                                      np.append(can_inputs, int(index < self.n)),
                                                                      np.append(can_outputs, (0 if CHECK_FOR_CANALIZED_VALUE_0_FIRST else 1)),
                                                                      )
        except ValueError:
            try:  # check for canalizing output 0
                index = list(np.dot(A, self.f if CHECK_FOR_CANALIZED_VALUE_0_FIRST else 1-self.f)).index(desired_value)
                new_f = self.f[np.where(A[index] == 0)[0]]
                new_bf = BooleanFunction(list(new_f))
                return new_bf._is_k_canalizing_return_inputs_outputs_corefunction(k - 1, 
                                                                          np.append(can_inputs, int(index < self.n)),
                                                                          np.append(can_outputs, (1 if CHECK_FOR_CANALIZED_VALUE_0_FIRST else 0)),
                                                                          )
            except ValueError: #or the recursion will end here (if self.f is non-canalizing)
                return (False, can_inputs, can_outputs, self)

    def is_k_canalizing_return_inputs_outputs_corefunction(self, k):
        """
        Determine if a Boolean function is k-canalizing and return associated canalizing data.

        This function recursively checks whether f is k-canalizing and returns:
          - A boolean indicating success.
          - The canalizing input values.
          - The canalized output values.
          - The core function that remains after removing the canalizing variables.

        Parameters:
            k (int): The canalizing depth to check.

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
        return self._is_k_canalizing_return_inputs_outputs_corefunction(k,can_inputs=np.array([], dtype=int),can_outputs=np.array([], dtype=int))


    def _is_k_canalizing_return_inputs_outputs_corefunction_order(self, k, can_inputs,can_outputs, can_order,variables):
        """
        Only for internal use by recursively defined is_k_canalizing_return_inputs_outputs_corefunction_order.
        """
        if k == 0: #any function is 0-canalizing, can immediately return True
            return (True, can_inputs, can_outputs, self, can_order)
        w = sum(self.f)
        if w == 0 or w == 2**self.n:  #eventually the recursion will end here (if self.f is a constant function)
            return (False, can_inputs, can_outputs, self, can_order)
        if type(variables) == np.ndarray:
            variables = list(variables)
        if variables == []:
            variables = list(range(self.n))
        desired_value = 2**(self.n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=self.n))).T
        A = np.r_[T, 1 - T]
        if len(can_outputs)==0 or can_outputs[-1]==0:
            CHECK_FOR_CANALIZED_VALUE_0_FIRST = True
        else:
            CHECK_FOR_CANALIZED_VALUE_0_FIRST = False
        try:  # check for the first canalized output 
            index = list(np.dot(A, 1 - self.f if CHECK_FOR_CANALIZED_VALUE_0_FIRST else self.f)).index(desired_value)
            new_f = self.f[np.where(A[index] == 0)[0]]
            variable = variables.pop(index % self.n)
            new_bf = BooleanFunction(list(new_f))
            return new_bf._is_k_canalizing_return_inputs_outputs_corefunction_order(k - 1, 
                                                                            np.append(can_inputs, int(index < self.n)),
                                                                            np.append(can_outputs, (0 if CHECK_FOR_CANALIZED_VALUE_0_FIRST else 1)),
                                                                            np.append(can_order, variable),
                                                                            variables)
        except ValueError:
            try:  # check for canalized output 1
                index = list(np.dot(A, self.f if CHECK_FOR_CANALIZED_VALUE_0_FIRST else 1-self.f)).index(desired_value)
                new_f = self.f[np.where(A[index] == 0)[0]]
                variable = variables.pop(index % self.n)
                new_bf = BooleanFunction(list(new_f))
                return new_bf._is_k_canalizing_return_inputs_outputs_corefunction_order(k - 1, 
                                                                                np.append(can_inputs, int(index < self.n)),
                                                                                np.append(can_outputs, (1 if CHECK_FOR_CANALIZED_VALUE_0_FIRST else 0)),
                                                                                np.append(can_order, variable),
                                                                                variables)
            except ValueError: #or the recursion will end here (if self.f is non-canalizing)
                return (False, can_inputs, can_outputs, self, can_order)
        

    def is_k_canalizing_return_inputs_outputs_corefunction_order(self, k):
        """
        Determine if a Boolean function is k-canalizing and return canalizing data including variable order.

        This function extends the k-canalizing check by additionally returning the order (indices) of the canalizing variables.
        It recursively collects:
          - Canalizing input values.
          - Canalized output values.
          - The core function after removing the canalizing layers.
          - The order of the canalizing variables.

        Parameters:
            k (int): The canalizing depth to check.

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
        return self._is_k_canalizing_return_inputs_outputs_corefunction_order(k,can_inputs=np.array([], dtype=int),can_outputs=np.array([], dtype=int), can_order=np.array([], dtype=int),variables=[])


    def _find_layers(self, can_inputs, can_outputs, can_order, variables, depth, number_layers):
        """
        Only for internal use by recursively defined find_layers.
        """
        n = self.n
        w = sum(self.f)
        if w == 0 or w == 2**n:  #eventually the recursion will end here (if self.f is a constant function)
            return (depth, number_layers, can_inputs, can_outputs, self, can_order)
        if type(variables) == np.ndarray:
            variables = list(variables)
        if variables == []:
            variables = list(range(n))
        desired_value = 2**(n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=n))).T
        A = np.r_[T, 1 - T]

        indices1 = np.where(np.dot(A, self.f) == desired_value)[0]
        indices0 = np.where(np.dot(A, 1 - self.f) == desired_value)[0]
        if len(indices1) > 0:
            sorted_order = sorted(range(len(indices1)), key=lambda x: (indices1 % n)[x])
            inputs = (1 - indices1 // n)[np.array(sorted_order)]
            outputs = np.ones(len(indices1), dtype=int)
            new_canalizing_variables = []
            for index in np.sort(indices1 % n)[::-1]:
                new_canalizing_variables.append(variables.pop(index))
            new_canalizing_variables.reverse()
            new_f = self.f[np.sort(list(set.intersection(*[] + [set(np.where(A[index] == 0)[0]) for index, INPUT in zip(indices1, inputs)])))]
            new_bf = BooleanFunction(list(new_f))
            return new_bf._find_layers(np.append(can_inputs, inputs), np.append(can_outputs, outputs),
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
            new_f = self.f[np.sort(list(set.intersection(*[] + [set(np.where(A[index] == 0)[0]) for index, INPUT in zip(indices0, inputs)])))]
            new_bf = BooleanFunction(list(new_f))
            return new_bf._find_layers(np.append(can_inputs, inputs), np.append(can_outputs, outputs),
                               np.append(can_order, new_canalizing_variables), variables, depth + len(new_canalizing_variables),
                               number_layers + 1)
        else:  #or the recursion will end here (if self.f is non-canalizing) 
            return (depth, number_layers, can_inputs, can_outputs, self, can_order)        

    def find_layers(self):
        """
        Determine the canalizing layer structure of a Boolean function.

        This function decomposes a Boolean function into its canalizing layers (standard monomial form)
        by recursively identifying and removing conditionally canalizing variables.
        The output includes the canalizing depth, the number of layers, the canalizing inputs and outputs,
        the core polynomial, and the order of the canalizing variables.

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
        return self._find_layers(can_inputs=np.array([], dtype=int), can_outputs=np.array([], dtype=int),
                                 can_order=np.array([], dtype=int), variables=[], depth=0, number_layers=0)
    
    def get_proportion_of_collectively_canalizing_input_sets(self, k, left_side_of_truth_table=None, verbose=False):
        """
        Compute the proportion of k-set canalizing input sets for a Boolean function.

        For a given k, this function calculates the probability that a randomly chosen set of k inputs canalizes the function,
        i.e., forces the output regardless of the remaining variables.

        Parameters:
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
        if k == 0:
            return float(self.is_constant())
        desired_value = 2**(self.n - k)
        if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len(self.f):
            left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=self.n)))
        T = left_side_of_truth_table.T
        Tk = list(itertools.product([0, 1], repeat=k))
        A = np.r_[T, 1 - T]
        Ak = []
        for indices in itertools.combinations(range(self.n), k):
            for canalizing_inputs in Tk:
                indices_values = np.array(indices) + self.n * np.array(canalizing_inputs)
                dummy = np.sum(A[indices_values, :], 0) == k
                if sum(dummy) == desired_value:
                    Ak.append(dummy)
                    if verbose and np.dot(dummy, self.f) in [0, desired_value]:
                        print(indices, canalizing_inputs, indices_values, np.dot(dummy, self.f))
                elif verbose:
                    print(indices, canalizing_inputs, sum(dummy), 'a')
        Ak = np.array(Ak)
        is_there_canalization = np.in1d(np.dot(Ak, self.f), [0, desired_value])
        return sum(is_there_canalization) / len(is_there_canalization)

    def is_kset_canalizing(self, k, left_side_of_truth_table = None):
        """
        Determine if a Boolean function is k-set canalizing.

        A Boolean function is k-set canalizing if there exists a set of k variables such that setting these variables to specific values
        forces the output of the function, irrespective of the other n - k inputs.

        Parameters:
            k (int): The size of the variable set (with 0 ≤ k ≤ n).
            left_side_of_truth_table (optional, array-like): Precomputed left-hand side of the truth table (2^n x n).
                If not provided or if its shape does not match, it will be computed.
                
        Returns:
            bool: True if f is k-set canalizing, False otherwise.

        References:
            Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
            Advances in Applied Mathematics, 145, 102475.
        """
        if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len(self.f):
            left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=self.n)))
        return self.get_proportion_of_collectively_canalizing_input_sets(k,left_side_of_truth_table)>0


    def get_canalizing_strength(self, left_side_of_truth_table=None):
        """
        Compute the canalizing strength of a Boolean function via exhaustive enumeration.

        The canalizing strength is defined as a weighted average of the proportions of k-set canalizing inputs for k = 1 to n-1.
        It is 0 for minimally canalizing functions (e.g., Boolean parity functions) and 1 for maximally canalizing functions
        (e.g., nested canalizing functions with one layer).

        Parameters:
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
        nfloat = np.log2(len(self.f))
        assert abs(self.n - nfloat) < 1e-10, "f needs to be of length 2^n for some n > 1"
        assert self.n > 1, "Canalizing strength is only defined for Boolean functions with n > 1 inputs"
        res = []
        if left_side_of_truth_table is None or type(left_side_of_truth_table) != np.ndarray or left_side_of_truth_table.shape[0] != len(self.f):
            left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=self.n)))
        for k in range(1, self.n):
            res.append(self.get_proportion_of_collectively_canalizing_input_sets(k, left_side_of_truth_table=left_side_of_truth_table))
        return np.mean(np.multiply(res, 2**np.arange(1, self.n) / (2**np.arange(1, self.n) - 1))), res
    
    def get_input_redundancy(self):
        """
        Compute the input redundancy of a Boolean function.

        The input redundancy quantifies how many inputs are not required to determine the function’s output.
        Constant functions have an input redundancy of 1 (none of the inputs are needed), whereas parity functions have an input redundancy of 0 (all inputs are necessary).
        
        Returns:
            float: Normalized input redundancy in the interval [0, 1].
        
        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return self.to_cana_BooleanNode().input_redundancy()
        print('The method \'get_input_redundancy\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    
    def get_edge_effectiveness(self):
        """
        Compute the edge effectiveness for each regulator of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.
        
        Returns:
            list: A list of n floats in [0, 1] representing the edge effectiveness for each input.

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return self.to_cana_BooleanNode().edge_effectiveness()
        print('The method \'get_edge_effectiveness\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None

    def get_effective_degree(self):
        """
        Compute the effective degree, i.e., the sum of the edge effectivenesses of each regulator, of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.

        Returns:
            list: A value in [0, 1] representing the effective degree for each input.

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return sum(self.get_edge_effectiveness())
        print('The method \'get_effective_degree\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    
    def to_cana_BooleanNode(self):
        return cana.boolean_node.BooleanNode(k=self.n, outputs=self.f)


    
    


        