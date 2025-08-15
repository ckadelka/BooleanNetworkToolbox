#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jul 29 09:25:40 2025

@author: Claus Kadelka
"""


##Imports
import numpy as np
import itertools

def bin2dec(binary_vector):
    """
    Convert a binary vector to an integer.

    Parameters:
        binary_vector (list): List containing binary digits (0 or 1).

    Returns:
        int: Integer value converted from the binary vector.
    """
    decimal = 0
    for bit in binary_vector:
        decimal = (decimal << 1) | bit
    return int(decimal)


def dec2bin(integer_value, num_bits):
    """
    Convert an integer to a binary vector.

    Parameters:
        integer_value (int): Integer value to be converted.
        num_bits (int): Number of bits in the binary representation.

    Returns:
        list: List containing binary digits (0 or 1).
    """
    binary_string = bin(integer_value)[2:].zfill(num_bits)
    return [int(bit) for bit in binary_string]


def bool_to_poly(f, left_side_of_truth_table=None, variables=None, prefix=''):
    """
    Transform a Boolean function from truth table format to polynomial format in non-reduced DNF.

    Parameters:
        f (list): Boolean function as a vector (list of length 2^n, where n is the number of inputs).
        left_side_of_truth_table (list, optional): The left-hand side of the Boolean truth table 
            (a list of tuples of size 2^n x n). If provided, it speeds up computation.
        indices (list, optional): List of indices to use for variable naming. If empty or not matching 
            the required number, defaults to list(range(n)).
        prefix (str, optional): Prefix for variable names in the polynomial, default ''.

    Returns:
        str: A string representing the Boolean function in disjunctive normal form (DNF).
    """
    len_f = len(f)
    n = int(np.log2(len_f))
    if variables is None or len(variables) != n:
        prefix = 'x'
        variables = [prefix+str(i) for i in range(n)]
    if left_side_of_truth_table is None:  # to reduce run time, this should be calculated once and then passed as argument
        left_side_of_truth_table = list(itertools.product([0, 1], repeat=n))
    num_values = 2 ** n
    text = []
    for i in range(num_values):
        if f[i] == True:
            monomial = ' * '.join([('%s' % (v)) if entry == 1 else ('(1 - %s)' % (v)) 
                                  for v, entry in zip(variables, left_side_of_truth_table[i])])
            text.append(monomial)
    if text != []:
        return ' + '.join(text)
    else:
        return '0'


def check_if_empty(my_list):
    """
    Check if the provided list or NumPy array is empty.

    Parameters:
        my_list (list or np.ndarray): The list or array to check.

    Returns:
        bool: True if my_list is empty (or has size 0 for a NumPy array), False otherwise.
    """
    if isinstance(my_list, np.ndarray):
        if my_list.size == 0:
            return True
    elif my_list == []:
        return True
    else:
        return False


def f_from_expression(expr):
    """
    Extract a Boolean function from a string expression.

    The function converts an input expression into its truth table representation.
    The expression can include Boolean operators and comparisons, and the order of variables
    is determined by their first occurrence in the expression.

    Parameters:
        expr (str): A text string containing an evaluable Boolean expression.
            Examples:
                'A AND NOT B'
                'x1 + x2 + x3 > 1'
                '(x1 + x2 + x3) % 2 == 0'

    Returns:
        tuple:
            - f (list): The right-hand side of the Boolean function (truth table) as a list of length 2**n,
              where n is the number of inputs.
            - var (list): A list of variable names (of length n) in the order they were encountered.
    
    Examples:
        >>> f_from_expression('A AND NOT B')
        ([0, 0, 1, 0], ['A', 'B'])
        
        >>> f_from_expression('x1 + x2 + x3 > 1')
        ([0, 0, 0, 1, 0, 1, 1, 1], ['x1', 'x2', 'x3'])
        
        >>> f_from_expression('(x1 + x2 + x3) % 2 == 0')
        ([1, 0, 0, 1, 0, 1, 1, 0], ['x1', 'x2', 'x3'])
    """
    expr = expr.replace('(', ' ( ').replace(')', ' ) ').replace('!','not ').replace('~','not ')
    expr_split = expr.split(' ')
    var = []
    dict_var = dict()
    n_var = 0
    for i, el in enumerate(expr_split):
        if el not in ['',' ','(',')','and','or','not','AND','OR','NOT','&','|','+','-','*','%','>','>=','==','<=','<',] and not el.isdigit():
            try:
                new_var = dict_var[el]
            except KeyError:
                new_var = 'x[%i]' % n_var
                dict_var.update({el: new_var})
                var.append(el)
                n_var += 1
            expr_split[i] = new_var
        elif el in ['AND','OR','NOT']:
            expr_split[i] = el.lower()
        elif el == '&':
            expr_split[i] = 'and'
        elif el == '|':
            expr_split[i] = 'or'
    expr = ' '.join(expr_split)
    f = []
    for x in itertools.product([0, 1], repeat=n_var):
        x = list(map(bool, x))
        f.append(int(eval(expr)))  # x_val is used implicitly in the eval context
    return f, var


def flatten(l):
    return [item for sublist in l for item in sublist]