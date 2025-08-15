#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:16:55 2025

@author: ckadelka
"""

import sys
sys.path.append('../BNToolbox/')

import numpy as np
import boolean_function

import generate
import utils


n=np.random.randint(2,9)
probability_one = np.random.random()

#Check that a non-canalizing function with exact canalizing depth k==0 is generated
bf = generate.random_non_canalizing_function(n,probability_one=probability_one)
assert bf.get_layer_structure()['CanalizingDepth']==0,"generate.random_non_canalizing_function failed"


#Check that a non-canalizing non-degenrated function with exact canalizing depth k==0 is generated
#All variables in non-degenerated functions have an edge effectiveness > 0
bf = generate.random_non_canalizing_non_degenerated_function(n,probability_one=probability_one)
assert bf.get_layer_structure()['CanalizingDepth']==0 and min(bf.get_edge_effectiveness())>0,"generate.random_non_canalizing_non_degenerated_function failed"




n=np.random.randint(1,9)
n=1
k=np.random.randint(0,n)
if k==n-1:#Boolean functions with exact canalizing depth k==n-1 do not exist
    k+=1


#Linear functions (XOR-type functions) must have normalized average sensitivity == 1
bf = generate.random_linear_function(n)
assert bf.get_average_sensitivity(EXACT=True)==1,"generate.random_linear_function or boolean_function.get_average_sensitivity(EXACT=True) failed"


#All variables in non-degenerated functions have an edge effectiveness > 0
bf = generate.random_non_degenerated_function(n,probability_one=probability_one)
assert min(bf.get_edge_effectiveness())>0,"generate.random_non_degenerated_function failed"


#At least one variable in a degenerated function must have an edge effectiveness == 0
bf = generate.random_degenerated_function(n,probability_one=probability_one)
assert min(bf.get_edge_effectiveness())==0,"generate.random_degenerated_function failed"


#Check that a function with exact canalizing depth k is generated
bf = generate.random_k_canalizing_function(n, k, EXACT_DEPTH=True)
assert bf.get_layer_structure()['CanalizingDepth']==k,"generate.random_k_canalizing failed"


#Check that a function with minimal canalizing depth k is generated
bf = generate.random_k_canalizing_function(n, k, EXACT_DEPTH=False)
assert bf.get_layer_structure()['CanalizingDepth']>=k,"generate.random_k_canalizing failed"


#All variables in an NCF are conditionally canalizing
bf = generate.random_nested_canalizing_function(n)
assert bf.is_k_canalizing(n),"generate.random_NCF failed"


#Generate all possible layer structures of n-input NCFs and test if the correct layer structure is recovered
for w in range(1,2**(n-1),2):
    layer_structure = utils.get_layer_structure_of_an_NCF_given_its_Hamming_weight(n,w)[-1]
    bf = generate.random_nested_canalizing_function(n,layer_structure=layer_structure)
    test = np.all(np.array(boolean_function.get_layer_structure_from_can_outputs(bf.get_layer_structure()['CanalizingOutputs'])) == np.array(layer_structure))
    assert test,"generate.random_NCF failed for n = {n} and layer_structure = {layer_structure}"


#TODO: Add thorough testing and examples for generate.random_network








