#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:16:55 2025

@author: ckadelka
"""

import sys
sys.path.append('../BNToolbox/')

import generate
import boolean_function
import boolean_network
import numpy as np


#Generate a random Boolean function, turn it into a cana object and back and ensure it is the same function
n=np.random.randint(1,9)
bf = generate.random_function(n)
bf_converted_to_cana = bf.to_cana_BooleanNode()
bf_reconverted = boolean_function.cana_BooleanNode_to_BooleanFunction(bf_converted_to_cana)
assert np.all(bf.f == bf_reconverted.f), 'failed cana_BooleanNode_to_BooleanFunction or to_cana_BooleanNode'

#TODO: add conversion test to/from cana_BooleanNetwork


#Generate a random Boolean network, turn it into a pyboolnet bnet and back and ensure it is the same network
N = np.random.randint(3,20)
n = np.random.randint(1,min(N,8))
bn = generate.random_network(N,n)
bnet = bn.to_pyboolnet_bnet()
bn_reconverted = boolean_network.pyboolnet_bnet_to_BooleanNetwork(bnet)
assert (np.all([np.all(bn.F[i].f == bn_reconverted.F[i].f) for i in range(N)]) and
        np.all([np.all(bn.I[i] == bn_reconverted.I[i]) for i in range(N)]) and 
        np.all(bn.variables == bn_reconverted.variables)), 'failed pyboolnet_bnet_to_BooleanNetwork or to_pyboolnet_bnet'
