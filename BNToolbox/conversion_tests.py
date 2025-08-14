#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:16:55 2025

@author: ckadelka
"""

import generate
import boolean_function
import boolean_network
import numpy as np

n=np.random.randint(1,9)
bf = generate.random_function(n)
bf_converted_to_cana = bf.to_cana_BooleanNode()
bf_reconverted = boolean_function.from_cana_BooleanNode(bf_converted_to_cana)
assert np.all(bf.f == bf_reconverted.f), 'failed from_cana_BooleanNode or to_cana_BooleanNode'

#TODO: add conversion test to/from pyboolnet

