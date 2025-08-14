## Utils
from .utils import bin2dec
from .utils import dec2bin
from .utils import bool_to_poly
from .utils import check_if_empty
from .utils import f_from_expression
from .utils import flatten

# TODO: now that most of this functionality is stored in the classes
# how much of these are we going to keep?
from .analyze_BF import *
from .analyze_BN import *

## Generation Utils
from .generate import random_adjacency_matrix
from .generate import random_edge_list
## Generate Random BF
from .generate import random_function
from .generate import random_linear_function
from .generate import random_non_degenerated_function
from .generate import random_degenerated_function
from .generate import random_non_canalizing_function
from .generate import random_non_canalizing_non_degenerated_function
from .generate import random_k_canalizing_function
from .generate import random_k_canalizing_function_with_specific_layer_structure
from .generate import random_nested_canalizing_function
## Generate Random BN
from .generate import random_network
# TODO: should this really be in the generate module?
from .generate import get_layer_structure_of_an_NCF_given_its_Hamming_weight

## Boolean Function
from .boolean_function import BooleanFunction
from .boolean_function import get_layer_structure_from_can_outputs

## Boolean Network
from .boolean_network import BooleanNetwork

## CANA Compatability
from .boolean_function import from_cana_BooleanNode as bf_from_cana_BooleanNode
from .boolean_network import from_cana_BooleanNetwork as bn_from_cana_BooleanNetwork

## pybooleannet Compatability
from .boolean_function import from_pybooleannet_xxxxx as bf_from_pybooleannet_xxxxx
from .boolean_network import from_pybooleannet_xxxxx as bn_from_pybooleannet_xxxxx