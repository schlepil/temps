from numba import jit, njit, prange

import numpy as np
import scipy as sp
import sympy as sy

from numpy import dot as ndot
from numpy.linalg import multi_dot as mndot
from numpy import array as narray
from numpy import matrix as nmatrix
from numpy import require as nrequire
from numpy import identity as nidentity
from numpy import zeros as nzeros
from numpy import ones as nones
from numpy import empty as nempty

from numpy import float_ as nfloat
from numpy import int_ as nint
from numpy import int_ as nintu

from scipy import sparse

from scipy.linalg import eig,eigh,cholesky,lstsq,eigvals,eigvalsh,inv

from typing import List, Tuple


from copy import deepcopy as dp
from copy import copy as cp


lmap = lambda f,l: list(map(f,l))

class variableStruct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)