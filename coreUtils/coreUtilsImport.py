import numpy as np
import scipy as sp
import sympy as sy

from numpy import dot as ndot
from numpy.linalg import multi_dot as mndot
from numpy import matmul as nmatmul
from numpy import array as narray
from numpy import matrix as nmatrix
from numpy import require as nrequire
from numpy import identity as nidentity
from numpy import zeros as nzeros
from numpy import ones as nones
from numpy import empty as nempty

from numpy import sum as nsum
from numpy import multiply as nmultiply
from numpy import power as npower
from numpy import prod as nprod
from numpy import einsum as neinsum

from numpy import max as nmax
from numpy import min as nmin
from numpy import abs as nabs

from numpy import maximum as nmaximum
from numpy import minimum as nminimum

from numpy import any as nany
from numpy import all as nall

from numpy import sin as nsin
from numpy import cos as ncos

from numpy import float_ as nfloat
from numpy import int_ as nint
from numpy import int_ as nintu
from numpy import bool_ as nbool

from scipy import sparse

from scipy.linalg import eig,eigh,cholesky,lstsq,eigvals,eigvalsh,inv,det, svd, schur
from scipy.linalg import expm, logm, ldl
from scipy.linalg import norm
from scipy.linalg import solve as ssolve

from typing import List, Tuple, Callable, Union, Iterable

from copy import deepcopy as dp
from copy import copy as cp

from itertools import chain as ichain