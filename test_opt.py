import numpy as np
from scipy.sparse import coo_matrix
import polynomial as poly
import plotting as plot
from random import random
import relaxations as relax
from relaxations.lasserre import *
from relaxations.constraints import *
from relaxations.optUtils import *
from relaxations.rref import robustRREF
from scipy.optimize import minimize as sp_minimize
from scipy.optimize import NonlinearConstraint
from copy import copy, deepcopy


coeff=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
coeff2=np.array([1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0])
coeff3=np.array([1.,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
coeff4=np.array([1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0])
coeff5=np.array([1,0,1,0,0,0,0,0,0,0,0,0,0,0,0])

