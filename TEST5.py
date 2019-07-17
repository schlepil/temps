import numpy as np
import polynomial as poly
import plotting as plot
import relaxations as relax
from relaxations.lasserre import *
from relaxations.constraints import *
from relaxations.optUtils import *
from relaxations.rref import robustRREF
coeff=np.array([[1,18,1,1,12,1,1],[1,1,1,1,1,1,1]])
print(coeff[0,3:])