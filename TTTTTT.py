from coreUtils import *
from systems.pendulum import getSys, getUlims
import polynomial as poly
import dynamicalSystems as dynSys
import trajectories as traj
import Lyapunov as lyap
import relaxations as relax

import plotting as plot
from plotting import plt

from scipy.integrate import solve_ivp
shape=(2,1)
shape_ = narray(shape, ndmin=1)
f = np.ndarray(shape_, dtype=object)
print(shape_)
print(f)