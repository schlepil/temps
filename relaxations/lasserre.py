from coreUtils import *
from polynomial import *

# Implements base relaxations and constraints

#We will only look at the upper triangular matrix as they are all symmetric

class lasserreRelax:
    def __init__(self):