from coreUtils import *

class propagators:
    def __init__(self):
        pass
    
    def doPropagate(*args, **kwargs):
        raise NotImplementedError

# TODO better embed this in the overall structure and allow for parallelization
# this is a "naive" first implementation

class localFixedPropagator(propagators):
    def __init__(self):
        pass
    
    def doPropagate(self,tLower:float, tUpper:float, criticalPointsOld:List, nSteps:int=10):
        """
        \brief: function that computes the trajectory of the old criticalPoints and checks wether they are stabilizable
        """
        
        if 