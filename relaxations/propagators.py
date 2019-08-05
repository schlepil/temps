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
        super(type(self), self).__init__()
        pass
    
    def doPropagate(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults,oldResultsLin, interStepsPropCrit:int=5):
        """
        \brief: propagate the critical point of each indidual proof
        """
        pass