from coreUtils import *

class propagators:
    def __init__(self):
        pass
    
    def doPropagate(*args, **kwargs):
        raise NotImplementedError

# TODO better embed this in the overall structure and allow for parallelization
# this is a "naive" first implementation

class dummyPropagator(propagators):
    def __init__(self):
        super(type(self), self).__init__()
        pass
    
    def doPropagate(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, interStepsPropCrit:int=5):
        return True, oldResults, oldResultsLin
    
    def doRescale(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, interStepsPropCrit:int=5):
        return True, oldResults, oldResultsLin

class localFixedPropagator(propagators):
    def __init__(self):
        super(type(self), self).__init__()
        pass
    
    def doPropagate(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, interStepsPropCrit:int=5):
        """
        \brief: propagate the informations of the last proof. Adapt the actions to the one used by Proofs2Prob
        """
        if self.lyapFunc.opts['zoneCompLvl'] == 1:
            # NOthing to be done here, the information of the last proof is disregarded with this setting
            return super(type(self), self).doPropagate(tSteps, funnelInstance, oldResults, oldResultsLin, interStepsPropCrit)
        elif self.lyapFunc.opts['zoneCompLvl'] ==2:
            # Propagate the critical points of the parents-proofs of all final proofs
            raise NotImplementedError
        elif self.lyapFunc.opts['zoneCompLvl'] == 3:
            # Propagate the worst point of each sub-tree (except the sub-tree with all linear feedback ctrl)
            
            # Search for "root-nodes"
            rootNodesDict = {}
            # oldResults[k][i][j] with k being the time-point -> oldProof last == new proof first
            for aSubProofList in oldResults[0]:
                for aProof in aSubProofList:
                    if not np.isfinite(oldResultsLin[aProof['probDict']['resPlacementLin']]):
                        # This proof is superseeded with new ones
                        continue
                    if nall(aProof['probDict']['u'].squeeze() == 2):
                        # All linear control -> disregard
                        continue
                    #Search for root
                    _,ip,jp = aProof['probDict']['resPlacementParent']
                    while oldResults[0][ip][jp]['probDict']['resPlacementParent'] is not None:
                        _,ip,jp = oldResults[0][ip][jp]['probDict']['resPlacementParent']
                    # Found the root
                    if (ip,jp) in rootNodesDict.keys():
                        
                    else:
                            
                    
                    
                    
                        
                    
        else:
            raise RuntimeError
    
    def doRescale(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, interStepsPropCrit:int=5):
        """
        \brief: propagate the informations of the last proof. Rescale is called when only the size of the funnel changes
        """
        if self.lyapFunc.opts['zoneCompLvl'] == 1:
            # NOthing to be done here, the information of the last proof is disregarded with this setting
            return super(type(self), self).doRescale(tSteps, funnelInstance, oldResults, oldResultsLin, interStepsPropCrit)
        elif self.lyapFunc.opts['zoneCompLvl'] ==2:
            # Rescale the critical points of the parents-proofs of all final proofs
            raise NotImplementedError
        elif self.lyapFunc.opts['zoneCompLvl'] == 3:
            # Propagate the worst point of each sub-tree (except the sub-tree with all linear feedback ctrl)
        
        else:
            raise RuntimeError
        
        