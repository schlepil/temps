from coreUtils import *

import polynomial
import relaxations

class propagators:
    def __init__(self):
        self.thisPolyObj_ = None
        self.thisPolyCstr_ = None #We only need one constraint -> the one restraining to the levelset
        self.thisRelax_ = None
        self.thisCstr_ = None
        self.thisCvxProb_ = None
    
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

def getWorstCritPointsPerSubtree(oldResults, oldResultsLin):
    # Search for "root-nodes"
    root2LeafNodesDict = {}
    # oldResults[k][i][j] with k being the time-point -> oldProof last == new proof first
    for aSubProofList in oldResults[0]:
        for aProof in aSubProofList:
            if not np.isfinite(oldResultsLin[aProof['probDict']['resPlacementLin']]):
                # This proof is superseeded with new ones
                continue
            if nall(aProof['probDict']['u'].squeeze() == 2):
                # All linear control -> disregard
                continue
            # Search for root
            _, ip, jp = aProof['probDict']['resPlacementParent']
            while oldResults[0][ip][jp]['probDict']['resPlacementParent'] is not None:
                _, ip, jp = oldResults[0][ip][jp]['probDict']['resPlacementParent']
            # Found the root
            if not (ip, jp) in root2LeafNodesDict.keys():
                root2LeafNodesDict[(ip, jp)] = aProof  # The currently worst solution for this subroot is the first one found
            else:
                # Compare and keep the worst
                if aProof['obj'] < root2LeafNodesDict[(ip, jp)]['obj']:
                    root2LeafNodesDict[(ip, jp)] = aProof
    
    return root2LeafNodesDict

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
            
            # Helper objects
            try:
                assert not (None in (self.thisPolyObj_, self.thisPolyCstr_, self.thisRelax_, self.thisCstr_, self.thisCvxProb))
                assert self.thisPolyObj_.repr is funnelInstance.repr
                thisPolyObj = self.thisPolyObj_
                thisPolyCstr = self.thisPolyCstr_
                thisRelax = self.thisRelax_
                thisCstr = self.thisCstr_
                thisCvxProb = self.thisCvxProb_
            except:
                self.thisPolyObj_ = thisPolyObj = polynomial.polynomial(funnelInstance.repr)
                self.thisPolyCstr_ = thisPolyCstr = polynomial.polynomial(funnelInstance.repr)
                self.thisRelax_ = thisRelax = relaxations.lasserreRelax(funnelInstance.repr)
                self.thisCstr_ = thisCstr = relaxations.lasserreConstraint(thisRelax, thisPolyCstr)
                self.thisCvxProb_ = thisCvxProb = relaxations.convexProg(funnelInstance.repr)
            nq_,nu_ = funnelInstance.dynSys.nq, funnelInstance.dynSys.nu
            
            # Assemble substeps
            tSubSteps = np.hstack([ np.linspace(tSteps[i], tSteps[i+1], num = interStepsPropCrit, endpoint=False, dtype=nfloat) for i in range(tSteps.size-1) ].append(tSteps[-1]))
            
            # Search for "root-nodes"
            root2LeafNodesDict = getWorstCritPointsPerSubtree(oldResults[0],oldResultsLin)
            
            # Now propagate them correctly
            # Loop over timepoints
            # Also use the last timepoint (equivalent to oldResults[0]) as the derivative changes instantaneously
            
            # Build-up the minimize dict
            localSolveDict = {'fun':None, 'constraints':{'type':'ineq', 'fun':None}, 'method':'COBYLA', 'tol':0.9*coreOptions.absTolCstr, 'options':{}}
            for at in reversed(tSubSteps):
                ctrlDict, thisZone = funnelInstance.lyapFunc.getCtrlDict(returnZone=True)
                for aKey,aProof in root2LeafNodesDict:
                    allYCrit = [aProof['critPoints']['yCrit'][:,[i]] for i in range(aProof['critPoints']['yCrit'].shape[1])]
                    allYZCrit = [funnelInstance.repr.evalAllMonoms(ay) for ay in allYCrit]
                    # Loop over each critical point
                    for aYCrit, aYZCrit in zip(allYCrit, allYZCrit):
                        # Objective is with respect to sphere -> rebuild
                        thisPolyObj.coeffs = ctrlDict[-1][0]
                        # Use the control from the proof
                        for i, type in enumerate(aProof['probDict']['u'].squeeze()):
                            thisPolyObj.coeffs += ctrlDict[i][type]
                            if __debug__:
                                if ndot(ctrlDict[i][type].reshape((1,-1)), aYZCrit) > 0.:
                                    print("Control input suboptimal")
                        # Set objective function
                        
                            
                        
                    
            
            
                    
                            
                    
                    
                    
                        
                    
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
        
        