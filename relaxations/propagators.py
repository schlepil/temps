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

def getWorstCritPointsPerSubtree(oldResults, oldResultsLin, delNonCrit=False):
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
    
    if delNonCrit:
        # Remove 'xCrit' and 'yCrit' form critPoints of not the one chosen for the subtree
        idList = [id(aProof) for aProof in root2LeafNodesDict.values()]
        for aSubProofList in oldResults[0]:
            for aProof in aSubProofList:
                if not id(aProof) in idList:
                    aProof['critPoints']['xCrit'] = None
                    aProof['critPoints']['yCrit'] = None
        
    
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
            # TODO check if necessary and how much time the deep copying takes
            oldResults = dp(oldResults)
            oldResultsLin = dp(oldResultsLin)
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
            root2LeafNodesDict = getWorstCritPointsPerSubtree(oldResults[0], oldResultsLin, True)
            
            # Now propagate them correctly
            # Loop over timepoints
            # Also use the last timepoint (equivalent to oldResults[0]) as the derivative changes instantaneously
            
            # Build-up the minimize dict
            localSolveDict = {'fun':None, 'x0':None, 'constraints':{'type':'ineq', 'fun':None}, 'method':'COBYLA', 'tol':0.9*coreOptions.absTolCstr, 'options':{}}
            for at in reversed(tSubSteps):
                ctrlDict, thisZone = funnelInstance.lyapFunc.getCtrlDict(returnZone=True)
                for aKey,aProof in root2LeafNodesDict:
                    allYCrit = [aProof['critPoints']['yCrit'][:,[i]] for i in range(aProof['critPoints']['yCrit'].shape[1])]
                    allYZCrit = [funnelInstance.repr.evalAllMonoms(ay) for ay in allYCrit]
                    # Get the constraints as a polynomial function
                    thisPolyFunCstr = polynomial.polyFunction(funnelInstance.repr, (nsum(nabs(aProof['probDict']['u']) == 1)+1,)) # Last constraint is to confine to levelset; others are to restrain to this optimal region
                    # TODO influence of linear and polynomial seperation?
                    # Loop over each critical point
                    for aYCrit, aYZCrit in zip(allYCrit, allYZCrit):
                        # Objective is with respect to sphere -> rebuild
                        thisPolyObj.coeffs = ctrlDict[-1][0].copy()
                        # Use the control from the proof (to build up objective and constraints)
                        cstrCount = 0
                        for i, atype in enumerate(aProof['probDict']['u'].squeeze()):
                            thisPolyObj.coeffs += ctrlDict[i][atype]
                            if atype in (-1,1):
                                # -> separation necessary
                                thisPolyObj[cstrCount,1] = polynomial.polynomial(funnelInstance.repr, -ctrlDict[i][atype])
                                cstrCount += 1
                            if __debug__:
                                if ndot(-ctrlDict[i][atype].reshape((1,-1)), aYZCrit) > 0.:
                                    print("Control input suboptimal")
                        # Set objective function
                        # But first inverse sign for min
                        thisPolyObj.coeffs = -thisPolyObj.coeffs
                        localSolveDict['fun'] = lambda x: thisPolyObj.eval2(x.reshape((nq_,1)))
                        
                        # Get the constraint to the level-set
                        # zone2Cstr(self,aZone, offset:np.ndarray=None)
                        thisPolyFunCstr[-1,1] = funnelInstance.lyapFunc.zone2Cstr(thisZone)
                        # Get the function
                        localSolveDict['constraints']['fun'] = lambda x: thisPolyFunCstr.eval2(x.reshape((nq_,1))).squeeze()
                        # Set the start point
                        localSolveDict['x0'] = aYCrit.reshape((nq_,))
                        
                        # Here we go
                        res = relaxations.localSolve(**localSolveDict)
                        assert res.succes
                        # Set the new value
                        aProof['critPoints']['yCrit'] = res.x.reshape((nq_, 1))
                        aProof['obj'] = res.fun
                        # Check if converging
                        isDiverging = False
                        if res.fun >= coreOptions.absTolCstr:
                            # Check if stabilizable (using the best possible input)
                            aYZCrit = funnelInstance.repr.evalAllMonoms(aProof['critPoints']['yCrit'])
                            thisPolyObj.coeffs = ctrlDict[-1][0].copy()
                            for i in range(nu_):
                                atype = -(int(ndot(ctrlDict[i][1].reshape((1,-1)), aYZCrit)>=0.)*2-1)
                                thisPolyObj.coeffs += ctrlDict[i][atype]
                            # Check
                            aProof['obj'] = thisPolyObj.eval2(aYZCrit)
                            isDiverging = ( aProof['obj'] >= coreOptions.absTolCstr )
                        oldResultsLin[aProof['probDict']['resPlacementLin']] = aProof['obj']
                        
                        if isDiverging:
                            # -> Found a proof for non-stabilizability
                            return False, oldResults, oldResultsLin
                            
                    

    













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
        
        