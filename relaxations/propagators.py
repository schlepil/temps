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
    
    def doPropagate(self, *args, **kwargs):
        raise NotImplementedError
    
    def doRescale(self, *args, **kwargs):
        raise NotImplementedError

# TODO better embed this in the overall structure and allow for parallelization
# this is a "naive" first implementation

class dummyPropagator(propagators):
    def __init__(self):
        super(type(self), self).__init__()
        pass
    
    def doPropagate(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, interStepsPropCrit:int=5):
        return True, oldResults, oldResultsLin
    
    def doRescale(self, tSteps, funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, allTaylorApprox, alphaFromTo, interStepsPropCrit:int=5):
        return ([True],[1]), oldResults, oldResultsLin

def getWorstCritPointsPerSubtree(aSubProof, oldResultsLin, delNonCrit=False):
    # Search for "root-nodes"
    root2LeafNodesDict = {}
    # oldResults[k][i][j] with k being the time-point -> oldProof last == new proof first
    for aSubProofList in aSubProof:
        for aProof in aSubProofList:
            if aProof is None:
                continue
            if not np.isfinite(oldResultsLin[aProof['probDict']['resPlacementLin']]):
                # This proof is superseeded with new ones
                continue
            if nall(aProof['probDict']['u'].reshape((-1,)) == 2):
                # All linear control -> disregard
                continue
            # Search for root
            _, ip, jp = aProof['probDict']['resPlacementParent']
            while aSubProof[ip][jp]['probDict']['resPlacementParent'] is not None:
                _, ip, jp = aSubProof[ip][jp]['probDict']['resPlacementParent']
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
        for aSubProofList in aSubProof:
            for aProof in aSubProofList:
                if aProof is None:
                    continue
                if not id(aProof) in idList:
                    aProof['critPoints']['xCrit'] = None
                    aProof['critPoints']['yCrit'] = None
        
    
    return root2LeafNodesDict

class localFixedPropagator(propagators):
    def __init__(self):
        super(type(self), self).__init__()
        pass
    
    def fillLocalSolveDict(self, localSolveDict:dict, x0:np.ndarray, funnelInstance, aProof:dict, aZone, ctrlDict:dict):
        nq_ = x0.size

        thisPolyFunCstr = polynomial.polyFunction(funnelInstance.repr, (nsum(nabs(aProof['probDict']['u']) == 1)+1,))  # Last constraint is to confine to levelset; others are to restrain to this optimal region
        
        thisPolyObj = self.thisPolyObj_
        # Objective is with respect to sphere -> rebuild
        thisPolyObj.coeffs = ctrlDict[-1][0].copy()
        # Use the control from the proof (to build up objective and constraints)
        cstrCount = 0
        thisPolyObj.unlock()
        for i, atype in enumerate(aProof['probDict']['u'].reshape((-1,))):
            thisPolyObj.coeffs += ctrlDict[i][atype]
            if atype in (-1, 1):
                # -> separation necessary
                thisPolyFunCstr[cstrCount] = polynomial.polynomial(funnelInstance.repr, -ctrlDict[i][atype])
                cstrCount += 1
        
        # Set objective function
        # But first inverse sign for min
        thisPolyObj.coeffs = -thisPolyObj.coeffs
        localSolveDict['fun'] = lambda x:thisPolyObj.eval2(x.reshape((nq_, 1)))
    
        # Get the constraint to the level-set
        # zone2Cstr(self,aZone, offset:np.ndarray=None)
        thisPolyFunCstr[-1] = funnelInstance.lyapFunc.zone2Cstr(aZone)
        # Get the function
        localSolveDict['constraints']['fun'] = lambda x:thisPolyFunCstr.eval2(x.reshape((nq_, 1))).squeeze()
        # Set the start point
        localSolveDict['x0'] = x0.reshape((nq_,))
        
        if __debug__:
            if nany(thisPolyFunCstr.eval2(x0) < coreOptions.absTolCstr):
                print("Constraints failed on starting points")
        
        return None
    
    def postProcLocalSol(self, res, aProof, ctrlDict, repr, oldResultsLin ):
        
        nu_ = aProof['probDict']['u'].size
        
        thisPolyObj = self.thisPolyObj_
        
        isDiverging = False
        
        assert res.succes
        # Set the new value
        aProof['critPoints']['yCrit'] = res.x.reshape((-1, 1))
        aProof['obj'] = res.fun
        if res.fun >= coreOptions.absTolCstr:
            # Check if stabilizable (using the best possible input)
            res.z = repr.evalAllMonoms(aProof['critPoints']['yCrit'])
            thisPolyObj.coeffs = ctrlDict[-1][0].copy()
            for i in range(nu_):
                atype = -(int(ndot(ctrlDict[i][1].reshape((1, -1)), res.z) >= 0.)*2-1)
                thisPolyObj.coeffs += ctrlDict[i][atype]
            # Check
            aProof['obj'] = thisPolyObj.eval2(res.z)
            oldResultsLin[aProof['probDict']['resPlacementLin']] = aProof['obj']
            # Found proof of non-stabilizability
            isDiverging = (aProof['obj'] >= coreOptions.absTolCstr)
        
        return isDiverging
    
    def lvl3Propagate(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, interStepsPropCrit:int=5):
        
        newResults = [[[]] for _ in range(len(tSteps))]
        newResultsLin = []
        
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
            self.thisPolyObj_ = thisPolyObj = polynomial.polynomial(funnelInstance.repr); self.thisPolyObj_.unlock() # Always "unsafe"
            self.thisPolyCstr_ = thisPolyCstr = polynomial.polynomial(funnelInstance.repr)
            self.thisRelax_ = thisRelax = relaxations.lasserreRelax(funnelInstance.repr)
            self.thisCstr_ = thisCstr = relaxations.lasserreConstraint(thisRelax, thisPolyCstr)
            self.thisCvxProb_ = thisCvxProb = relaxations.convexProg(funnelInstance.repr)
        nq_, nu_ = funnelInstance.dynSys.nq, funnelInstance.dynSys.nu
    
        # Assemble substeps
        tSubSteps = np.hstack([np.linspace(tSteps[i], tSteps[i+1], num=interStepsPropCrit, endpoint=False, dtype=nfloat) for i in range(tSteps.size-1)].append(tSteps[-1]))
    
        # Search for "root-nodes"
        root2LeafNodesDict = getWorstCritPointsPerSubtree(oldResults[0], oldResultsLin, True)
    
        # Now propagate them correctly
        # Loop over timepoints
        # Also use the last timepoint (equivalent to oldResults[0]) as the derivative changes instantaneously
    
        idxResLinNew = 0
    
        # Build-up the minimize dict
        localSolveDict = {'fun':None, 'x0':None, 'constraints':{'type':'ineq', 'fun':None}, 'method':'COBYLA', 'tol':0.9*coreOptions.absTolCstr, 'options':{}}
        # Check if converging
        isDiverging = False
        for at in reversed(tSubSteps):
            if isDiverging:
                break

            ctrlDict = thisZone = None
            for aKey, aProof in root2LeafNodesDict:
                # Lazy eval
                if ctrlDict is None:
                    ctrlDict, thisZone = funnelInstance.lyapFunc.getCtrlDict(at, returnZone=True)
                allYCrit = [aProof['critPoints']['yCrit'][:, [i]] for i in range(aProof['critPoints']['yCrit'].shape[1])]
                for aYCrit in allYCrit:
                    # TODO influence of linear and polynomial seperation?
                    # Loop over each critical point
                    # Fill it
                    self.fillLocalSolveDict(localSolveDict, aYCrit, funnelInstance, aProof, thisZone, ctrlDict)
                    # Here we go
                    res = relaxations.localSolve(**localSolveDict)
                    # Post-Proc
                    isDiverging &= self.postProcLocalSol(res, aProof, ctrlDict, funnelInstance.repr, oldResultsLin)
            
            # Build up new dict
            if at in tSteps:
                idxK = tSteps.size
                idxI = 0
                idxJ = 0
                for aSubProof in reversed(newResults):
                    idxK -= 1
                    for aSubProofList in aSubProof:
                        if not len(aSubProofList):
                            # Copy all necessary proofs
                            aSubProofList.extend(dp(root2LeafNodesDict.values()))
                        for aProof in aSubProofList:
                            aProof['probDict']['resPlacement'] = [idxK, idxI, idxJ]
                            aProof['probDict']['resPlacementLin'] = idxResLinNew
                            aProof['probDict']['resPlacementParent'] = None
                            newResultsLin.append(aProof['obj'])
                            idxJ += 1
                            idxResLinNew += 1
    
        return isDiverging, newResults, newResultsLin
    
    def doPropagate(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, interStepsPropCrit:int=5):
        """
        \brief: propagate the informations of the last proof. Adapt the actions to the one used by Proofs2Prob
        """
        
        if funnelInstance.lyapFunc.opts_['zoneCompLvl'] == 1:
            # NOthing to be done here, the information of the last proof is disregarded with this setting
            return super(type(self), self).doPropagate(tSteps, funnelInstance, oldResults, oldResultsLin, interStepsPropCrit)
        elif funnelInstance.lyapFunc.opts_['zoneCompLvl'] ==2:
            # Propagate the critical points of the parents-proofs of all final proofs
            raise NotImplementedError
        elif funnelInstance.lyapFunc.opts_['zoneCompLvl'] == 3:
            return self.lvl3Propagate(tSteps,funnelInstance, oldResults, oldResultsLin, interStepsPropCrit)
        else:
            raise RuntimeError

    def lvl3Rescale(self, tSteps, funnelInstance: "funnels.distributedFunnel", oldResults, oldResultsLin, allTaylorApprox, alphaFromTo, interStepsPropCrit: int = 5):
        """
        \brief : Rescale
        :param tSteps:
        :param funnelInstance:
        :param oldResults:
        :param oldResultsLin:
        :param interStepsPropCrit:
        :return:
        """
        
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
            self.thisPolyObj_ = thisPolyObj = polynomial.polynomial(funnelInstance.repr); self.thisPolyObj_.unlock() #Always "unsafe"
            self.thisPolyCstr_ = thisPolyCstr = polynomial.polynomial(funnelInstance.repr)
            self.thisRelax_ = thisRelax = relaxations.lasserreRelax(funnelInstance.repr)
            self.thisCstr_ = thisCstr = relaxations.lasserreConstraint(thisRelax, thisPolyCstr)
            self.thisCvxProb_ = thisCvxProb = relaxations.convexProg(funnelInstance.repr)
        
        # Get all critical leaves
        root2LeafNodesList = [ getWorstCritPointsPerSubtree(aSubProofList, oldResultsLin, delNonCrit=True) for aSubProofList in oldResults ]
        
        # Create all alpha steps
        allAlphas = np.linspace(alphaFromTo[0], alphaFromTo[1], interStepsPropCrit, endpoint=True)

        # Build-up the minimize dict
        localSolveDict = {'fun':None, 'x0':None, 'constraints':{'type':'ineq', 'fun':None}, 'method':'COBYLA', 'tol':0.9*coreOptions.absTolCstr, 'options':{}}
        # Check if converging
        isDiverging = [False for _ in range(interStepsPropCrit)]

        # Loop through them
        for i, aAlpha in enumerate(allAlphas):
            # -> Set this alpha
            funnelInstance.lyapFunc.setAlpha(aAlpha, 0)
            
            # Lazy eval
            ctrlDict = thisZone = None
            #Loop through timepoints
            for j, (at, root2LeafDict) in enumerate(zip(tSteps, root2LeafNodesList)):
                # Get zone and control dict
                # def getCtrlDict(self, t:float, fTaylorApprox=None, gTaylorApprox=None,returnZone=True, taylorDeg=None, maxCtrlDeg=2, opts={}):
                
                # Loop through the critical points
                for aProof in root2LeafDict.values():
                    if ctrlDict is None:
                        # Only compute this if necessary
                        ctrlDict, thisZone = funnelInstance.lyapFunc.getCtrlDict(at, fTaylorApprox=allTaylorApprox[j][0], gTaylorApprox=allTaylorApprox[j][1], returnZone=True)
                    
                    allYCrit = [aProof['critPoints']['yCrit'][:, [i]] for i in range(aProof['critPoints']['yCrit'].shape[1])]
                    for aYCrit in allYCrit:
                        # TODO influence of linear and polynomial seperation?
                        # Loop over each critical point
                        # Fill it
                        self.fillLocalSolveDict(localSolveDict, aYCrit, funnelInstance, aProof, thisZone, ctrlDict)
                        # Here we go
                        res = relaxations.localSolve(**localSolveDict)
                        # Post-Proc
                        isDiverging[i] &= self.postProcLocalSol(res, aProof, ctrlDict, funnelInstance.repr, oldResultsLin)
        
        newResultsLin = []
        newResults = [[[]] for _ in tSteps]
        idxResLinNew = 0
        for idxK, root2LeafDict in enumerate(root2LeafNodesList):
            idxI = 0
            idxJ = 0
            
            for aSubProofList in newResults[idxK]:
                aSubProofList.extend(root2LeafDict.values())
                for aProof in aSubProofList:
                    aProof['probDict']['resPlacement'] = [idxK, idxI, idxJ]
                    aProof['probDict']['resPlacementLin'] = idxResLinNew
                    aProof['probDict']['resPlacementParent'] = None
                    newResultsLin.append(aProof['obj'])
                    idxJ += 1
                    idxResLinNew += 1
        
        return ([not a for a in isDiverging], allAlphas), newResults, newResultsLin #TODO change this; Main prog expects is converging
        
    
    def doRescale(self,tSteps,funnelInstance:"funnels.distributedFunnel", oldResults, oldResultsLin, allTaylorApprox, alphaFromTo, interStepsPropCrit:int=5):
        """
        \brief: propagate the informations of the last proof. Rescale is called when only the size of the funnel changes
        """
        
        if funnelInstance.lyapFunc.opts_['zoneCompLvl'] == 1:
            # NOthing to be done here, the information of the last proof is disregarded with this setting
            return super(type(self), self).doRescale(tSteps, funnelInstance, oldResults, oldResultsLin, alphaFromTo, interStepsPropCrit)
        elif funnelInstance.lyapFunc.opts_['zoneCompLvl'] ==2:
            # Rescale the critical points of the parents-proofs of all final proofs
            raise NotImplementedError
        elif funnelInstance.lyapFunc.opts_['zoneCompLvl'] == 3:
            # Propagate the worst point of each sub-tree (except the sub-tree with all linear feedback ctrl)
            return self.lvl3Rescale(tSteps, funnelInstance, oldResults, oldResultsLin, allTaylorApprox, alphaFromTo, interStepsPropCrit)
        else:
            raise RuntimeError
        
        