from coreUtils import *

from parallelChecker import parallelDefinitions as paraDef
#from parallelChecker.parallelWorkers import distributor
import parallelChecker.parallelWorkers

from dynamicalSystems import dynamicalSystem
import Lyapunov
from Lyapunov import LyapunovFunction
from Lyapunov.lyapPropagators import lyapEvol
from trajectories import referenceTrajectory

from polynomial import polynomialRepr, polynomial
import relaxations as relax
relax.propagators.propagators
from funnels.testUtils import testSol

class distributedFunnel:

    def __init__(self, dynSys:dynamicalSystem, lyapFunc:LyapunovFunction, traj:referenceTrajectory, evolveLyap:lyapEvol, propagator:relax.propagators, opts={}):
        self.dynSys = dynSys
        self.lyapFunc = lyapFunc
        self.traj = traj
        self.evolveLyap = evolveLyap
        self.propagator = propagator
        
        self.repr = self.lyapFunc.repr

        self.opts = {'convLim':1e-3, #Dichotomic
                     'minDistToSep':"0.1+0.25/self.dynSys.nu", #When to use linear feedback and "ignore" separation
                     'sphereBoundCritPoint':True, # Whether to use separation or spheric confinement
                     'interSteps':3, # How many points to check per interval
                     'interStepsPropCrit':5, #How many local searches have to be performed
                     'projection':'sphere',
                     'solver':'cvxopt',
                     'numericEpsPos':coreOptions.numericEpsPos,
                     'minFinalValue':1,
                     'earlyExit':True,
                     #'minConvRate':-0., #TODO this does not seem to be propagated correctly # Moved to LyapFunc
                     'optsEvol':{
                                    'tDeltaMax':0.1
                                },
                     'storeProof':True,
                     'useAllAlphas':True
                     }
        recursiveExclusiveUpdate(self.opts, opts)

        if isinstance(self.opts['minDistToSep'], str):
            self.opts['minDistToSep'] = eval(self.opts['minDistToSep'])
        
        assert self.opts['sphereBoundCritPoint'] == True #TODO
        
        self.proof_ = {}
        
        # New to counter profiling issues?
        self.distributor = parallelChecker.parallelWorkers.getDistributor()

    def doProjection(self, zone, aSubProof:List[dict]=None, ctrlDict:dict=None):
        """
        # TODO move this to Lyapunov function?
        Simplifying the problem using some projection.
        Currently only the mapping between ellipsoids and unit-spheres is implemented
        :param zonesToCheck:
        :param objectiveStar:
        :param criticalPoints:
        :return: directly work on input
        """
        
        if self.opts['projection'] == "sphere":
            if isinstance(self.lyapFunc, Lyapunov.lyapunovFunctions):
                
                Ci = cholesky(zone[0]/zone[1])
                Ci_inv = inv(Ci) #aZone=(P_i, alpha_i)
                if ctrlDict is not None:
                    ctrlDict['sphereProj'] = True
                    for aKey, aVal in ctrlDict.items():
                        try:
                            for bKey, bVal in aVal.items():
                                aVal[bKey] = self.repr.doLinCoordChange(bVal, Ci_inv)
                        except:
                            if aKey == 'PG0':
                                # change y = P^{1/2}.x; P = P^{1/2}^T.P^{1/2}
                                # x.T.P.G.x.u -> x.T.P^{1/2}^T.P^{1/2}.G.P^{-1/2}.y.u -> y.T.P^{-1/2}^T."PG0".P^{-1/2}.y.u
                                # P.G = P^{1/2}^T.P^{1/2}.G
                                ctrlDict['PG0'] = ndot(Ci_inv.T, ctrlDict['PG0'])
                            else:
                                print(f"key:{aKey} and val:{aVal} are not affected from projection")
                if aSubProof is not None:
                    #SubProof is [i][j]
                    for aSubProofList in aSubProof:
                        for aProof in aSubProofList:
                            if aProof is None:
                                continue
                            # TODO check if only critical points need to be projected
                            # This is ellip2sphere
                            try:
                                if aProof['critPoints']['yCrit'] is not None:
                                    aProof['critPoints']['xCrit'] = ndot(Ci, aProof['critPoints']['yCrit'])
                            except:
                                print('a')
            else:
                raise NotImplementedError
        elif self.opts['projection'] in (None, 'None', False):
            pass
        else:
            raise NotImplementedError
        
        return None
    
    def doInvProjection(self, zone, aSubProof:List[dict]=None, ctrlDict:dict=None):
        """
        # TODO move this to Lyapunov function?
        :param zone:
        :param result:
        :return:
        """
        if self.opts['projection'] == "sphere":
            if isinstance(self.lyapFunc, Lyapunov.lyapunovFunctions):
            
                Ci = cholesky(zone[0]/zone[1])
                Ci_inv = inv(Ci)  # aZone=(P_i, alpha_i)
                if ctrlDict is not None:
                    ctrlDict['sphereProj'] = True
                    for aKey, aVal in ctrlDict.items():
                        try:
                            for bKey, bVal in aVal.items():
                                aVal[bKey] = self.repr.doLinCoordChange(bVal, Ci)
                        except:
                            if aKey == 'PG0':
                                # change y = P^{1/2}.x; P = P^{1/2}^T.P^{1/2}
                                # x.T.P.G.x.u -> x.T.P^{1/2}^T.P^{1/2}.G.P^{-1/2}.y.u -> y.T.P^{-1/2}^T."PG0".P^{-1/2}.y.u
                                # P.G = P^{1/2}^T.P^{1/2}.G
                                ctrlDict['PG0'] = ndot(Ci.T, ctrlDict['PG0'])
                            else:
                                print(f"key:{aKey} and val:{aVal} are not affected from projection")
                if aSubProof is not None:
                    #SubProof is [i][j]
                    for aSubProofList in aSubProof:
                        for aProof in aSubProofList:
                            if aProof is None:
                                continue
                            # TODO check if only critical points need to be projected
                            # This is sphere2ellip
                            try:
                                if aProof['critPoints']['xCrit'] is not None:
                                    aProof['critPoints']['yCrit'] = ndot(Ci_inv, aProof['critPoints']['xCrit'])
                            except:
                                print('a')
    # OLD moved lyapfun
    # def getCtrlDict(self, t, fTaylorApprox, gTaylorApprox, returnZone=True):
    #
    #     ctrlDict = {} # Return value
    #
    #     thisPoly = polynomial(self.repr) #Helper
    #
    #     allU = [self.dynSys.ctrlInput.getMinU(t), self.traj.getU(t), self.dynSys.ctrlInput.getMaxU(t)]
    #     uRef = allU[1]
    #     allDeltaU = [allU[0]-allU[1], allU[2]-allU[1]]
    #
    #     if isinstance(self.lyapFunc, Lyapunov.lyapunovFunctions):
    #         raise NotImplementedError
    #         # Get the zone
    #         #zone = self.lyapFunc.getPnPdot(t, True)
    #         # todo
    #         #zone = [zone[0], 1., zone[1]]
    #         zone = self.lyapFunc.getZone(t)
    #         # Optimal control
    #         objectiveStar = self.lyapFunc.getObjectiveAsArray(fTaylorApprox, gTaylorApprox, self.dynSys.maxTaylorDeg, np.ones((self.dynSys.nu, 1)), self.repr.varNumsPerDeg[0], dx0=self.traj.getDX(t), t=t, P=zone[0], Pdot=zone[2])
    #
    #         # Parse
    #         ctrlDict[-1] = {0:objectiveStar[0,:]}
    #         # Add minimal exponential convergence rate
    #         thisPoly.setQuadraticForm(nidentity(self.dynSys.nq))
    #         ctrlDict[-1][0] -= thisPoly.coeffs*self.opts['minConvRate'] # TODO check if only polynomial coefficients generated by this function are used
    #
    #         for k in range(self.dynSys.nu):
    #             if __debug__:
    #                 assert abs(objectiveStar[k+1,0]) <= 1e-9
    #             ctrlDict[k] = { -1:objectiveStar[k+1,:]*allDeltaU[0][k,0], 1:objectiveStar[k+1,:]*allDeltaU[1][k,0] }  # Best is minimal or maximal
    #
    #         # Linear control based on separating hyperplanes
    #         ctrlDict['PG0'] = ndot(zone[0], gTaylorApprox[0, :, :])
    #         uCtrlLin, uMonomLin = self.dynSys.ctrlInput.getU(2*np.ones((self.dynSys.nu, ), dtype=nint), 0., P=zone[0], PG0=ctrlDict['PG0'], alpha=zone[1], monomOut=True)
    #         # Attention here the resulting polynomial coefficients are already scaled correctly (no multiplication with deltaU necessary)
    #         objectivePolyCtrlLin = self.lyapFunc.getObjectiveAsArray(fTaylorApprox, gTaylorApprox, self.dynSys.maxTaylorDeg, uCtrlLin, uMonomLin, dx0=self.traj.getDX(t), t=t, P=zone[0], Pdot=zone[2])
    #
    #         # Parse
    #         for k in range(self.dynSys.nu):
    #             ctrlDict[k][2] = objectivePolyCtrlLin[k+1,:] # set linear
    #
    #         ctrlDict['sphereProj'] = False
    #
    #     else:
    #         raise NotImplementedError
    #
    #
    #     if returnZone:
    #         return ctrlDict, zone
    #     else:
    #         return  ctrlDict

    # def ZoneNPoints2Prob(self, zone:"A Lyap Zone", critPList:List[dict], ctrlDict:dict):
    #
    #     raise RuntimeError("Deprecated")
    #
    #     nu_ = self.dynSys.nu
    #     nq_ = self.dynSys.nq
    #
    #     # TODO: Put this in the corresponding Lyapunov function, not here
    #     if isinstance(self.lyapFunc, Lyapunov.lyapunovFunctions):
    #
    #         if self.opts['sphereBoundCritPoint']:
    #             # Use optimal control for each critical point (except point is very close to separating hyperplane), inside a sphere (in transformed coords) surrounding it
    #             # Exclude the largest sphere possible (in transformed space) for the linear control
    #
    #             if not ctrlDict['sphereProj']:
    #                 raise NotImplementedError
    #
    #             # 1 Create the problems
    #             probList = []
    #
    #             # 1.1 The linear control approximation
    #             thisProbBase = {'probDict':{'nPt':-1, 'solver':self.opts['solver'], 'minDist':1., 'scaleFacK':1., 'dimsNDeg':(self.dynSys.nq, self.repr.maxDeg),
    #                             'nCstrNDegType':[]}, 'cstr':[]}
    #             thisProbLin = dp(thisProbBase)
    #             thisProbLin['probDict']['isTerminal']=-1 # Base case, non-convergence is "treatable" via critPoints
    #             probList.append([thisProbLin])
    #
    #             # 1.1.1 Construct the objective
    #             thisPoly = polynomial(self.repr) #Helper
    #             #thisCoeffs = nzeros((self.repr.nMonoms,), dtype=nfloat)
    #             thisCoeffs = ctrlDict[-1][0].copy() # Objective resulting from system dynamics
    #             for k in range(nu_):
    #                 thisCoeffs += ctrlDict[k][2]
    #             thisProbLin['obj'] = -thisCoeffs # Inverse sign to maximize divergence <-> minimize convergence
    #             # 1.2 Construct the constraints
    #             # 1.2.1 Confine to hypersphere
    #             #thisPoly.setQuadraticForm(-nidentity((nq_), dtype=nfloat), self.repr.varNumsPerDeg[1]) # Attention sign!
    #             #thisPoly.coeffs[0] = 1.
    #             thisPoly.setEllipsoidalConstraint(nzeros((2,1), dtype=nfloat), 1.)
    #             thisProbLin['probDict']['nCstrNDegType'].append( (2,'s') )
    #             thisProbLin['cstr'].append( thisPoly.coeffs )
    #
    #             #Set information
    #             thisProbLin['probDict']['u'] = 2*nones((nu_,), dtype=nint)
    #
    #             PG0n = ctrlDict['PG0']/(norm(ctrlDict['PG0'],axis=0,keepdims=True)+coreOptions.floatEps)
    #
    #             # Now loop through the critical points
    #             for nPt, aCritPoint in enumerate(critPList):
    #                 thisProbBase['probDict']['nPt'] = nPt
    #                 thisY = aCritPoint['y']  # transformed coords of the point
    #                 # get the distance to all hyperplanes
    #                 # TODO check sign of PG0 with respect to sign of objectives
    #                 yPlaneDist = ndot(thisY.T, PG0n).reshape((nu_,))
    #
    #                 if aCritPoint['strictSep'] == 0:
    #                     # Use the linear approximator to avoid splitting up
    #                     thisProb = dp(thisProbBase)
    #                     thisProb['probDict']['isTerminal'] = 0  # Convergence can be improved but only by increasing the computation load
    #                     thisProb['strictSep'] = aCritPoint['strictSep']
    #                     probList.append([thisProb]) # Use sublist for each point
    #
    #                     thisCtrlType = nzeros((nu_,1), dtype=nint)#None for _ in range(nu_, 1)]
    #
    #                     # Decide what to do
    #                     minDist = .9 #np.Inf # Reasonable bound for sphere
    #                     for i, iDist in enumerate(yPlaneDist):
    #                         if iDist < -self.opts['minDistToSep']:
    #                             # Negative -> use maximal control input
    #                             thisCtrlType[i, 0] = 1
    #                             minDist = min(minDist, abs(iDist))
    #                         elif iDist < self.opts['minDistToSep']:
    #                             # Linear control as iDist is self.opts['minDistToSep'] <= iDist < self.opts['minDistToSep']
    #                             thisCtrlType[i, 0] = 2
    #                         else:
    #                             # Large positive -> use minimal control input
    #                             thisCtrlType[i, 0] = -1
    #                     #Remember
    #                     thisProb['probDict']['u'] = thisCtrlType.copy()
    #                     thisProb['probDict']['minDist'] = minDist
    #                     thisProb['probDict']['center'] = thisY.copy()
    #
    #                     # Now we have the necessary information and we can construct the actual problem
    #                     thisCoeffs = ctrlDict[-1][0].copy()  # Objective resulting from system dynamics
    #                     for i, type in enumerate(thisCtrlType.reshape((-1,))):
    #                         if type == 2:
    #                             # Rescale due to the reduced size
    #                             if __debug__:
    #                                 assert abs(ctrlDict[i][type][0]) < 1e-10
    #                             thisCoeffs[1:] += ctrlDict[i][type][1:]*(1./minDist)
    #                         else:
    #                             thisCoeffs += ctrlDict[i][type]
    #                     thisProb['obj'] = -thisCoeffs # Inverse sign to maximize divergence <-> minimize convergence
    #                     # get the sphere
    #                     thisPoly.setEllipsoidalConstraint(thisY, minDist)
    #
    #                     # Confine the current problem to the sphere
    #                     thisProb['probDict']['nCstrNDegType'].append((2, 's'))
    #                     thisProb['cstr'].append(thisPoly.coeffs.copy())
    #
    #                     # Exclude the sphere from linear prob
    #                     thisProbLin['probDict']['nCstrNDegType'].append((2, 's'))
    #                     thisProbLin['cstr'].append(-thisPoly.coeffs.copy())
    #                 else:
    #                     # Use the separation hyerplane/surface ( degree of the separation polynomial given by aCritPoint['strictSep']
    #                     # This increases modeling accuracy but at the expense of (exponentially) more optimization problems
    #
    #                     sepDeg_ = aCritPoint['strictSep']
    #
    #                     thisCtrlType = np.sign(yPlaneDist).astype(nint)
    #                     thisCtrlType[thisCtrlType==0] = 1
    #                     sepList = np.argwhere(nabs(yPlaneDist)<self.opts['minDistToSep'])
    #
    #                     minDist = nmin(nabs(yPlaneDist[nabs(yPlaneDist)>=self.opts['minDistToSep']]))
    #
    #                     # Get all possible "combinations"
    #                     regionDefList = list(itertools.product([-1,1], repeat=len(sepList)))
    #
    #                     # Create all the problems
    #                     thisProbBase_ = dp(thisProbBase)
    #                     thisProbBase_['probDict']['isTerminal'] = aCritPoint['strictSep']
    #                     thisProbBase_['probDict']['strictSep'] = aCritPoint['strictSep']
    #
    #                     thisProbBase_['probDict']['minDist'] = minDist
    #                     thisProbBase_['probDict']['center'] = thisY
    #
    #                     # Confine the base problem to the hypersphere, exclude the hypersphere form the linear problem
    #                     # Form the objective for all (discrete) control laws that do not change
    #                     thisCoeffs = nzeros((self.repr.nMonoms,), dtype=nfloat)
    #                     thisCoeffs += ctrlDict[-1][0]  # Objective resulting from system dynamics
    #                     for i, type in enumerate(thisCtrlType.squeeze()):
    #                         if i in sepList:
    #                             #Will be treated later on
    #                             continue
    #                         thisCoeffs += ctrlDict[i][type]
    #                     thisProbBase_['obj'] = -thisCoeffs # Inverse sign to maximize divergence <-> minimize convergence
    #
    #                     # get the sphere
    #                     thisPoly.setQuadraticForm(-nidentity((nq_), dtype=nfloat), self.repr.varNumsPerDeg[1], 2.*thisY.squeeze(), self.repr.varNumsPerDeg[1])  # Attention sign!
    #                     thisPoly.coeffs[0] += minDist**2 + mndot([thisY.T, -nidentity((nq_), dtype=nfloat), thisY])
    #
    #                     # Confine the current problem to the sphere
    #                     thisProbBase_['probDict']['nCstrNDegType'].append((2, 's'))
    #                     thisProbBase_['cstr'].append(thisPoly.coeffs.copy())
    #
    #                     # Exclude the sphere from linear prob
    #                     thisProbLin['probDict']['nCstrNDegType'].append((2, 's'))
    #                     thisProbLin['cstr'].append(-thisPoly.coeffs.copy())
    #
    #                     ## Construction of base problem is done
    #                     # -> specialise
    #
    #                     # All problems
    #                     thisProbList = [ dp(thisProbBase_) for _ in range(len(sepList)) ]
    #
    #                     # Loop
    #                     sepMonomNum_ = self.repr.varNumsUpToDeg[sepDeg_]
    #                     for i, aSepList in enumerate(sepList):
    #                         thisProb = thisProbList[i]
    #                         for j, aSep in enumerate(aSepList):
    #                             # if aSep is 1 -> "positive" side of the separating hyperplane
    #                             # -> use minimal control input (indexed with -1)
    #                             # if aSep is -1 -> "negative" side of the separating hyperplane
    #                             # -> use maximal control input (indexed with 1)
    #
    #                             thisProb['obj'] -= ctrlDict[sepList[j]][-aSep]  # Inverse sign to maximize divergence <-> minimize convergence
    #                             # Add the constraint
    #                             thisCoeffs = nzero((self.repr.nMonoms,), dtype=nfloat)
    #                             thisCoeffs[:sepMonomNum_] = ctrlDict[sepList[j]][aSep][:sepMonomNum_] #Will be automatically rescaled
    #
    #                             # Confine the current problem
    #                             thisProb['probDict']['nCstrNDegType'].append((sepDeg_, 's'))
    #                             thisProb['cstr'].append(thisCoeffs)
    #
    #                     #Done
    #                     probList.append(thisProbList)
    #         else:
    #             raise NotImplementedError
    #
    #     else:
    #         print("Others need to be implemented first")
    #         raise NotImplementedError
    #
    #     return probList
    
    def analyzeSol(self, thisSol:dict, ctrlDict):
        
        """
        Analyze a solution and check if a better approximation can be found
        :param thisSol:
        :return:
        """
        
        if thisSol['origProb']['probDict']['isTerminal'] >= self.opts['minFinalValue']:
            # Thats it, this part of the state-space is proven to be non-stabilizable
            # Correctly: to contain at least one non-stabilizable point
            return []

        resPlacementParent = dp(thisSol['origProb']['probDict']['resPlacement'])

        newProbList = self.lyapFunc.analyzeSol(thisSol, ctrlDict, opts=self.opts)
        
        for aProb in newProbList:
            aProb['probDict']['resPlacementParent'] = resPlacementParent
        
        return newProbList
        
    def solve1(self, timePoints, critPoints, allTaylorApprox):
        """
        Checks the feasibility of one concrete problem
        :param zonesToCheck:
        :param critPoints:
        :param objStar:
        :param deltaU:
        :param routingDict:
        :param probQueus:
        :param solQueues:
        :return:
        """
        
        allSolutionsPerGroup = []
        probDict = { 'workerRouting':{} }
        probIdC = 0
        
        # The corresponding zones
        zones = [None for _ in range(len(critPoints))]
        
        # Set problems
        for k, (at, aCritPList, aTaylorApprox) in enumerate(zip(timePoints, critPoints, allTaylorApprox)):
    
            # Get the control dict
            aCtrlDict, aZone = self.getCtrlDict(at, aTaylorApprox[0], aTaylorApprox[1], returnZone=True)
            zones.append(aZone)
            # Project the problem if this reduced complexity
            self.doProjection(aZone, aCritPList, aCtrlDict)
            
            aProbList = self.ZoneNPoints2Prob(aZone, aCritPList, aCtrlDict)
            allSolutionsPerGroup.append([None for _ in range(len(aProbList))])
            
            for i, aProb in enumerate(aProbList):
                aProb['probDict']['probId'] = probIdC
                self.distributor.setProb(aProb)
                probDict[probIdC] = (k,i)
                
                probIdC += 1
        
        #Retrieve solutions and regroup them
        for _ in range(probIdC):
            thisSol = self.distributor.getSol()
            allIdxSol = probDict[thisSol['probId']]
            allSolutionsPerGroup[allIdxSol[0]][allIdxSol[1]] = thisSol

        return allSolutionsPerGroup
    
    def verify1(self, timePoints, lastProofInfo, allTaylorApprox):
        
        """
        
        :param timePoints:
        :param lastProofInfo: (results, resultsLin)
        :param allTaylorApprox:
        :return:
        """
        
        results = []
        resultsLin = -np.Inf*nones((0,), dtype=nfloat)
        resIdLin = 0
        doesConverge = None
        
        if lastProofInfo is None:
            oldResultsLin = oldResults=len(timePoints)*[None]
        else:
            oldResults = lastProofInfo[0]
            if oldResults is None:
                oldResults = len(timePoints)*[None]
            oldResultsLin = lastProofInfo[1]
        
        
        # Get all zones to be checked
        allCtrlDictsNzones = [self.lyapFunc.getCtrlDict(at, aTaylorApprox[0], aTaylorApprox[1], returnZone=True, opts=self.opts) for at, aTaylorApprox in zip(timePoints, allTaylorApprox)]
        # Project
        for at, aSubProof, aCtrlNZone in zip(timePoints, oldResults, allCtrlDictsNzones):
            self.doProjection(aCtrlNZone[1], aSubProof, aCtrlNZone[0])
        
        
        #Set initial problems
        for k, (at, aCtrlNZone, aSubProof) in enumerate(zip(timePoints, allCtrlDictsNzones, oldResults)):
            results.append([])
            thisProbList = self.lyapFunc.Proofs2Prob(at, aCtrlNZone[1], oldResultsLin, aSubProof, aCtrlNZone[0], self.opts)
            #results.append([[None for _ in range(len(aSubProbList))] for aSubProbList in thisProbList ])
            for i, aSubProbList in enumerate(thisProbList):
                assert i==0, "TODO"
                results[-1].append([])
                #resultsLin.extend( [-np.Inf for _ in range(len(aSubProbList))] )
                resultsLin = np.hstack( [resultsLin, -np.Inf*nones((len(aSubProbList),), dtype=nfloat)] ) #Attention set to -inf (total divergence,
                # if not yet proven
                for j, aProb in enumerate(aSubProbList):
                    results[-1][-1].append(None)
                    # New structure coming from Proof2Prob:
                    # If there is more than one problem in the list, then all remaining problemns are the child of the first
                    aProb['probDict']['resPlacementParent'] = None if ((i==0) and (j==0)) else (k,i,0)
                    aProb['probDict']['resPlacement'] = (k,i,j)
                    aProb['probDict']['resPlacementLin'] = resIdLin
                    resIdLin += 1
                    if __debug__:
                        if nany(aProb['probDict']['u'] != 2) and ((aProb['probDict']['resPlacementParent'] is None) or not (len(aProb['probDict']['resPlacementParent']) == 3)):
                            print("snap")
                        if np.allclose(aProb['cstr'][0], 0):
                            print("a")
                        
                    self.distributor.setProb(aProb)
                    

        while nany(resultsLin<self.opts['numericEpsPos']):
            thisSol = self.distributor.getSol()
            assert thisSol['sol']['status'] == 'optimal' #TODO make compatible with other solvers
            # Store the critical points within the result
            # TODO improve structure -> Works only if projection is done in main program not by the worker!
            thisSol['critPoints'] = {'xCrit':thisSol['xSol'].copy(), 'currU':thisSol['probDict']['u'].copy()}
            # Store the proof
            k, i, j = thisSol['probDict']['resPlacement']
            results[k][i][j] = dp(thisSol)
            resultsLin[thisSol['probDict']['resPlacementLin']] = thisSol['obj']
            if __debug__:
                print(f"Checking result for {[k,i,j]}")
                testSol(thisSol, allCtrlDictsNzones[k][0]) # TODO sth is wrong here
                if nany(thisSol['probDict']['u'] != 2) and (thisSol['probDict']['resPlacementParent'] is None):
                    print("snap")

            if thisSol['obj']>=self.opts['numericEpsPos']:
                # Proves convergence for the sub region treated
                pass
            else:
                newProbList = self.analyzeSol(thisSol, allCtrlDictsNzones[k][0])
                if (not len(newProbList)) and self.opts['earlyExit']:
                    # this region is a proof for non-convergence
                    break
                # New (less conservative) problems could be derived
                # "Accept" this solution and place the new problems
                resultsLin[thisSol['probDict']['resPlacementLin']] = np.Inf
                jMax = len(results[k][i])
                lMax = len(resultsLin)
                #resultsLin.extend( [None for _ in range(len(newProbList))] )
                resultsLin = np.hstack([resultsLin, -np.Inf*nones((len(newProbList),), dtype=nfloat)])
                #results[k][i].extend( [None for _ in range(len(newProbList))] )

                for aProb in newProbList:
                    results[k][i].append(None)
                    aProb['probDict']['resPlacement'] = (k,i,jMax)
                    aProb['probDict']['resPlacementLin'] = lMax
                    if np.allclose(aProb['cstr'][0], 0):
                        print("a")
                    self.distributor.setProb(aProb)
                    jMax+=1
                    lMax+=1
        
        # Inverse projection to get actual critical points
        # TODO check if this is the best place to do this and if the structure is sufficiently generic
        for at, aSubProof, aCtrlNZone in zip(timePoints, results, allCtrlDictsNzones):
            self.doInvProjection(aCtrlNZone[1], aSubProof, None)
        
        doesConverge = nall(resultsLin>=self.opts['numericEpsPos'])
        #print('resultsLin',resultsLin)
        #print('self.optsnum',self.opts['numericEpsPos'])

        #print('doseConverge',doesConverge)
        self.distributor.reset()

        return doesConverge, results, resultsLin, timePoints
    
    
    def storeProof(self, doesConverge, results, resultsLin, timePoints):
        
        assert doesConverge
        
        for k, at in enumerate(timePoints):
            thisSubProof =  {'t':at, 'proofs':[], 'allResults':dp(results)}
            for aSubList in results[k]:
                for aProb in aSubList: # To each crtitical point a list is associated
                    if np.isfinite( resultsLin[aProb['probDict']['resPlacementLin']]):
                        # This is part of the proof
                        thisSubProof['proofs'].append( dp(aProb) ) # TODO dp?
                
            # Save
            if not at in self.proof_.keys():
                self.proof_[at] = []
            self.proof_[at].append( thisSubProof )
    
        return None
            
        

    def compute(self, tStart:float, tStop:float, initZone):
        
        """
        Backpropagates the funnel
        :param routingDict:
        :param probQueues:
        :param solQueues:
        :param tStart:
        :param tStop:
        :param initZone:
        :return:
        """

        lyapFunc_ = self.lyapFunc
        storeProof__ = self.opts['storeProof']

        assert tStart>=self.traj.tLims[0]
        assert tStop<=self.traj.tLims[1]

        tL = tStop
        tC = tStop

        lyapFunc_.reset()
        lyapFunc_.register(tStop, initZone)
        
        results = None
        resultsLin = None
        resultsProp = [[[]] for _ in range(self.opts['interSteps'])] # Propagated result structure do not possess the tree structure
        resultsLinProp = []

        # Back propagate
        while tC > tStart:
            if __debug__:
                print(f"\nAt {tC}\n")
            
            # Countering some numerical issues
            if ((tC - self.opts['optsEvol']['tDeltaMax']) > 0.) and (tC - self.opts['optsEvol']['tDeltaMax'])*1e4 < self.opts['optsEvol']['tDeltaMax']:
                # The remaining time is very very small -> enlarge maximal step
                deltaTNumeric = self.opts['optsEvol']['tDeltaMax']
            else:
                deltaTNumeric = 0.
            tC, nextZoneGuess = self.evolveLyap(tC, min(self.opts['optsEvol']['tDeltaMax'] + deltaTNumeric, tC-tStart), lyapFunc_.getLyap(tC))
            tSteps = np.linspace(tC, tL, self.opts['interSteps']) # TODO check order; Does the order play a role here?
            
            # Step 1 - critical points
            # TODO retropropagate the trajectory of the current crit points for 
            # current guess of the zone. For this only the very last criticalPoints are necessary
            if results is not None:
                critIsConverging, resultsProp, resultsLinProp = self.propagator.doPropagate(tSteps, self, results, resultsLin, self.opts['interStepsPropCrit'])
                critIsConverging = nall(critIsConverging) # TODO sth smarter
            else:
                critIsConverging = True #Dummy to force search

            # Get the current taylor series
            allTaylorApprox = [ self.dynSys.getTaylorApprox(self.traj.getX(aT)) for aT in tSteps ]
            
            # TODO this is too long and should be separated
            
            lyapFunc_.register(tC, nextZoneGuess)

            # Step 2 check if current is feasible
            if critIsConverging:
                # minConvList = self.solve1(tSteps, criticalPoints, allTaylorApprox)
                isConverging, results, resultsLin, timePoints = self.verify1(tSteps, (resultsProp, resultsLinProp), allTaylorApprox) #Change to store all in order to exploit the last proof
            else:
                # We already posses provably non-converging points
                isConverging = False

            # Last positive proof
            proofDataLargest = None

            if isConverging:
                if storeProof__:
                    proofDataLargest = dp((isConverging, results, resultsLin, timePoints)) #Store last positive #TODO avoid dp
                # Make the zone bigger to find an upper bound before using dichotomic search
                alphaL = lyapFunc_.getAlpha(0)
                alphaU = 2.*alphaL
                alphaFromTo = lyapFunc_.setAlpha(alphaU, 0, returnInfo=True)
                
                #while self.verify1(tSteps, criticalPoints, allTaylorApprox)[0]:
                while True:
                    (critIsConverging, allAlphas), resultsProp, resultsLinProp = self.propagator.doRescale(tSteps, self, results, resultsLin, allTaylorApprox, alphaFromTo, self.opts['interStepsPropCrit'])
                    # First critical point is "original" so the convergence should be ensured
                    assert critIsConverging[0], "Something is wrong with the local solve proof"

                    # Heuristic: Use the smallest non-converging (with respect to critical points
                    if ((self.opts['useAllAlphas']) and (critIsConverging[-1]==False)) :
                        idx = np.flatnonzero(critIsConverging)[-1] + 1  # Last converging + 1 = First non-converging
                        alphaU = allAlphas[idx]
                        lyapFunc_.setAlpha(alphaU, 0, returnInfo=False)
                        critIsConverging = critIsConverging[idx]
                        if __debug__:
                            assert critIsConverging == False, "idx got it wrong"
                            # Test
                            isConverging, results, resultsLin, timePoints = self.verify1(tSteps, (resultsProp, resultsLinProp),
                                                                                         allTaylorApprox)  # Change to store all in order to exploit the last proof
                            assert isConverging == False, "Local proof contradicts global proof"
                    else:
                        critIsConverging = critIsConverging[-1] # Simply use last

                    if critIsConverging:
                        try:
                            isConverging, results, resultsLin, timePoints = self.verify1(tSteps, (resultsProp, resultsLinProp), allTaylorApprox) #Change to store all in order to exploit the last proof
                        except:
                            isConverging, results, resultsLin, timePoints = self.verify1(tSteps, (resultsProp, resultsLinProp),
                                                                                         allTaylorApprox)  # Change to store all in order to exploit the last proof
                        if isConverging and storeProof__:
                            proofDataLargest = dp((isConverging, results, resultsLin, timePoints))  # Store last positive
                    else:
                        # The propagations of critical points found non-stabilizable points
                        isConverging = False
                    if not isConverging:
                        # Break if first not converging size is found
                        break
                    alphaL = alphaU
                    alphaU *= 2.
                    alphaFromTo = lyapFunc_.setAlpha(alphaU, 0, returnInfo=True)

            else:
                # Make the zone smaller to find a lower bound before using dichotomic search
                alphaU = lyapFunc_.getAlpha(0)
                alphaL = alphaU/2.
                alphaFromTo = lyapFunc_.setAlpha(alphaL, 0, returnInfo=True)

                #while not self.verify1(tSteps, criticalPoints, allTaylorApprox)[0]:
                while True:
                    (critIsConverging, allAlphas), resultsProp, resultsLinProp = self.propagator.doRescale(tSteps, self, results, resultsLin, allTaylorApprox, alphaFromTo, self.opts['interStepsPropCrit'])
                    # First critical point is "original" so the DI-convergence should be ensured
                    assert critIsConverging[0] == False, "Something is wrong with the local solve proof"

                    # Heuristic: Use the largest converging (with respect to critical points)
                    if ((self.opts['useAllAlphas']) and (critIsConverging[-1])) :
                        idx = np.flatnonzero(critIsConverging)[0]  # First converging -> largest with respect to crit points
                        alphaL = allAlphas(idx)
                        lyapFunc_.setAlpha(alphaL, 0, returnInfo=False)
                        critIsConverging = critIsConverging[idx]
                        if __debug__:
                            assert critIsConverginge, "idx got it wrong"
                            # Test
                            isConverging, results, resultsLin, timePoints = self.verify1(tSteps, (resultsProp, resultsLinProp),
                                                                                         allTaylorApprox)  # Change to store all in order to exploit the last proof
                            if isConverging != critIsConverging:
                                raise UserWarning("Local proof gives different convergence than global proof -> Heuristic suboptimal")
                    else:
                        critIsConverging = critIsConverging[-1] # Simply use last -> Here smallest

                    if critIsConverging:
                        isConverging, results, resultsLin, timePoints = self.verify1(tSteps, (resultsProp, resultsLinProp), allTaylorApprox) #Change to store all in order to exploit the last proof
                        if isConverging and storeProof__:
                            proofDataLargest = dp((isConverging, results, resultsLin, timePoints))  # Store last positive
                    else:
                        isConverging = False
                    if isConverging:
                        # Break at first converging size
                        break
                    alphaU = alphaL
                    alphaL /= 2.
                    alphaFromTo = lyapFunc_.setAlpha(alphaL, 0, returnInfo=True)
            
            # Now we can perform dichotomic search
            assert alphaL<alphaU
            
            while (alphaU-alphaL)>self.opts['convLim']*alphaL:
                alpha = (alphaL+alphaU)/2.
                alphaFromTo = lyapFunc_.setAlpha(alpha, 0, returnInfo=True)
                
                if __debug__:
                    print(f"Bisection at {alphaL}, {alpha}, {alphaU}")
                
                #if self.verify1(tSteps, criticalPoints, allTaylorApprox)[0]:
                (critIsConverging, allAlphas), resultsProp, resultsLinProp = self.propagator.doRescale(tSteps, self, results, resultsLin, allTaylorApprox, alphaFromTo, self.opts['interStepsPropCrit'])

                if ((self.opts['useAllAlphas']) and (critIsConverging[0]!=critIsConverging[-1])): #Can only be used if there is a zero-crossing
                    idxChange = np.flatnonzero(critIsConverging[:-1] != critIsConverging[1:]) #Get the index
                    # Use the non-converging side to avoid global opt
                    if critIsConverging[idxChange]:
                        idxChange += 1
                    alpha = allAlphas[idxChange]
                    critIsConverging = critIsConverging[idxChange]
                    assert critIsConverging == False, "idx got sth wrong"
                else:
                    critIsConverging = critIsConverging[-1]


                if critIsConverging:
                    isConverging, results, resultsLin, timePoints = self.verify1(tSteps, (resultsProp, resultsLinProp), allTaylorApprox) #Change to store all in order to exploit the last proof
                else:
                    isConverging = False
                if isConverging:
                    # Converges
                    alphaL = alpha
                    if storeProof__:
                        proofDataLargest = dp((isConverging, results, resultsLin, timePoints))
                else:
                    alphaU = alpha
            
            if proofDataLargest is None:
                raise RuntimeError("Never found a proof for convergence")
            
            # Conservative -> Choose the largest converging value found
            lyapFunc_.setAlpha(alphaL,0)
            # Additional work if seeking to store the proof
            if storeProof__:
                # Necessary to ensure that the proof for alphaL is stored
                self.storeProof(*proofDataLargest)
            
            tL = tC
            
        return None
    

# TODO: also propagate critical points of linear control