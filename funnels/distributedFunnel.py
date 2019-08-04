from coreUtils import *

from parallelChecker import parallelDefinitions as paraDef
from parallelChecker.parallelWorkers import distributor

from dynamicalSystems import dynamicalSystem
import Lyapunov
from Lyapunov import LyapunovFunction
from Lyapunov.lyapPropagators import lyapEvol
from trajectories import referenceTrajectory

from polynomial import polynomialRepr, polynomial
import relaxations as relax

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
                     'numericEpsPos':numericEpsPos,
                     'minFinalValue':1,
                     'earlyExit':True,
                     'minConvRate':-0.,
                     'optsEvol':{
                                    'tDeltaMax':.1
                                },
                     'storeProof':True
                     }
        recursiveExclusiveUpdate(self.opts, opts)

        if isinstance(self.opts['minDistToSep'], str):
            self.opts['minDistToSep'] = eval(self.opts['minDistToSep'])
        
        assert self.opts['sphereBoundCritPoint'] == True #TODO
        
        self.proof_ = {}

    def doProjection(self, zone, criticalPoints:List[dict]=None, ctrlDict:dict=None):
        """
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
                if criticalPoints is not None:
                    # Map the critical points from unit-sphere (solution x) to "real" point (denoted y) 
                    for aCritPoint in criticalPoints:
                        if not 'y' in aCritPoint.keys():
                            aCritPoint['y'] = ndot(Ci, aCritPoint['x'])
            else:
                raise NotImplementedError
        elif self.opts['projection'] in (None, 'None', False):
            pass
        else:
            raise NotImplementedError
        
        return None
    
    def getCtrlDict(self, t, fTaylorApprox, gTaylorApprox, returnZone=True):
        
        ctrlDict = {} # Return value
        
        thisPoly = polynomial(self.repr) #Helper

        allU = [self.dynSys.ctrlInput.getMinU(t), self.traj.getU(t), self.dynSys.ctrlInput.getMaxU(t)]
        uRef = allU[1]
        allDeltaU = [allU[0]-allU[1], allU[2]-allU[1]]
        
        if isinstance(self.lyapFunc, Lyapunov.lyapunovFunctions):
            # Get the zone
            #zone = self.lyapFunc.getPnPdot(t, True)
            # todo
            #zone = [zone[0], 1., zone[1]]
            zone = self.lyapFunc.getZone(t)
            # Optimal control
            objectiveStar = self.lyapFunc.getObjectiveAsArray(fTaylorApprox, gTaylorApprox, self.dynSys.maxTaylorDeg, np.ones((self.dynSys.nu, 1)), self.repr.varNumsPerDeg[0], dx0=self.traj.getDX(t), t=t, P=zone[0], Pdot=zone[2])
        
            # Parse
            ctrlDict[-1] = {0:objectiveStar[0,:]}
            # Add minimal exponential convergence rate
            thisPoly.setQuadraticForm(nidentity(self.dynSys.nq))
            ctrlDict[-1][0] -= thisPoly.coeffs*self.opts['minConvRate']
            
            for k in range(self.dynSys.nu):
                if __debug__:
                    assert abs(objectiveStar[k+1,0]) <= 1e-9
                ctrlDict[k] = { -1:objectiveStar[k+1,:]*allDeltaU[0][k,0], 1:objectiveStar[k+1,:]*allDeltaU[1][k,0] }  # Best is minimal or maximal
                
            # Linear control based on separating hyperplanes
            ctrlDict['PG0'] = ndot(zone[0], gTaylorApprox[0, :, :])
            uCtrlLin, uMonomLin = self.dynSys.ctrlInput.getU(2*np.ones((self.dynSys.nu, ), dtype=nint), 0., P=zone[0], PG0=ctrlDict['PG0'], alpha=zone[1], monomOut=True)
            # Attention here the resulting polynomial coefficients are already scaled correctly (no multiplication with deltaU necessary)
            objectivePolyCtrlLin = self.lyapFunc.getObjectiveAsArray(fTaylorApprox, gTaylorApprox, self.dynSys.maxTaylorDeg, uCtrlLin, uMonomLin, dx0=self.traj.getDX(t), t=t, P=zone[0], Pdot=zone[2])
            
            # Parse
            for k in range(self.dynSys.nu):
                ctrlDict[k][2] = objectivePolyCtrlLin[k+1,:] # set linear
            
            ctrlDict['sphereProj'] = False
        
        else:
            raise NotImplementedError
        
        
        if returnZone:
            return ctrlDict, zone
        else:
            return  ctrlDict

    def ZoneNPoints2Prob(self, zone:"A Lyap Zone", critPList:List[dict], ctrlDict:dict):
        
        nu_ = self.dynSys.nu
        nq_ = self.dynSys.nq
        
        # TODO: Put this in the corresponding Lyapunov function, not here
        if isinstance(self.lyapFunc, Lyapunov.lyapunovFunctions):
            
            if self.opts['sphereBoundCritPoint']:
                # Use optimal control for each critical point (except point is very close to separating hyperplane), inside a sphere (in transformed coords) surrounding it
                # Exclude the largest sphere possible (in transformed space) for the linear control
    
                if not ctrlDict['sphereProj']:
                    raise NotImplementedError
    
                # 1 Create the problems
                probList = []
    
                # 1.1 The linear control approximation
                thisProbBase = {'probDict':{'nPt':-1, 'solver':self.opts['solver'], 'minDist':1., 'scaleFacK':1., 'dimsNDeg':(self.dynSys.nq, self.repr.maxDeg),
                                'nCstrNDegType':[]}, 'cstr':[]}
                thisProbLin = dp(thisProbBase)
                thisProbLin['probDict']['isTerminal']=-1 # Base case, non-convergence is "treatable" via critPoints
                probList.append([thisProbLin])
    
                # 1.1.1 Construct the objective
                thisPoly = polynomial(self.repr) #Helper
                #thisCoeffs = nzeros((self.repr.nMonoms,), dtype=nfloat)
                thisCoeffs = ctrlDict[-1][0].copy() # Objective resulting from system dynamics
                for k in range(nu_):
                    thisCoeffs += ctrlDict[k][2]
                thisProbLin['obj'] = -thisCoeffs # Inverse sign to maximize divergence <-> minimize convergence
                # 1.2 Construct the constraints
                # 1.2.1 Confine to hypersphere
                #thisPoly.setQuadraticForm(-nidentity((nq_), dtype=nfloat), self.repr.varNumsPerDeg[1]) # Attention sign!
                #thisPoly.coeffs[0] = 1.
                thisPoly.setEllipsoidalConstraint(nzeros((2,1), dtype=nfloat), 1.)
                thisProbLin['probDict']['nCstrNDegType'].append( (2,'s') )
                thisProbLin['cstr'].append( thisPoly.coeffs )
                
                #Set information
                thisProbLin['probDict']['u'] = 2*nones((nu_,), dtype=nint)

                PG0n = ctrlDict['PG0']/(norm(ctrlDict['PG0'],axis=0,keepdims=True)+coreOptions.floatEps)
                
                # Now loop through the critical points
                for nPt, aCritPoint in enumerate(critPList):
                    thisProbBase['probDict']['nPt'] = nPt
                    thisY = aCritPoint['y']  # transformed coords of the point
                    # get the distance to all hyperplanes
                    # TODO check sign of PG0 with respect to sign of objectives
                    yPlaneDist = ndot(thisY.T, PG0n).reshape((nu_,))
    
                    if aCritPoint['strictSep'] == 0:
                        # Use the linear approximator to avoid splitting up
                        thisProb = dp(thisProbBase)
                        thisProb['probDict']['isTerminal'] = 0  # Convergence can be improved but only by increasing the computation load
                        thisProb['strictSep'] = aCritPoint['strictSep']
                        probList.append([thisProb]) # Use sublist for each point
    
                        thisCtrlType = nzeros((nu_,1), dtype=nint)#None for _ in range(nu_, 1)]
    
                        # Decide what to do
                        minDist = .9 #np.Inf # Reasonable bound for sphere
                        for i, iDist in enumerate(yPlaneDist):
                            if iDist < -self.opts['minDistToSep']:
                                # Negative -> use maximal control input
                                thisCtrlType[i, 0] = 1
                                minDist = min(minDist, abs(iDist))
                            elif iDist < self.opts['minDistToSep']:
                                # Linear control as iDist is self.opts['minDistToSep'] <= iDist < self.opts['minDistToSep']
                                thisCtrlType[i, 0] = 2
                            else:
                                # Large positive -> use minimal control input
                                thisCtrlType[i, 0] = -1
                        #Remember
                        thisProb['probDict']['u'] = thisCtrlType.copy()
                        thisProb['probDict']['minDist'] = minDist
                        thisProb['probDict']['center'] = thisY.copy()
    
                        # Now we have the necessary information and we can construct the actual problem
                        thisCoeffs = ctrlDict[-1][0].copy()  # Objective resulting from system dynamics
                        for i, type in enumerate(thisCtrlType.reshape((-1,))):
                            if type == 2:
                                # Rescale due to the reduced size
                                if __debug__:
                                    assert abs(ctrlDict[i][type][0]) < 1e-10
                                thisCoeffs[1:] += ctrlDict[i][type][1:]*(1./minDist)
                            else:
                                thisCoeffs += ctrlDict[i][type]
                        thisProb['obj'] = -thisCoeffs # Inverse sign to maximize divergence <-> minimize convergence
                        # get the sphere
                        thisPoly.setEllipsoidalConstraint(thisY, minDist)

                        # Confine the current problem to the sphere
                        thisProb['probDict']['nCstrNDegType'].append((2, 's'))
                        thisProb['cstr'].append(thisPoly.coeffs.copy())
    
                        # Exclude the sphere from linear prob
                        thisProbLin['probDict']['nCstrNDegType'].append((2, 's'))
                        thisProbLin['cstr'].append(-thisPoly.coeffs.copy())
                    else:
                        # Use the separation hyerplane/surface ( degree of the separation polynomial given by aCritPoint['strictSep']
                        # This increases modeling accuracy but at the expense of (exponentially) more optimization problems
                        
                        sepDeg_ = aCritPoint['strictSep']

                        thisCtrlType = np.sign(yPlaneDist).astype(nint)
                        thisCtrlType[thisCtrlType==0] = 1
                        sepList = np.argwhere(nabs(yPlaneDist)<self.opts['minDistToSep'])
                        
                        minDist = nmin(nabs(yPlaneDist[nabs(yPlaneDist)>=self.opts['minDistToSep']]))
                        
                        # Get all possible "combinations"
                        regionDefList = list(itertools.product([-1,1], repeat=len(sepList)))

                        # Create all the problems
                        thisProbBase_ = dp(thisProbBase)
                        thisProbBase_['probDict']['isTerminal'] = aCritPoint['strictSep']
                        thisProbBase_['probDict']['strictSep'] = aCritPoint['strictSep']

                        thisProbBase_['probDict']['minDist'] = minDist
                        thisProbBase_['probDict']['center'] = thisY
                        
                        # Confine the base problem to the hypersphere, exclude the hypersphere form the linear problem
                        # Form the objective for all (discrete) control laws that do not change
                        thisCoeffs = nzeros((self.repr.nMonoms,), dtype=nfloat)
                        thisCoeffs += ctrlDict[-1][0]  # Objective resulting from system dynamics
                        for i, type in enumerate(thisCtrlType.squeeze()):
                            if i in sepList:
                                #Will be treated later on
                                continue
                            thisCoeffs += ctrlDict[i][type]
                        thisProbBase_['obj'] = -thisCoeffs # Inverse sign to maximize divergence <-> minimize convergence

                        # get the sphere
                        thisPoly.setQuadraticForm(-nidentity((nq_), dtype=nfloat), self.repr.varNumsPerDeg[1], 2.*thisY.squeeze(), self.repr.varNumsPerDeg[1])  # Attention sign!
                        thisPoly.coeffs[0] += minDist**2 + mndot([thisY.T, -nidentity((nq_), dtype=nfloat), thisY])
                        
                        # Confine the current problem to the sphere
                        thisProbBase_['probDict']['nCstrNDegType'].append((2, 's'))
                        thisProbBase_['cstr'].append(thisPoly.coeffs.copy())

                        # Exclude the sphere from linear prob
                        thisProbLin['probDict']['nCstrNDegType'].append((2, 's'))
                        thisProbLin['cstr'].append(-thisPoly.coeffs.copy())
                        
                        ## Construction of base problem is done
                        # -> specialise
                        
                        # All problems
                        thisProbList = [ dp(thisProbBase_) for _ in range(len(sepList)) ]
                        
                        # Loop
                        sepMonomNum_ = self.repr.varNumsUpToDeg[sepDeg_]
                        for i, aSepList in enumerate(sepList):
                            thisProb = thisProbList[i]
                            for j, aSep in enumerate(aSepList):
                                # if aSep is 1 -> "positive" side of the separating hyperplane
                                # -> use minimal control input (indexed with -1)
                                # if aSep is -1 -> "negative" side of the separating hyperplane
                                # -> use maximal control input (indexed with 1)

                                thisProb['obj'] -= ctrlDict[sepList[j]][-aSep]  # Inverse sign to maximize divergence <-> minimize convergence
                                # Add the constraint
                                thisCoeffs = nzero((self.repr.nMonoms,), dtype=nfloat)
                                thisCoeffs[:sepMonomNum_] = ctrlDict[sepList[j]][aSep][:sepMonomNum_] #Will be automatically rescaled

                                # Confine the current problem
                                thisProb['probDict']['nCstrNDegType'].append((sepDeg_, 's'))
                                thisProb['cstr'].append(thisCoeffs)
                                
                        #Done
                        probList.append(thisProbList)
            else:
                raise NotImplementedError
                
        else:
            print("Others need to be implemented first")
            raise NotImplementedError
        
        return probList
    
    def analyzeSol(self, thisSol:dict, ctrlDict, critPoints):
        
        """
        Analyze a solution and check if a better approximation can be found
        :param thisSol:
        :return:
        """
        
        if thisSol['origProb']['probDict']['isTerminal'] >= self.opts['minFinalValue']:
            # Thats it, this part of the state-space is proven to be non-stabilizable
            return []
        
        newProbList = self.lyapFunc.analyzeSol(thisSol, ctrlDict, critPoints, opts=self.opts)
        
        for aProb in newProbList:
            aProb['probDict']['resPlacementParent'] = thisSol['origProb']['probDict']['resPlacement']
        
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
            aCtrlDict, aZone = self.getCtrlDict(at, aTaylorApprox[0], aTaylorApprox[1], returnZone=True, opts=self.opts)
            zones.append(aZone)
            # Project the problem if this reduced complexity
            self.doProjection(aZone, aCritPList, aCtrlDict)
            
            aProbList = self.ZoneNPoints2Prob(aZone, aCritPList, aCtrlDict)
            allSolutionsPerGroup.append([None for _ in range(len(aProbList))])
            
            for i, aProb in enumerate(aProbList):
                aProb['probDict']['probId'] = probIdC
                distributor.setProb(aProb)
                probDict[probIdC] = (k,i)
                
                probIdC += 1
        
        #Retrieve solutions and regroup them
        for _ in range(probIdC):
            thisSol = distributor.getSol()
            allIdxSol = probDict[thisSol['probId']]
            allSolutionsPerGroup[allIdxSol[0]][allIdxSol[1]] = thisSol

        return allSolutionsPerGroup
    
    def verify1(self, timePoints, critPoints, allTaylorApprox):
        
        """
        
        :param timePoints:
        :param critPoints:
        :param allTaylorApprox:
        :return:
        """
        
        critPoints = [[] for _ in range(len(timePoints))] if critPoints is None else critPoints
        
        results = []
        resultsLin = -np.Inf*nones((0,), dtype=nfloat)
        resIdLin = 0
        doesConverge = None
        
        # Get all zones to be checked
        allCtrlDictsNzones = [self.lyapFunc.getCtrlDict(at, aTaylorApprox[0], aTaylorApprox[1], returnZone=True, opts=self.opts) for at, aTaylorApprox in zip(timePoints, allTaylorApprox)]
        # Project
        for at, aPointList, aCtrlNZone in zip(timePoints, critPoints, allCtrlDictsNzones):
            self.doProjection(aCtrlNZone[1], aPointList, aCtrlNZone[0])
        
        
        #Set initial problems
        for k, (aCtrlNZone, aCritPList) in enumerate(zip(allCtrlDictsNzones, critPoints)):
            results.append([])
            thisProbList = self.ZoneNPoints2Prob(aCtrlNZone[1], aCritPList, aCtrlNZone[0])
            #results.append([[None for _ in range(len(aSubProbList))] for aSubProbList in thisProbList ])
            for i, aSubProbList in enumerate(thisProbList):
                results[-1].append([])
                #resultsLin.extend( [-np.Inf for _ in range(len(aSubProbList))] )
                resultsLin = np.hstack( [resultsLin, -np.Inf*nones((len(aSubProbList),), dtype=nfloat)] )
                for j, aProb in enumerate(aSubProbList):
                    results[-1][-1].append(None)
                    aProb['probDict']['resPlacementParent'] = None
                    aProb['probDict']['resPlacement'] = (k,i,j)
                    aProb['probDict']['resPlacementLin'] = resIdLin
                    resIdLin += 1
                    distributor.setProb(aProb)

        while nany(resultsLin<self.opts['numericEpsPos']):
            thisSol = distributor.getSol()
            assert thisSol['sol']['status'] == 'optimal' #TODO make compatible with other solvers
            k, i, j = thisSol['probDict']['resPlacement']
            results[k][i][j] = dp(thisSol)
            resultsLin[thisSol['probDict']['resPlacementLin']] = thisSol['sol']['primal objective']

            if __debug__:
                print(f"Checking result for {[k,i,j]}")
                testSol(thisSol, allCtrlDictsNzones[k][0])

            if thisSol['sol']['primal objective']>=self.opts['numericEpsPos']:
                # Proves convergence for the sub region treated
                pass
            else:
                newProbList = self.analyzeSol(thisSol, allCtrlDictsNzones[k][0], critPoints[k])
                if not len(newProbList):
                    # this region is a proof for non-convergence
                    if self.opts['earlyExit']:
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
                    distributor.setProb(aProb)
                    jMax+=1
                    lMax+=1
        
        doesConverge = nall(resultsLin>=self.opts['numericEpsPos'])
        #print('resultsLin',resultsLin)
        #print('self.optsnum',self.opts['numericEpsPos'])

        #print('doseConverge',doesConverge)
        distributor.reset()

        return doesConverge, critPoints, results, resultsLin, timePoints
    
    
    def storeProof(self, doesConverge, critPoints, results, resultsLin, timePoints):
        
        assert doesConverge
        
        for k, at in enumerate(timePoints):
            thisSubProof =  {'t':at, 'critPoints':dp(critPoints[k]), 'sigProbAndVals':[], 'origProb':[]}
            for aSubList in results[k]:
                for aProb in aSubList: # To each crtitical point a list is associated
                    if np.isfinite( resultsLin[results[0][0][0]['probDict']['resPlacementLin']] ):
                        # This is part of the proof
                        thisSubProof['sigProbAndVals'].append( (float(resultsLin[aProb['probDict']['resPlacementLin']]), dp(results[0][0][0])) )
                        thisSubProof['origProb'].append( dp(aProb['origProb']) )
                
            # Save
            if not at in self.proof_.keys():
                self.proof_[at] = []
            self.proof_[at].append( (thisSubProof, results) )
            
    
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

        assert tStart>=self.traj.tLims[0]
        assert tStop<=self.traj.tLims[1]

        tL = tStop
        tC = tStop

        lyapFunc_.reset()
        lyapFunc_.register(tStop, initZone)
        
        criticalPoints = None

        # Back propagate
        while tC > tStart:
            if __debug__:
                print(f"\nAt {tC}\n")
            
            
            tC, nextZoneGuess = self.evolveLyap(tC, self.opts['optsEvol']['tDeltaMax'], lyapFunc_.getLyap(tC))
            tSteps = np.linspace(tL, tC, self.opts['interSteps'])
            
            
            # Step 1 - critical points
            # TODO retropropagate the trajectory of the current crit points for 
            # current guess of the zone. For this only the very last criticalPoints are necessary
            if criticalPoints is not None:
                criticalPoints[1:] = (len(crtitical)-1)*[None]
                criticalPoints, critIsConverging = self.propagator.doPropagate(tSteps,self,criticalPoints,self.opts['interStepsPropCrit'])
            else:
                critIsConverging = True #Dummy to force search

            # Get the current taylor series
            allTaylorApprox = [ self.dynSys.getTaylorApprox(self.traj.getX(aT)) for aT in tSteps ]
            
            # TODO this is too long and should be seperated
            
            lyapFunc_.register(tC, nextZoneGuess)

            # Step 2 check if current is feasible
            if critIsConverging:
                # minConvList = self.solve1(tSteps, criticalPoints, allTaylorApprox)
                isConverging = self.verify1(tSteps, criticalPoints, allTaylorApprox)[0]
            else:
                # We already posses provably non-converging points
                isConverging = False
                    
            
            # TODO use forward propagate critical points !!!
            # TODO really do it
            
            if isConverging:
                # Make the zone bigger to find an upper bound before using dichotomic search
                alphaL = lyapFunc_.getAlpha(0)
                alphaU = 2.*alphaL
                lyapFunc_.setAlpha(alphaU, 0)
                
                while self.verify1(tSteps, criticalPoints, allTaylorApprox)[0]:
                    alphaL = alphaU
                    alphaU *= 2.
                    lyapFunc_.setAlpha(alphaU, 0)

            else:
                # Make the zone smaller to find a lower bound before using dichotomic search
                alphaU = lyapFunc_.getAlpha(0)
                alphaL = alphaU/2.
                lyapFunc_.setAlpha(alphaL, 0)

                while not self.verify1(tSteps, criticalPoints, allTaylorApprox)[0]:
                    alphaU = alphaL
                    alphaL /= 2.
                    lyapFunc_.setAlpha(alphaL, 0)
            
            # Now we can perform dichotomic search
            assert alphaL<alphaU

            while (alphaU-alphaL)>self.opts['convLim']*alphaL:
                alpha = (alphaL+alphaU)/2.
                lyapFunc_.setAlpha(alpha, 0)
                
                if __debug__:
                    print(f"Bisection at {alphaL}, {alpha}, {alphaU}")
                
                # TODO propagate crit points
                # TODO I'm serious here
                if self.verify1(tSteps, criticalPoints, allTaylorApprox)[0]:
                    # Converges
                    alphaL = alpha
                else:
                    alphaU = alpha
            
            # Conservative
            lyapFunc_.setAlpha(alphaL,0)
            # Additional work if seeking to store the proof
            if self.opts['storeProof']:

                self.storeProof(*self.verify1(tSteps, criticalPoints, allTaylorApprox))
            
            tL = tC
            
        return None



