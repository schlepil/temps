from coreUtils import *

from parallelChecker import parallelDefinitions as paraDef
from parallelChecker import probSetter

from dynamicalSystems import dynamicalSystem
from Lyapunov import LyapunovFunction, lyapEvol
from Lyapunov import lyapunovFunctions
from trajectories import referenceTrajectory

from polynomial import polynomialRepr, polynomial
import relaxations as relax

class distributedFunnel:

    def __init__(self, dynSys:dynamicalSystem, lyapFunc:LyapunovFunction, traj:referenceTrajectory, evolveLyap:lyapEvol, branchingAlg, opts={}):
        self.dynSys = dynSys
        self.lyapFunc = lyapFunc
        self.traj = traj
        self.evolveLyap = evolveLyap
        self.branchingAlg = branchingAlg
        
        self.repr = self.lyapFunc.repr

        self.opts = {'convLim':1e-3, #Dichotomic
                     'minDistToSep':0.2, #When to refine search
                     'sphereBoundCritPoint':False, # Whether to use separation or spheric confinement
                     'interSteps':3, # How many points to check per interval
                     'projection':'sphere',
                     'optsEvol':{
                         'tDeltaMax':.1
                     }
                     }
        self.opts.update(opts)
        
        assert self.opts['sphereBoundCritPoint'] == False

    def doProjection(self, zonesToCheck, polyCoeffs:np.ndarray=None, criticalPoints:dict=None):
        """
        Simplifying the problem using some projection.
        Currently only the mapping between ellipsoids and unit-spheres is implemented
        :param zonesToCheck:
        :param objectiveStar:
        :param criticalPoints:
        :return: directly work on input
        """
        
        assert (polyCoeffs is None) or (len(zonesToCheck) == len(polyCoeffs))
        assert (criticalPoints is None) or (len(zonesToCheck) == len(criticalPoints))
        
        if self.opts['projection'] == "sphere":
            if isinstance(self.lyapFunc, lyapunovFunctions.quadraticLyapunovFunction):
                
                for i, aZone in enumerate(zonesToCheck):
                    Ci = cholesky(aZone[0]/aZone[1])
                    Ci_inv = inv(Ci) #aZone=(P_i, alpha_i)
                    if polyCoeffs is not None:
                        for k in range(polyCoeffs[i].shape[0]):
                            polyCoeffs[i][k,:] = self.repr.doLinCoordChange(polyCoeffs[i][k,:], Ci_inv)
                    if criticalPoints is not None:
                        for k in range(len(criticalPoints)):
                            criticalPoints['y'] = ndot(Ci, criticalPoints['x'])
            else:
                raise NotImplementedError
        elif self.opts['projection'] in (None, 'None'):
            pass
        else:
            raise NotImplementedError
        
        return None
    
    def ZoneNPoints2Prob(self, aZone:"A Lyap Zone", aCritPList:List, objStarinput:np.ndarray):
        """
        
        :param aZone: [[P_0, alpha_0], ...]
        :param aCritPList: [{'x':np.ndarray, other}, ...]
        :return:
        """
        
        
        
        if isinstance(self.lyapFunc, lyapunovFunctions.quadraticLyapunovFunction):
            if self.opts['sphereBoundCritPoint']:
                raise NotImplementedError
            else:
                # Use optimal control for each critical point, that is use optimal control on the polytope
                # Exclude the largest sphere possible (in transformed space) for the linear control
                
                # 1 Get the linear control laws
                
                
                helperPoly = [polynomial(self.repr, helperObj[k,:]) for k in range(helperObj.shape[0])]
                
                # 2 Extract separating hyperplanes
                Lcstr = helperObj[:, self.repr.varNumsPerDeg[1]].copy()
                
                # 3 Create the problems
                probList = []
                # 3.1 The linear control approximation
                probList.append()
                
                
                
                
                
                
                
                
                
        
        else:
            print("Others need to be implemented first")
            raise NotImplementedError
            
        
    def solve1(self, zonesToCheck, critPoints, objStar, deltaU, routingDict, probQueus, solQueues):
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
        
        probDict = dict( [(aId,[]) for aId in range(len(probQueus))] )
        probIdC = 0
        nQueues = len(probQueus)
        
        
        # Set problems
        for k, (aZone, aCritPList, aObjStar) in zip(zonesToCheck, critPoints):
    
            # Get the optimal linear control law
            uCtrlLin, uMonomLin = pendSys.ctrlInput.getU(narray([2]), 0., P=lyapF.P, PG0=ndot(lyapF.P_, gTaylorX0[0, :, :]), alpha=lyapF.alpha,
                                                         monomOut=True)
            objectArrayLin = lyapF.getObjectiveAsArray(fTaylorX0, gTaylorX0, taylorDeg=3, u=uCtrlLin, uMonom=uMonomLin)
            aObjLin =
            
            thisProbList = self.ZoneNPoints2Prob(aZone, aCritPList, aObjStar[1:,:])
            
            for aProb in thisProbList:
                aProb['probId'] = probIdC
                thisId = probSetter(aProb, probQueus, routingDict, divmod(probIdC, nQueues)[0])
                probDict[thisId].append(probIdC)
                probIdC += 1
        
        # Collect results
        
            
        
            
            
            
            
        
        

    def compute(self, routingDict, probQueues, solQueues, tStart:float, tStop:float, initZone):
        
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

        # Back propagate
        while tC > tStart:
            tC, nextZoneGuess = self.evolveLyap(tC, self.opts['optsEvol']['tDeltaMax'], lyapFunc_.getLyap(tL))
            tSteps = np.linspace(tL, tC, self.opts['interSteps'])

            # Get the current taylor series
            allTaylorApprox = [ self.dynSys.getTaylorApprox(self.traj.getX(aT)) for aT in tSteps ]
            allU = [ [self.dynSys.ctrlInput.getMinU(aT), self.traj.getU(aT), self.dynSys.ctrlInput.getMaxU(aT)] for aT in tSteps ]
            allDeltaU = [ [aU[0]-aU[1], aU[2]-aU[1]] for aU in allU ]

            objectiveStar = [ self.lyapFunc.getObjectiveAsArray(allTaylorApprox[k][0], allTaylorApprox[k][1], np.ones((self.dynSys.nu,1)),
                                                                np.tile(self.repr.varNumsPerDeg[0], (self.dynSys.nu,1)), self.traj.getX(tSteps[k]), self.traj.getDX(tSteps[k]), tSteps[k]) for k in range(self.opts['interSteps']) ]
            
            # TODO this is too long and should be seperated
            
            lyapFunc_.register(tC, nextZoneGuess)

            zonesToCheck = lyapFunc_.getPnPdot(tSteps)
            
            # Step 1 - critical points
            # TODO forward calculate the trajectory of the current crit points
            criticalPoints = [[] for _ in range(self.opts['interSteps'])]

            # Perform problem projection if demanded
            self.doProjection(zonesToCheck, objectiveStar, criticalPoints)
            

            # Step 2 check if current is feasible

            minConvList = self.solve1(zonesToCheck, criticalPoints, objectiveStar, allDeltaU, routingDict, probQueues, solQueues)





            tL=tC #Reset




