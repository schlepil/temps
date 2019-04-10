from coreUtils import *

from parallelChecker import parallelDefinitions as paraDef

from dynamicalSystems import dynamicalSystem
from Lyapunov import LyapunovFunction
from trajectories import referenceTrajectory

class distributedFunnel:

    def __init__(self, dynSys:dynamicalSystem, lyapFunc:LyapunovFunction, traj:referenceTrajectory, evolveLyap, branchingAlg, opts={}):
        self.dynSys = dynSys
        self.lyapFunc = lyapFunc
        self.traj = traj
        self.evolveLyap = evolveLyap
        self.branchingAlg = branchingAlg
        
        self.repr = self.lyapFunc.repr

        self.opts = {'convLim':1e-3, #Dichotomic
                     'minDistToSep':0.2,
                     'sphereBoundCritPoint':True,
                     'interSteps':3
                     }
        self.opts.update(opts)



    def compute(self, routingDict, probQueues, solQueues, tStart:float, tStop:float, initZone):

        lyapFunc_ = self.lyapFunc

        assert tStart>=self.traj.tLims[0]
        assert tStop<=self.traj.tLims[1]

        tL = tStop
        tC = tStop


        lyapFunc_.reset()
        lyapFunc_.register(tStop, initZone)

        # Back propagate
        while tC > tStart:
            tC, nextZoneGuess = self.evolveLyap(tC, lyapFunc_.getLyap(tL))
            tSteps = np.linspace(tL, tC, self.opts['interSteps'])

            # Get the current taylor series
            allTaylorApprox = [ self.dynSys.getTaylorApprox(self.traj.getX(aT)) for aT in tSteps ]
            allU = [ (self.dynSys.ctrlInput.getMinU(aT), self.traj.getU(aT), self.dynSys.ctrlInput.getMaxU(aT)) for aT in tSteps ]
            allDeltaU = [allU[0]-allU[1], allU[2]-allU[1]]

            objectiveStar = [ self.lyapFunc.getObjectiveAsArray(allTaylorApprox[k][0], allTaylorApprox[k][1], np.ones((self.dynSys.nu,1)),
                                                                np.tile(self.repr.varNumsPerDeg[0], (self.dynSys.nu,1)), self.traj.getX(tSteps[k]), self.traj.getDX(tSteps[k]), tSteps[k]) for k in range(self.opts['interSteps']) ]
            
            # TODO this is too long and should be seperated

            lyapFunc_.register(tStop, nextZoneGuess)
            criticalPoints = [ [] for _ in range(self.opts['interSteps'])]

            zonesToCheck = lyapFunc_.getPnPdot(tSteps)

            # Step 1 check if current is feasible

            minConvList = self.solve1(zonesToCheck, criticalPoints, routingDict, probQueues, solQueues)

            concreteProblems = self.ZonesNPoints2Problems(zonesToCheck, criticalPoints)





            tL=tC #Reset




