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

            objectiveStar = [ self.lyapFunc.getObjectiveAsArray(self.traj.getX(tSteps[k]), self.traj.getDX(tSteps[k]), allTaylorApprox[k][0],
                                                             allTaylorApprox[k][1], np.ones((self.dynSys.nu,1)) )  for k in range(self.opts[
                                                                                                                                                                    'interSteps'])]

            lyapFunc_.register(tStop, nextZoneGuess)

            stateDicts = [ [] ]

            zonesToCheck = lyapFunc_.getLyapNDeriv(tSteps)

            concreteProblems = self.ZonesNPoints2Problems(zonesToCheck, criticalPoints)





            tL=tC #Reset




