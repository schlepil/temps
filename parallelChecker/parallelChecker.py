from coreUtils import *

import relaxations as rel
import polynomial as poly


from parallelChecker.parallelDefinitions import *
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import RawArray as mpRawArray

import ctypes
import queue
import time

cfloat = np.ctypeslib.as_ctypes_type(nfloat)

# The shared variables for faster com
quadraticLyapShared_ = [ mpRawArray(cfloat, lenBuffer_) for _ in range(nThreads_)] # Assuming x'.P.x <= 1.
polyObjShared_ = [ mpRawArray(cfloat, lenBuffer_) for _ in range(nThreads_)]
polyCstrShared_ = [ [mpRawArray(cfloat, lenBuffer_) for _ in range(nCstrMax_)] for _ in range(nThreads_) ]
# Easy access via np
quadraticLyapSharedNP_ = [ np.frombuffer(aS, dtype=nfloat, count=lenBuffer_) for aS in quadraticLyapShared_ ]
polyObjSharedNP_ = [ np.frombuffer(aS, dtype=nfloat, count=lenBuffer_) for aS in polyObjShared_ ]
polyCstrSharedNP_ = [ [np.frombuffer(aS, dtype=nfloat, count=lenBuffer_) for aS in aSL]  for aSL in polyCstrShared_ ]


def probSetter(problem: dict, probQueue: Queue, workerId: int):
    # First copy, then put into queue
    # Necessary
    polyObjSharedNP_[workerId][:problem['obj'].size] = problem['obj']
    for k, aCstr in enumerate(problem['cstr']):
        polyCstrSharedNP_[workerId][k][:aCstr.size] = aCstr
    # Optional
    if 'quadLyapP' in problem.keys():
        quadraticLyapSharedNP_[workerId][:problem['quadLyaP'].size] = problem['quadLyaP'].flatten()
    
    problem['probDict']['workerId'] = workerId
    probQueue.put(problem['probDict'])
    
    
    return workerId


def solGetter(solQueues: List[Queue], workerId: int = None, block=True, timeout=0.0001):
    if workerId is None:
        if block:
            while True:
                for aQueue in solQueues:
                    try:
                        return aQueue.get(block=block, timeout=timeout)
                    except queue.Empty:
                        pass
                time.sleep(0.001)
        else:
            sol = None
            for aQueue in solQueues:
                if sol is not None:
                    break
                try:
                    sol = aQueue.get(block=block, timeout=timeout)
                except queue.Empty:
                    pass
            if sol is None:
                raise queue.Empty
            else:
                return sol
    
    else:
        return solQueues[workerId].get(block=block, timeout=timeout)


class workDistributor:
    def __init__(self, probQueues:List[Queue], solQueues:List[Queue]):
        self.inUse = narray([0 for _ in range(nThreads_)]).astype(np.bool_)
        self.waitingList = []
        
        self.probQ = probQueues
        self.solQ = solQueues
        
        self.block_ = True
        self.timeout_ = 0.0001
        
        self.nbrOfUnreturnedPbr = 0
        
        self.probStore_ = {}
        self.doStoreOrig = True
    
    def reset(self):
        self.waitingList = []
        self.nbrOfUnreturnedPbr = 0
        
        while any(self.inUse):
            for k, aInUse in enumerate(self.inUse):
                if not aInUse:
                    continue
                self.solQ[k].get()
                self.inUse[k] = False
        
        return None
    
    def spin(self):
        while len(self.waitingList) and self.inUse.sum()<nThreads_:
            thisProb = self.waitingList.pop()
            thisWorkerId = np.argwhere(np.logical_not(self.inUse))[0,0] #TODO check why argwhere is always 2d
            probSetter(thisProb, self.probQ[thisWorkerId], thisWorkerId)
            self.inUse[thisWorkerId] = True
            
            # Test
            #workerSolve(self.probQ[thisWorkerId], self.solQ[thisWorkerId])
            
        return None
    
    def setProb(self, problem:dict):
        self.nbrOfUnreturnedPbr += 1
        
        if self.doStoreOrig:
            while True:
                id = np.random.randint(np.core.getlimits.iinfo(np.int_)['min']+1, np.core.getlimits.iinfo(np.int_)['max']-1 )
                if not (id in self.probStore_.keys()):
                    break
            problem['probDict']['probIdStore__'] = id
            self.probStore_[id] = dp(problem)
        
        self.waitingList.append(problem)
        self.spin()
        return None
    
    def getSol(self, workerId=None, block=None, timeout=None):
        block = self.block_ if block is None else block
        timeout = self.timeout_ if timeout is None else timeout
        if workerId is None:
            while True:
                try:
                    sol = solGetter(self.solQ, workerId, False, timeout)
                    self.inUse[sol['probDict']['workerId']] = False
                    self.spin()
                    break
                except queue.Empty:
                    self.spin()
                    time.sleep(0.001)
                if not block:
                    if sol is None:
                        raise queue.Empty
                    break
        else:
            sol = solGetter(self.solQ, workerId, block, timeout)
            self.inUse[sol['probDict']['workerId']] = False
            self.spin()
        self.nbrOfUnreturnedPbr -= 1
        
        if self.doStoreOrig:
            sol['origProb'] = self.probStore_[sol['probDict']['probIdStore__']]
            del self.probStore_[sol['probDict']['probIdStore__']]
        
        return sol

# inputDict
# {'dimsNDeg':(nDims,nDeg)
# {'nCstrNDegType':(nCstr,deg0,deg1,...)

def workerSolve(inQueue, outQueue):

    reprDict = {}
    relaxationDict = {}
    problemDict = {}

    #input = inQueue.get()

    #assert 'initial' in input.keys()
    #assert 'nr' in input.keys()
    #assert 'solver' in input.keys()

    #selfNr = input['nr']
    #selfSolver = input['solver']

    while True:

        input = inQueue.get()
        
        selfNr = input['workerId']
        selfSolver = input['solver']

        if input == "":
            print(f"Worker {selfNr} is terminating")
            break

        try:
            thisRepr = reprDict[input['dimsNDeg']]
            thisRelax = relaxationDict[input['dimsNDeg']]
            thisProb = problemDict[input['dimsNDeg']][input['nCstrNDegType']] # TODO reduce memory footprint by making it order invariant

            #thisProb.objective = np.frombuffer(polyObjShared_[selfNr], nfloat, thisRepr.nMonoms)
            thisProb.objective = polyObjSharedNP_[selfNr][:thisRepr.nMonoms].copy()

            counters = {'l':0, 'q':0, 's':1 }
            for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType']):
                assert cstrType in counters.keys()
                if cstrType == 'l':
                    raise NotImplemented
                elif cstrType == 'q':
                    raise NotImplemented
                else:
                    thisCstr = thisProb.constraints.s.cstrList[counters[cstrType]]
                    assert thisCstr.polyDeg == nDeg
                    #thisCstr.coeffs = np.frombuffer(polyCstrShared_[selfNr][k], nfloat, thisRepr.nMonoms)
                    thisCstr.coeffs = polyCstrSharedNP_[selfNr][k][:thisRepr.nMonoms].copy()
                    assert thisCstr.polyDeg == nDeg
                    counters[cstrType] += 1

        except KeyError:
            nDims, maxDeg = input['dimsNDeg']

            try:
                thisRepr = reprDict[input['dimsNDeg']]
                thisRelax = relaxationDict[input['dimsNDeg']]
            except KeyError:
                thisRepr = poly.polynomialRepr(nDims, maxDeg)
                thisRelax = rel.lasserreRelax(thisRepr)
                relaxationDict[input['dimsNDeg']] = thisRepr

            # Get all polynomials
            # objective
            #polyObj = poly.polynomial(thisRepr, coeffs=np.frombuffer(polyObjShared_[selfNr], nfloat, thisRepr.nMonoms), alwaysFull=True)
            polyObj = poly.polynomial(thisRepr, coeffs=polyObjSharedNP_[selfNr][:thisRepr.nMonoms].copy(), alwaysFull=True)

            # Add the objective
            thisProb = rel.convexProg(thisRepr, selfSolver, objective=polyObj)

            # constraints
            thisProb.addCstr( thisRelax )
            for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType']):
                #thisCstr = rel.lasserreConstraint(thisRepr, poly.polynomial(thisRepr, coeffs=np.frombuffer(polyCstrShared_[selfNr][k], nfloat, thisRepr.nMonoms), alwaysFull=True))
                thisCstr = rel.lasserreConstraint(thisRelax, poly.polynomial(thisRepr, coeffs=polyCstrSharedNP_[selfNr][k][:thisRepr.nMonoms].copy(), alwaysFull=True))
                assert (nDeg == thisCstr.poly.maxDeg), "Incompatible degrees"
                # Add the constraints
                thisProb.addCstr(thisCstr)


        # Check if need to transform
        if ('toUnitCircle' in input.keys()) and (input['toUnitCircle']):
            nDims = input['dimsNDeg'][0]
            #P = np.frombuffer(quadraticLyapShared_[selfNr], nfloat, nDims**2).copy().reshape((nDims, nDims))
            P = quadraticLyapSharedNP_[selfNr][:nDims**2].copy().reshape((nDims, nDims))
            C = cholesky(P, lower=False)
            Ci = inv(C)
            thisProb.objective.coeffs = thisRepr.doLinCoordChange(thisProb.objective.coeffs, Ci)

            assert thisProb.constraints.l.nCstr == 0, 'TODO' #TODO
            assert thisProb.constraints.q.nCstr == 0, 'TODO' #TODO

            for k in range(1,thisProb.constraints.s.nCstr):
                thisProb.constraints.s.cstrList[k].poly.coeffs = thisRepr.doLinCoordChange(thisProb.constraints.s.cstrList[k].poly.coeffs, Ci)

        # Actually solve
        solution = thisProb.solve()
        assert solution['status'] == 'optimal'
        extraction = thisProb.extractOptSol(solution)

        if ('toUnitCircle' in input.keys()) and (input['toUnitCircle']):
            # Add the unscaled solution
            ySol = ndot(C, extraction[0])

        else:
            ySol = extraction[0]

        outQueue.put({'probDict':input, 'probId':input['probId'], 'xSol':extraction[0], 'ySol':ySol, 'sol':solution, 'ext':extraction})

    return 0





        
        
















