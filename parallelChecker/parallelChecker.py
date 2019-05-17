from coreUtils import *

import relaxations as rel
import polynomial as poly


from parallelChecker.parallelDefinitions import *
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import RawArray as mpRawArray


import queue
import time

import os, json, pickle

mySerializer = variableStruct()
mySerializer.dumps = lambda x: str(x)

serializer = pickle
serializer = mySerializer

if useSharedMem_:
    import ctypes
    
    cfloat = np.ctypeslib.as_ctypes_type(nfloat)
    # The shared variables for faster com
    quadraticLyapShared_ = [ mpRawArray(cfloat, lenBuffer_) for _ in range(nThreads_)] # Assuming x'.P.x <= 1.
    polyObjShared_ = [ mpRawArray(cfloat, lenBuffer_) for _ in range(nThreads_)]
    polyCstrShared_ = [ [mpRawArray(cfloat, lenBuffer_) for _ in range(nCstrMax_)] for _ in range(nThreads_) ]
    # Easy access via np
    quadraticLyapSharedNP_ = [ np.frombuffer(aS, dtype=nfloat, count=lenBuffer_) for aS in quadraticLyapShared_ ]
    polyObjSharedNP_ = [ np.frombuffer(aS, dtype=nfloat, count=lenBuffer_) for aS in polyObjShared_ ]
    polyCstrSharedNP_ = [ [np.frombuffer(aS, dtype=nfloat, count=lenBuffer_) for aS in aSL]  for aSL in polyCstrShared_ ]

# Dict to store representations and others to avoid recomputation
reprDict_ = {}
relaxationDict_ = {}
problemDict_ = {}

def probSetterShared_(problem: dict, probQueue: Queue, workerId: int):
    # First copy, then put into queue
    # Necessary
    
    if __debug__ and printProbNSol_:
        print(f"Putting; \n {serializer.dumps(problem)}")
    
    polyObjSharedNP_[workerId][:problem['obj'].size] = problem['obj']
    for k, aCstr in enumerate(problem['cstr']):
        polyCstrSharedNP_[workerId][k][:aCstr.size] = aCstr
    # Optional
    if 'quadLyapP' in problem.keys():
        quadraticLyapSharedNP_[workerId][:problem['quadLyaP'].size] = problem['quadLyaP'].flatten()
    
    problem['probDict']['workerId'] = workerId
    probQueue.put(problem['probDict'])
    
    
    return workerId


def probSetterBare_(problem: dict, probQueue: Queue, workerId: int):
    # Put all into the queue
    
    if __debug__:
        print(f"Putting; \n {serializer.dumps(problem)}")
    
    problem['probDict']['workerId'] = workerId
    probQueue.put(problem)
    
    return workerId

if useSharedMem_:
    probSetter = probSetterShared_
else:
    probSetter = probSetterBare_


def solGetter(solQueues: List[Queue], workerId: int = None, block=True, timeout=0.0001):
    sol = None
    if workerId is None:
        if block:
            while sol is None:
                for aQueue in solQueues:
                    try:
                        sol =  aQueue.get(block=block, timeout=timeout)
                        break
                    except queue.Empty:
                        pass
                time.sleep(0.0001)
        else:
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
        sol = solQueues[workerId].get(block=block, timeout=timeout)
    
    if __debug__ and printProbNSol_:
        print(f"Recieving; \n {serializer.dumps(sol)}")
    
    return sol


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
    
    def terminate(self):
        self.reset()
        for aQ in self.probQ:
            aQ.put("")
        return None
    
    def reset(self):
        self.waitingList = []
        self.nbrOfUnreturnedPbr = 0
        
        while any(self.inUse):
            for k, aInUse in enumerate(self.inUse):
                if not aInUse:
                    continue
                self.solQ[k].get()
                self.inUse[k] = False
        
        if self.doStoreOrig:
            self.probStore_ = {}
        
        return None
    
    def spin(self):
        while len(self.waitingList) and self.inUse.sum()<nThreads_:
            thisProb = self.waitingList.pop()
            thisWorkerId = np.argwhere(np.logical_not(self.inUse))[0,0] #TODO check why argwhere is always 2d
            probSetter(thisProb, self.probQ[thisWorkerId], thisWorkerId)
            self.inUse[thisWorkerId] = True
            
        return None
    
    def setProb(self, problem:dict):
        self.nbrOfUnreturnedPbr += 1
        
        if self.doStoreOrig:
            while True:
                id = np.random.randint(np.core.getlimits.iinfo(np.int_).min+1, np.core.getlimits.iinfo(np.int_).max-1 )
                if not (id in self.probStore_.keys()):
                    break
            problem['probDict']['probIdStore__'] = id
            self.probStore_[id] = dp(problem)
        
        self.waitingList.append(problem)
        self.spin()
        
        if __debug__ and printProbNSol_:
            print(f"Appending: \n {serializer.dumps(problem)}")
        
        return None
    
    def getSol(self, workerId=None, block=None, timeout=None):
        block = self.block_ if block is None else block
        timeout = self.timeout_ if timeout is None else timeout
        if workerId is None:
            while True:
                try:
                    sol = solGetter(self.solQ, workerId, False, timeout)
                    if sol == "":
                        raise RuntimeError
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
            if sol == "":
                raise RuntimeError
            self.inUse[sol['probDict']['workerId']] = False
            self.spin()
        self.nbrOfUnreturnedPbr -= 1
        
        if self.doStoreOrig:
            sol['origProb'] = self.probStore_[sol['probDict']['probIdStore__']]
            del self.probStore_[sol['probDict']['probIdStore__']]
        
        if __debug__ and printProbNSol_:
            print(f"Returning: \n {serializer.dumps(sol)}")
        
        return sol

# No threading
class workDistributorNoThread:
    def __init__(self, probQueues: List[Queue], solQueues: List[Queue]):
        self.waitingList = []
        
        self.probQ = probQueues
        self.solQ = solQueues
        
        self.nbrOfUnreturnedPbr = 0
        
        self.probStore_ = {}
        self.doStoreOrig = True
    
    def terminate(self):
        pass
    
    def reset(self):
        self.waitingList = []
        self.nbrOfUnreturnedPbr = 0
        
        if self.doStoreOrig:
            self.probStore_ = {}
        
        return None
    
    def spin(self):
        
        thisProb = self.waitingList.pop()
        thisWorkerId = 0
        probSetter(thisProb, self.probQ[thisWorkerId], thisWorkerId)
        #probSetter({"probDict":""}, self.probQ[thisWorkerId], thisWorkerId)
        self.probQ[thisWorkerId].put("")
        workerSolve(self.probQ[0], self.solQ[0])
        
        return None
    
    def setProb(self, problem: dict):
        self.nbrOfUnreturnedPbr += 1
        
        if self.doStoreOrig:
            while True:
                id = np.random.randint(np.core.getlimits.iinfo(np.int_).min+1, np.core.getlimits.iinfo(np.int_).max-1)
                if not (id in self.probStore_.keys()):
                    break
            problem['probDict']['probIdStore__'] = id
            self.probStore_[id] = dp(problem)
        
        self.waitingList.append(problem)
        
        if __debug__ and printProbNSol_:
            print(f"Appending: \n {serializer.dumps(problem)}")
        
        return None
    
    def getSol(self, workerId=None, block=None, timeout=None):
        # No threading always uses zero worker
        self.spin()
        sol = solGetter(self.solQ, 0, True, timeout=100.)
        if sol == "":
            raise RuntimeError
        
        self.nbrOfUnreturnedPbr -= 1
        
        if self.doStoreOrig:
            sol['origProb'] = self.probStore_[sol['probDict']['probIdStore__']]
            del self.probStore_[sol['probDict']['probIdStore__']]
        
        if __debug__ and printProbNSol_:
            print(f"Returning: \n {serializer.dumps(sol)}")
        
        return sol

# inputDict
# {'dimsNDeg':(nDims,nDeg)
# {'nCstrNDegType':(nCstr,deg0,deg1,...)

def workerSolve(inQueue, outQueue):

    #input = inQueue.get()

    #assert 'initial' in input.keys()
    #assert 'nr' in input.keys()
    #assert 'solver' in input.keys()

    #selfNr = input['nr']
    #selfSolver = input['solver']
    
    global reprDict_
    global relaxationDict_
    global problemDict_


    while True:

        input = inQueue.get()

        if input == "":
            print(f"Worker {selfNr} is terminating")
            break
        
        if not useSharedMem_:
            inputAll = input
            input = input['probDict']
            
        try:
            selfNr = input['workerId']
        except KeyError:
            selfNr = input['workerId'] = os.getpid()
        selfSolver = input['solver']
        
        if __debug__:
            print(f"Worker {selfNr} recieved new input")
        
        try:
            thisRepr = reprDict_[input['dimsNDeg']]
            thisRelax = relaxationDict_[input['dimsNDeg']]
            thisProb = problemDict_[input['dimsNDeg']][tuple(input['nCstrNDegType'])] # TODO reduce memory footprint by making it order invariant
            
            if __debug__:
                print(f"Worker {selfNr} found corresponding representation, relaxation and problem")
            
            #thisProb.objective = np.frombuffer(polyObjShared_[selfNr], nfloat, thisRepr.nMonoms)
            if useSharedMem_:
                thisProb.objective = polyObjSharedNP_[selfNr][:thisRepr.nMonoms].copy()
            else:
                thisProb.objective = inputAll['obj']

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
                    if useSharedMem_:
                        thisCstr.coeffs = polyCstrSharedNP_[selfNr][k][:thisRepr.nMonoms].copy()
                    else:
                        thisCstr.coeffs = inputAll['cstr'][k]
                    assert thisCstr.polyDeg == nDeg
                    counters[cstrType] += 1

        except KeyError:
            nDims, maxDeg = input['dimsNDeg']

            try:
                thisRepr = reprDict_[input['dimsNDeg']]
                if __debug__:
                    print(f"Worker {selfNr} found representation")
            except KeyError:
                thisRepr = poly.polynomialRepr(nDims, maxDeg)
                reprDict_[input['dimsNDeg']] = thisRepr
                if __debug__:
                    print(f"Worker {selfNr} created representation")
            
            try:
                thisRelax = relaxationDict_[input['dimsNDeg']]
                if __debug__:
                    print(f"Worker {selfNr} found relaxation")
            except KeyError:
                thisRelax = rel.lasserreRelax(thisRepr)
                relaxationDict_[input['dimsNDeg']] = thisRelax
                if __debug__:
                    print(f"Worker {selfNr} created relaxation")

            # Get all polynomials
            # objective
            if useSharedMem_:
                polyObj = poly.polynomial(thisRepr, coeffs=polyObjSharedNP_[selfNr][:thisRepr.nMonoms].copy(), alwaysFull=True)
            else:
                polyObj = poly.polynomial(thisRepr, coeffs=inputAll['obj'], alwaysFull=True)

            # Add the objective
            thisProb = rel.convexProg(thisRepr, selfSolver, objective=polyObj)

            # constraints
            thisProb.addCstr( thisRelax )
            for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType']):
                if useSharedMem_:
                    thisCstr = rel.lasserreConstraint(thisRelax, poly.polynomial(thisRepr, coeffs=polyCstrSharedNP_[selfNr][k][:thisRepr.nMonoms].copy(), alwaysFull=True))
                else:
                    thisCstr = rel.lasserreConstraint(thisRelax, poly.polynomial(thisRepr, coeffs=inputAll['cstr'][k], alwaysFull=True))
                assert (nDeg == thisCstr.poly.maxDeg), "Incompatible degrees"
                # Add the constraints
                thisProb.addCstr(thisCstr)
            
            # Save the problem
            if not input['dimsNDeg'] in problemDict_.keys():
                problemDict_[input['dimsNDeg']] = {}
            problemDict_[input['dimsNDeg']][tuple(input['nCstrNDegType'])] = thisProb
            if __debug__:
                print(f"Worker {selfNr} created the problem structure for {input['dimsNDeg']} and {input['nCstrNDegType']}")


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
        if doThreading_:
            if not solution['status'] == 'optimal':
                if useSharedMem_:
                    print(f"Failed on \n {input} \n\n with \n {solution}")
                else:
                    print(f"Failed on \n {inputAll} \n\n with \n {solution}")
                outQueue.put("")
        else:
            if not (solution['status'] == 'optimal'):
                print("Error in solving")
                import plotting as plt
                ff,aa = plt.plt.subplots(1,1)
                aa.set_xlim(-2,2)
                aa.set_ylim(-2,2)
                plt.plot2dCstr(thisProb, aa, {'binaryPlot':True}, fig=ff)
        try:
            extraction = thisProb.extractOptSol(solution)
        except:
            print('a')

        if ('toUnitCircle' in input.keys()) and (input['toUnitCircle']):
            # Add the unscaled solution
            ySol = ndot(C, extraction[0])

        else:
            ySol = extraction[0]
        
        if __debug__:
            if solution['primal objective'] < -1.e-6:
                print(f"Found critical point with {solution['primal objective']} at \n {ySol}")
            print(f"Optimal value is ")
            if extraction[0].size == 0:
                thisProb.extractOptSol(solution)

        outQueue.put({'probDict':input, 'xSol':extraction[0], 'ySol':ySol, 'sol':solution, 'ext':extraction})

    return 0





        
        
















