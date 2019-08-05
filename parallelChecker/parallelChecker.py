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
mySerializer.dumps = lambda x: ""

#serializer = pickle
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

if waitingListType_ == 'heap':
    from heapq import heappush, heappop
    
    class heapStoreClass(tuple):
        def __lt__(self, other):
            return self[0] < other[0]
    
    class waitingListClass:
        def __init__(self):
            self.container = []
        def push(self, *args):
            if len(args) == 2:
                #Explicit value is given
                heappush(self.container, heapStoreClass(args))
            else:
                #Use the (negated) sum over the control indices -> the higher the indices the more specific the problem -> the higher the chance to fail
                heappush(self.container, heapStoreClass((int(-nsum(args[0]['probDict']['u'])), args[0])))
        
        def pop(self):
            return heappop(self.container)[1]
            
elif waitingListType_ == 'list':
    class waitingListClass(list):
        def push(self, val):
            self.append(val)
else:
    raise RuntimeError("list or heap")
        

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
        self.waitingList = waitingListClass()
        
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
        self.waitingList = waitingListClass()
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
        
        self.waitingList.push(problem)
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
        self.waitingList = waitingListClass()
        
        self.probQ = probQueues
        self.solQ = solQueues
        
        self.nbrOfUnreturnedPbr = 0
        
        self.probStore_ = {}
        self.doStoreOrig = True
    
    def terminate(self):
        pass
    
    def reset(self):
        self.waitingList = waitingListClass()
        self.nbrOfUnreturnedPbr = 0
        
        if self.doStoreOrig:
            self.probStore_ = {}
            
        while not self.solQ[0].empty():
            self.solQ.get()
        while not self.probQ[0].empty():
            self.probQ.get()
        
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
        
        self.waitingList.push(problem)
        
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


def get_or_create(dims_deg:Tuple, cstr_deg_type:List, solver='cvxopt', id_str:str="Base"):

    """
    Checks if a corresponding problem already exists and creates the necessary structures if non-existant
    :param dims_deg:
    :param cstr_deg_type:
    :param solver:
    :param id_str:
    :return:
    """

    global reprDict_
    global relaxationDict_
    global problemDict_

    if not isinstance(dims_deg, tuple):
        dims_deg = tuple(dims_deg)

    try:
        this_repr = reprDict_[dims_deg]
        if __debug__:
            print(f"{id_str} found representation")
    except KeyError:
        this_repr = poly.polynomialRepr(*dims_deg)
        reprDict_[dims_deg] = this_repr
        if __debug__:
            print(f"{id_str} created representation")

    try:
        this_relax = relaxationDict_[dims_deg]
        if __debug__:
            print(f"{id_str} found relaxation")
    except KeyError:
        this_relax = rel.lasserreRelax(this_repr)
        relaxationDict_[dims_deg] = this_relax
        if __debug__:
            print(f"{id_str} created relaxation")

    try:
        this_prob = problemDict_[dims_deg][tuple(cstr_deg_type)]
    except KeyError:
        # Add the objective
        poly_obj = poly.polynomial(this_repr, alwaysFull=True)
        this_prob = rel.convexProg(this_repr, solver, objective=poly_obj)

        this_prob.addCstr(this_relax)
        for k, (deg, cstr_type) in enumerate(cstr_deg_type):
            dummy = nzeros((this_repr.nMonoms,), dtype=nfloat)
            dummy[this_repr.varNumsUpToDeg[deg]] = 1.  # To enforce correct degree detection
            this_cstr = rel.lasserreConstraint(this_relax, poly.polynomial(this_repr, coeffs=dummy, alwaysFull=True))
            # Add the constraints
            this_prob.addCstr(this_cstr)

        # Save the problem
        if not dims_deg in problemDict_.keys():
            problemDict_[dims_deg] = {}
        problemDict_[dims_deg][tuple(cstr_deg_type)] = this_prob
        if __debug__:
            print(f"{id_str} created the problem structure for {dims_deg} and {cstr_deg_type}")

    return this_repr, this_relax, this_prob


def do_fill(a_prob:rel.convexProg, input:dict):
    """
    Fills the problem with the given coefficients
    :param a_prob:
    :param input:
    :return:
    """


    # Ensure the consistency of the used degrees
    deg_prob = a_prob.repr.maxDeg
    if not useSharedMem_:
        deg_input = [var_uptodeg_a.size for var_uptodeg_a in a_prob.repr.varNumsUpToDeg].index(input['obj'].size)
    else:
        raise NotImplementedError
        deg_input = deg_prob


    # pregenerate the slices
    slice_min = slice(0, len(a_prob.repr.varNumsUpToDeg[min(deg_input, deg_prob)]))
    if __debug__:
        slice_excl = slice(len(a_prob.repr.varNumsUpToDeg[min(deg_input, deg_prob)]), input['obj'].size)
        assert nall(input['obj'][slice_excl] == 0.)

    # Fill
    if useSharedMem_:
        raise NotImplementedError
    else:
        # input corresponds to input all

        probDict = input['probDict']

        # In order to be safe, no matter the input and problem degrees, always set zero first
        # objective
        # TODO check if it would not be better to always allow for modifications
        a_prob.objective.unlock()
        a_prob.objective.coeffs.fill(0.)
        a_prob.objective.coeffs[slice_min] = input['obj'][slice_min]

        # constraints
        counters = {'l': 0, 'q': 0, 's': 1}
        for k, (nDeg, cstrType) in enumerate(probDict['nCstrNDegType']):
            assert cstrType in counters.keys()
            if cstrType == 'l':
                raise NotImplemented
            elif cstrType == 'q':
                raise NotImplemented
            else:
                if __debug__:
                    assert input['cstr'][k].size == input['obj'].size
                    assert nall(input['cstr'][k][slice_excl]==0.)

                thisCstr = a_prob.constraints.s.cstrList[counters[cstrType]]
                assert thisCstr.polyDeg == nDeg
                thisCstr.poly.unlock()
                thisCstr.coeffs.fill(0.)
                thisCstr.coeffs[slice_min] = input['cstr'][k][slice_min]
                counters[cstrType] += 1

    # Check if need to transform
    # TODO this is now done in the main program

    return None

def do_fill_solve(a_prob:rel.convexProg, input:dict):
    """
    fills and solves the convex problem
    :param a_prob:
    :param input:
    :return:
    """
    do_fill(a_prob, input)
    # solve
    return a_prob.solve()


def workerSolveFixed(inQueue, outQueue):

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
        
        # try:
        #     thisRepr = reprDict_[input['dimsNDeg']]
        #     thisRelax = relaxationDict_[input['dimsNDeg']]
        #     thisProb = problemDict_[input['dimsNDeg']][tuple(input['nCstrNDegType'])] # TODO reduce memory footprint by making it order invariant
        #
        #     if __debug__:
        #         print(f"Worker {selfNr} found corresponding representation, relaxation and problem")
        #
        #     #thisProb.objective = np.frombuffer(polyObjShared_[selfNr], nfloat, thisRepr.nMonoms)
        #     #if useSharedMem_:
        #     #     thisProb.objective = polyObjSharedNP_[selfNr][:thisRepr.nMonoms].copy()
        #     # else:
        #     #     thisProb.objective = inputAll['obj']
        #
        #     # counters = {'l':0, 'q':0, 's':1 }
        #     # for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType']):
        #     #     assert cstrType in counters.keys()
        #     #     if cstrType == 'l':
        #     #         raise NotImplemented
        #     #     elif cstrType == 'q':
        #     #         raise NotImplemented
        #     #     else:
        #     #         thisCstr = thisProb.constraints.s.cstrList[counters[cstrType]]
        #     #         assert thisCstr.polyDeg == nDeg
        #     #         if useSharedMem_:
        #     #             thisCstr.coeffs = polyCstrSharedNP_[selfNr][k][:thisRepr.nMonoms].copy()
        #     #         else:
        #     #             thisCstr.coeffs = inputAll['cstr'][k]
        #     #         assert thisCstr.polyDeg == nDeg
        #     #         counters[cstrType] += 1
        #
        # except KeyError:
        #     nDims, maxDeg = input['dimsNDeg']
        #
        #     try:
        #         thisRepr = reprDict_[input['dimsNDeg']]
        #         if __debug__:
        #             print(f"Worker {selfNr} found representation")
        #     except KeyError:
        #         thisRepr = poly.polynomialRepr(nDims, maxDeg)
        #         reprDict_[input['dimsNDeg']] = thisRepr
        #         if __debug__:
        #             print(f"Worker {selfNr} created representation")
        #
        #     try:
        #         thisRelax = relaxationDict_[input['dimsNDeg']]
        #         if __debug__:
        #             print(f"Worker {selfNr} found relaxation")
        #     except KeyError:
        #         thisRelax = rel.lasserreRelax(thisRepr)
        #         relaxationDict_[input['dimsNDeg']] = thisRelax
        #         if __debug__:
        #             print(f"Worker {selfNr} created relaxation")
        #
        #     # Get all polynomials
        #     # objective
        #     # if useSharedMem_:
        #     #     polyObj = poly.polynomial(thisRepr, coeffs=polyObjSharedNP_[selfNr][:thisRepr.nMonoms].copy(), alwaysFull=True)
        #     # else:
        #     #     polyObj = poly.polynomial(thisRepr, coeffs=inputAll['obj'], alwaysFull=True)
        #
        #     # Add the objective
        #     polyObj = poly.polynomial(thisRepr, alwaysFull=True)
        #     thisProb = rel.convexProg(thisRepr, selfSolver, objective=polyObj)
        #
        #     # constraints
        #     # thisProb.addCstr( thisRelax )
        #     # for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType']):
        #     #     if useSharedMem_:
        #     #         thisCstr = rel.lasserreConstraint(thisRelax, poly.polynomial(thisRepr, coeffs=polyCstrSharedNP_[selfNr][k][:thisRepr.nMonoms].copy(), alwaysFull=True))
        #     #     else:
        #     #         thisCstr = rel.lasserreConstraint(thisRelax, poly.polynomial(thisRepr, coeffs=inputAll['cstr'][k], alwaysFull=True))
        #     #     assert (nDeg == thisCstr.poly.maxDeg), "Incompatible degrees"
        #     #     # Add the constraints
        #     #     thisProb.addCstr(thisCstr)
        #     # Add empty constraints
        #     thisProb.addCstr( thisRelax )
        #     for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType']):
        #         dummy = nzeros((thisRepr.nMonoms,), dtype=nfloat)
        #         dummy[thisRepr.varNumsUpToDeg[nDeg]] = 1. #To enforce correct degree detection
        #         thisCstr = rel.lasserreConstraint(thisRelax, poly.polynomial(thisRepr, coeffs=dummy, alwaysFull=True))
        #         # Add the constraints
        #         thisProb.addCstr(thisCstr)
        #
        #     # Save the problem
        #     if not input['dimsNDeg'] in problemDict_.keys():
        #         problemDict_[input['dimsNDeg']] = {}
        #     problemDict_[input['dimsNDeg']][tuple(input['nCstrNDegType'])] = thisProb
        #     if __debug__:
        #         print(f"Worker {selfNr} created the problem structure for {input['dimsNDeg']} and {input['nCstrNDegType']}")

        thisRepr, thisRelax, thisProb = get_or_create(input['dimsNDeg'], input['nCstrNDegType'], solver = input['solver'], id_str=f"Worker {selfNr} ")

        # Actually solve
        #solution = thisProb.solve()
        if useSharedMem_:
            solution = do_fill_solve(thisProb, input)
        else:
            solution = do_fill_solve(thisProb, inputAll)
        
        
        if doThreading_:
            if not solution['status'] == 'optimal':
                if useSharedMem_:
                    print(f"Failed on \n {input} \n\n with \n {solution}")
                else:
                    print(f"Failed on \n {inputAll} \n\n with \n {solution}")
                outQueue.put("")
        else:
            if not (solution['status'] == 'optimal'):
                print(solution)
                if coreOptions.doPlot:
                    print("Error in solving")
                    import plotting as plt
                    ff,aa = plt.plt.subplots(1,1)
                    aa.set_xlim(-2,2)
                    aa.set_ylim(-2,2)
                    plt.plot2dCstr(thisProb, aa, {'binaryPlot':True}, fig=ff)
                raise RuntimeError('non-optimal solution')
        try:
            extraction = thisProb.extractOptSol(solution)
        except:
            print('a')
            extraction = thisProb.extractOptSol(solution)

        if ('toUnitCircle' in input.keys()) and (input['toUnitCircle']):
            # Add the unscaled solution
            raise RuntimeError("This is no longer supported, transformation is done in main prog")
        else:
            ySol = extraction[0]
        
        
        if __debug__:
            ySol = extraction[0]
            if ySol is None:
                print("What the hell")
                import plotting as plot
                ff, aa = plot.plt.subplots(1, 1)
                extraction = thisProb.extractOptSol(solution)
                aa.set_xlim(-2, 2)
                aa.set_ylim(-2, 2)
                xx, yy, XX = plot.ax2Grid(aa, 100, True)
                ZZ = thisRepr.evalAllMonoms(XX)
                thisPoly = poly.polynomial(thisRepr)
                for acstr in inputAll['cstr']:
                    thisPoly.coeffs = acstr
                    z = thisPoly.eval2(ZZ).reshape((100,100))
                    aa.contour(xx,yy,z, levels=[-0.1, 0., 0.01])
                
            zSol = thisRepr.evalAllMonoms(xSol)
            for k, (acstr_coeffs, acstr) in enumerate(zip(inputAll['cstr'], thisProb.constraints.s.cstrList)):
                is_valid = acstr.isValid(zSol)
                if not nall(is_valid):
                    raise RuntimeError
                try:
                    cstr_val = acstr.poly.eval2(xSol)
                    if nany(cstr_val<-coreOptions.absTolCstr):
                        raise RuntimeError
                except AttributeError:
                    pass
                
            if solution['primal objective'] < coreOptions.numericEpsPos:
                print(f"Found critical point with {solution['primal objective']} at \n {ySol}")
            print(f"Optimal value is ")
            if extraction[0].size == 0:
                thisProb.extractOptSol(solution)

        outQueue.put({'probDict':input, 'xSol':extraction[0], 'ySol':ySol, 'sol':solution, 'ext':extraction})

    return 0


def refine_solution(old_sol:dict, old_extract, old_prob:rel.convexProg, input:dict):

    dims,deg_prob = old_prob.repr.nDims, old_prob.repr.maxDeg
    # Refine
    deg_prob += 2

    dims_deg = (dims, deg_prob)

    if useSharedMem_:
        raise NotImplementedError
    else:
        input['probDict']['dimsNDeg'] = dims_deg
        cstr_deg_type = input['probDict']['nCstrNDegType']
        solver = input['probDict']['solver']

    _,_,this_prob = get_or_create(dims_deg, cstr_deg_type, solver, id_str="Refiner ")

    do_fill(this_prob, input)

    # TODO use the last extraction to reduce monomial base

    return  this_prob, this_prob.solve()



def workerSolveVariable(inQueue, outQueue):

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


        # TODO here is just a dirty fix for the degrees
        inputAll['probDict']['dims'] = inputAll['probDict']['dimsNDeg'][0]
        del inputAll['probDict']['dimsNDeg']


        #Figure out the needed degree
        try:
            # If the keyword is given then this
            dims, deg_prob = input['dimsNDeg']
        except KeyError:
            if not useSharedMem_:
                
                dims = input['dims']
                deg_min_relax = input.get('deg_relax_min', 2)
                
                deg_in = poly.get_max_degree(dims, inputAll['obj'])
                deg_prob = deg_in
                
                for ((a_deg, a_type), a_cstr) in zip(inputAll['probDict']['nCstrNDegType'], inputAll['cstr']):
                    if a_type == 's':
                        a_cstr_deg = poly.get_max_degree(dims, a_cstr)
                        assert a_cstr_deg <= a_deg
                        
                        deg_in = max(deg_in, a_cstr_deg)
                        deg_prob = max(deg_prob, np.ceil(a_cstr_deg/2.)*2 + deg_min_relax)
                    else:
                        raise NotImplementedError

                input['dimsNDeg'] = [dims, deg_prob]
            else:
                raise NotImplementedError

        thisRepr, thisRelax, thisProb = get_or_create(input['dimsNDeg'], input['nCstrNDegType'], solver = input['solver'], id_str=f"Worker {selfNr} ")

        # Actually solve
        #solution = thisProb.solve()
        solution = do_fill_solve(thisProb, input if useSharedMem_ else inputAll)


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
                print("Solution is")
                print(solution)
                print("Problem is")
                print(thisProb)
                if coreOptions.doPlot:
                    import plotting as plt
                    ff, aa = plt.plt.subplots(1, 1)
                    aa.set_xlim(-2, 2)
                    aa.set_ylim(-2, 2)
                    plt.plot2dCstr(thisProb, aa, {'binaryPlot':True}, fig=ff)
                raise RuntimeError("Non optimal solution")


        while True:
            try:
                extraction = thisProb.extractOptSol(solution)
            except:
                if __debug__:
                    print('a')
                    extraction = thisProb.extractOptSol(solution)
                else:
                    raise RuntimeError

            if extraction[0] is not None:
                # Extraction successful -> all done
                break
            # refine_solution(old_sol:dict, old_extract, old_prob:rel.convexProg, input:dict):
            try:
                thisProb, solution = refine_solution(solution, extraction, thisProb, input if useSharedMem_ else inputAll)
            except:
                thisProb, solution = refine_solution(solution, extraction, thisProb, input if useSharedMem_ else inputAll)
                print('A')
            thisRepr = thisProb.repr

        # Save the final relaxation size
        solution['dimsNDeg'] = (thisProb.repr.nDims, thisProb.repr.maxDeg)


        if ('toUnitCircle' in input.keys()) and (input['toUnitCircle']):
            # Add the unscaled solution
            raise NotImplementedError('Moved to main program')
        else:
            ySol = extraction[0]

        #Testing
        if __debug__:
            xSol = extraction[0]
            if xSol is None:
                print("What the hell")
                import plotting as plot
                ff, aa = plot.plt.subplots(1, 1)
                extraction = thisProb.extractOptSol(solution)
                aa.set_xlim(-2, 2)
                aa.set_ylim(-2, 2)
                xx, yy, XX = plot.ax2Grid(aa, 100, True)
                ZZ = thisRepr.evalAllMonoms(XX)
                thisPoly = poly.polynomial(thisRepr)
                for acstr in inputAll['cstr']:
                    thisPoly.coeffs = acstr
                    z = thisPoly.eval2(ZZ).reshape((100, 100))
                    aa.contour(xx, yy, z, levels=[-0.1, 0., 0.01])

            zSol = thisRepr.evalAllMonoms(xSol)
            for k, (acstr_coeffs, acstr) in enumerate(zip(inputAll['cstr'], thisProb.constraints.s.cstrList)):
                is_valid = acstr.isValid(zSol, atol=-1e-4)
                if not nall(is_valid):
                    raise RuntimeError
                try:
                    cstr_val = acstr.poly.eval2(xSol)
                    if nany(cstr_val < -1.e-4):
                        raise RuntimeError
                except AttributeError:
                    pass

            if solution['primal objective'] < coreOptions.numericEpsPos:
                print(f"Found critical point with {solution['primal objective']} at \n {ySol}")
            print(f"Optimal value is ")
            if extraction[0].size == 0:
                thisProb.extractOptSol(solution)

        outQueue.put({'probDict':input, 'xSol':extraction[0], 'ySol':ySol, 'sol':solution, 'ext':extraction})

    return 0


#workerSolve = workerSolveFixed
workerSolve = workerSolveVariable








