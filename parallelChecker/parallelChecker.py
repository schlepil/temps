from coreUtils import *

import relaxations as rel
import polynomial as poly


from parallelChecker.parallelDefinitions import *
from multiprocessing import Process, SimpleQueue
from multiprocessing.sharedctypes import RawArray as mpRawArray
import ctypes


cfloat = np.ctypeslib.as_ctypes_type(nfloat)

# The shared variables for faster com
quadraticLyapShared_ = [ mpRawArray(cfloat, lenBuffer_) for _ in range(nThreads_)] # Assuming x'.P.x <= 1.
polyObjShared_ = [ mpRawArray(cfloat, lenBuffer_) for _ in range(nThreads_)]
polyCstrShared_ = [ [mpRawArray(cfloat, lenBuffer_) for _ in nCstrMax_] for _ in range(nThreads_) ]


# inputDict
# {'dimsNDeg':(nDims,nDeg)
# {'nCstrNDegType':(nCstr,deg0,deg1,...)

def workerSolve(inQueue, outQueue):

    relaxationDict = {}
    problemDict = {}

    input = inQueue.get()

    assert 'initial' in input.keys()
    assert 'nr' in input.keys()
    assert 'solver' in input.keys()

    selfNr = input['nr']
    selfSolver = input['solver']

    while True:

        input = inQueue.get()

        if input == "":
            print(f"Worker {selfNr} is terminating")
            break

        try:
            thisRepr = relaxationDict[input['dimsNDeg']]
            thisProb = problemDict[input['dimsNDeg']][input['nCstrNDegType']] # TODO reduce memory footprint by making it order invariant

            thisProb.objective = np.frombuffer(polyObjShared_[selfNr], nfloat, thisRepr.nMonoms)

            counters = {'l':0, 'q':0, 's':1 }
            for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType'][1:]):
                assert cstrType in counters.keys()
                if cstrType == 'l':
                    raise NotImplemented
                elif cstrType == 'q':
                    raise NotImplemented
                else:
                    thisCstr = thisProb.constraints.cstrList[counters[cstrType]]
                    assert thisCstr.polyDeg == nDeg
                    thisCstr.coeffs = np.frombuffer(polyCstrShared_[selfNr][k], nfloat, thisRepr.nMonoms)
                    assert thisCstr.polyDeg == nDeg
                    counters[cstrType] += 1

        except KeyError:
            nDims, maxDeg = input['dimsNDeg']

            try:
                thisRepr = relaxationDict[input['dimsNDeg']]
            except KeyError:
                thisRepr = poly.polynomialRepr(nDims, maxDeg)
                relaxationDict[input['dimsNDeg']] = thisRepr

            # Get all polynomials
            # objective
            polyObj = poly.polynomial(thisRepr, coeffs=np.frombuffer(polyObjShared_[selfNr], nfloat, thisRepr.nMonoms), alwaysFull=True)

            # Add the objective
            thisProb = rel.convexProg(thisRepr, selfSolver, objective=polyObj)

            # constraints
            thisProb.addCstr( rel.lasserreRelax(thisRepr) )
            for k, (nDeg, cstrType) in enumerate(input['nCstrNDegType'][1:]):
                thisCstr = rel.lasserreConstraint(thisRepr, poly.polynomial(thisRepr, coeffs=np.frombuffer(polyCstrShared_[selfNr][k], nfloat,
                                                                           thisRepr.nMonoms), alwaysFull=True))
                assert (nDeg == thisCstr.maxDeg()), "Incompatible degrees"
                # Add the constraints
                thisProb.addCstr(thisCstr)


        # Check if need to transform
        if ('toUnitCircle' in input.keys()) and (input['toUnitCircle']):
            nDims = input['dimsNDeg'][0]
            P = np.frombuffer(quadraticLyapShared_[selfNr], nfloat, nDims**2).copy().reshape((nDims, nDims))
            C = cholesky(P, lower=False)
            Ci = inv(C)
            thisProb.objective.coeffs = thisRepr.doLinCoordChange(thisProb.objective.coeffs, Ci)

            assert thisProb.constraints.l.nCstr == 0, 'TODO' #TODO
            assert thisProb.constraints.q.nCstr == 0, 'TODO' #TODO

            for k in range(1,thisProb.constraints.q.nCstr):
                thisProb.constraints.q.cstrList[k].poly.coeffs = thisRepr.doLinCoordChange(thisProb.constraints.q.cstrList[k].poly.coeffs, Ci)

        # Actually solve
        solution = thisProb.solve()
        assert solution['status'] == 'optimal'
        extraction = thisProb.extractOptSol(solution)

        if ('toUnitCircle' in input.keys()) and (input['toUnitCircle']):
            # Add the unscaled solution
            ySol = ndot(C, extraction[0])

        else:
            ySol = extraction[0]

        outQueue.put({'xSol':extraction[0], 'ySol':ySol, 'sol':solution, 'ext':extraction})

    return 0

# Set up the queues and workers
probQueues = [SimpleQueue for _ in range(nThreads_)]
solQueues = [SimpleQueue for _ in range(nThreads_)]
allWorkers = []
for k in range(nThreads_):
    allWorkers.append( Process(target=workerSolve, args=(probQueues[k], solQueues[k])) )
    allWorkers[-1].deamon = True
    allWorkers[-1].start()





















