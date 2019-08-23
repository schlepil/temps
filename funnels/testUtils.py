from coreUtils import *

import polynomial as poly
import relaxations as relax
import warnings

def testSol(sol, ctrlDict:dict):

    nDim, maxDeg = sol['probDict']['dimsNDeg']

    # Set up helpers
    thisRepr = poly.polynomialRepr(nDim, maxDeg)
    thisPoly = poly.polynomial(thisRepr)

    # Test if solution(s) is(are) valid
    xSol = sol['xSol'] #At
    if dbg__1:
        print(f"Minimizers are:\n{xSol}")
    zxSol = thisRepr.evalAllMonoms(xSol)
    
    try:
        PG0n = ctrlDict['PG0']/(norm(ctrlDict['PG0'], axis=0, keepdims=True)+coreOptions.floatEps)
    except KeyError:
        pass

    # Loop over constraint polynomials
    for k, acstr in enumerate(sol['origProb']['cstr']):
        thisPoly.coeffs = acstr
        for i in range(zxSol.shape[1]):
            if not thisPoly.eval2(zxSol[:,[i]])>=0.:
                thisRelax = relax.lasserreRelax(thisRepr)
                print(f"Failed on cstr {k} with {thisPoly.eval2(thisPoly.eval2(zxSol[:,[i]]))} for point {i}")
                thisCstr = relax.lasserreConstraint(thisRelax, thisPoly)
                print(f"Original sol with {sol['sol']['x_np']} is valid : {thisCstr.isValid(sol['sol']['x_np'].T, simpleEval=False)}")
                print(f"Reconstructed sol with {xSol} is valid : {thisCstr.isValid(xSol, simpleEval=False)}")
                
                print(f"Constraint is {acstr}")
                print(f"Points are {acstr}")
                
                if nany(thisCstr.isValid(sol['sol']['x_np'].T, simpleEval=False) != thisCstr.isValid(xSol, simpleEval=False)):
                    print("Reoptimized?")

    # Check if minimizer is compatible with control law definition
    if 'PG0' in ctrlDict.keys():
        dist2Planes = ndot(PG0n.T, xSol)
        signPlanes = np.sign(dist2Planes).astype(nint)
        signPlanes[signPlanes==0] = 1
        
        for i,type in enumerate(sol['probDict']['u'].reshape((-1,))):
            if (nany(signPlanes[0,:] != signPlanes[0,0])) and not type==2:
                print(f"Attention: minimizers are on different signs of the plane. Check wich separation is used")
            if not type in list(-signPlanes[i,:])+[2]:
                warnings.warn(f"Failed on input {i} with dist {dist2Planes[i]} and ctrl {type}")

    # Check if minimal value corresponds to ctrlDict value
    thisPoly.coeffs = -ctrlDict[-1][0]
    optsVals = thisPoly.eval2(zxSol).reshape((-1,))

    for i,type in enumerate(sol['probDict']['u'].reshape((-1,))):
        thisPoly.coeffs = -ctrlDict[i][type]
        for k in range(zxSol.shape[1]):
            thisVal = thisPoly.eval2(zxSol[:,[k]])
            optsVals[k] += thisVal
            if thisVal <= 0.:
                print(f"The control type {type} for input {i} failed on minimizer {k}")
    if dbg__1:
        print(f"global minimum was \n{sol['obj']}\n, computed values from ctrlDict are \n{optsVals}")



    print('a')




