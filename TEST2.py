import copy
import numpy as np
from scipy import sparse
maxDeg=3
nVars=3
listOfMonomials = [[] for _ in range(maxDeg+1)]
print(listOfMonomials)
listOfMonomials[0] = [[0 for _ in range(nVars)]]
print(listOfMonomials)
print(listOfMonomials[0])
for deg in range(1, maxDeg + 1):
    for aMonom in listOfMonomials[deg - 1]:
        print("aMon",aMonom)
        for j in range(nVars):
            newMonom = copy.deepcopy(aMonom)
            print("newMonom", newMonom)
            newMonom[j] += 1
            print("newMonom+1",newMonom)
            if not newMonom in listOfMonomials[deg]:
                listOfMonomials[deg].append(newMonom)
fTrans = lambda aMonom: np.array(aMonom, dtype=int)
listOfMonomials = [list(map(fTrans, aMonomList)) for aMonomList in listOfMonomials]
print(listOfMonomials)

def compLinChangeList(nDims: int, digits: int, aMonomList: np.ndarray):
    coefMat0 = sparse.dok_matrix((nDims + 1, nDims + 1), dtype=int)

    if sum(aMonomList) == 0:
        thisList = [(0, coefMat0)]
        thisList[0][1][0, 0] += 1
    else:
        thisList = [[0, copy.deepcopy(coefMat0)]]  # Start
        for i, aExp in enumerate(aMonomList):
            # For each exponent multiply all existing monomials with the new linear combination
            # corresponding to x[i,0]
            for _ in range(aExp):
                thisListOld = thisList
                thisList = []
                for j in range(nDims):
                    # Single exponent as int
                    sExpInt = 10 ** (digits * (nDims - j - 1))  # exponent of the jth new dim as integer
                    for oldExp, oldValue in thisListOld:
                        newExp = oldExp + sExpInt
                        newValue = copy.deepcopy(oldValue)
                        newValue[i + 1, j + 1] += 1  # i+1,j+1 necessary due to the addition of the constant propagation
                        thisList.append([newExp, newValue])

    return thisList
