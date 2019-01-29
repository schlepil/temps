from coreUtils import *

def importDynFunc(name):
    import importlib.util
    import sys
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError("can't find the {0} module".format(name))
    else:
        # If you chose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Adding the module to sys.modules is optional.
        sys.modules[name] = module
    
    return module.linCoordChange


@njit(cache=True)
def linChangeNumba(firstIdxPerMonom, cIdx,cpIdx,multiplier,numElem,idxList,c,cp,Aflat):
    
    nMonoms = firstIdxPerMonom.size
    
    idxListShape0 = idxList.shape[0]
    idxListShape1 = idxList.shape[1]
    
    #Loop over old coeffs/monoms
    for k in range(nMonoms):
        tmpVal = c[cIdx[k]]
        if tmpVal != 0:
            #old monomial has nonzero coeffs -> take into account
            startIdx = firstIdxPerMonom[k]
            if k+1 == nMonoms:
                #Last monomial loop till end
                stopIdx = idxListShape1
            else:
                # Other -> Loop till nex
                stopIdx = firstIdxPerMonom[k+1]
            
            for j in range(startIdx, stopIdx):
                # Do the actual computation
                tmpVal2 = c[cIdx[k]]*multiplier[j]
                for i in range(numElem[j]):
                    tmpVal2 *= Aflat[idxList[i,j]]
                cp[cpIdx[j]] += tmpVal2
    
    return cp
    
    


@njit
def list2int(anArr:np.ndarray, digits:int=1):
    """
    Computes an integer corresponding to the list of exponents.
    digits determines the maximally occuring degree: digits=1 -> maximal degree = 9
    [2,6,4] -> 264 with digits=1
    [2,6,4] -> 020604 with digits=2
    
    :param anArr: array representing the exponents
    :type anArr: np.ndarray
    :param digits: Determines the maximal degree
    :type digits: int
    :return: int representing the components
    :rtype: int
    """
    
    out = 0
    multi0 = 1
    multi1 = 10**digits
    
    for k in range(anArr.size-1,-1,-1):
        out += anArr[k]*multi0
        multi0 *= multi1
    
    return out


@njit
def int2list(aInt:int, nVars:int, digits:int=1, out:np.ndarray=None)->np.ndarray:
    """
    Inverse operation of list2int
    :param aInt:
    :param nVars:
    :param digits:
    :param out:
    :return:
    """
    if out is None:
        out = np.empty((nVars,),dtype=nint)
    
    for k in range(nVars-1,0-1,-1):
        out[k],aInt = divmod(aInt,10**(digits*k))
    
    return out

def getListOfMonoms(nVars:int, maxDeg:int, digits:int=1)->List[np.array]:
    
    listOfMonomials = [[] for _ in range(maxDeg+1)]
    listOfMonomials[0] = [[0 for _ in range(nVars)]]
    
    for deg in range(1,maxDeg+1):
        for aMonom in listOfMonomials[deg-1]:
            for j in range(nVars):
                newMonom = dp(aMonom)
                newMonom[j] += 1
                if not newMonom in listOfMonomials[deg]:
                    listOfMonomials[deg].append(newMonom)
    
    # Transform to np
    fTrans = lambda aMonom:np.array(aMonom,dtype=nint)
    listOfMonomials = [ lmap(fTrans, aMonomList) for aMonomList in listOfMonomials ]
    
    return listOfMonomials


def compLinChangeList(nDims:int, digits:int, aMonomList:np.ndarray):
    
    coefMat0 = sparse.dok_matrix((nDims+1,nDims+1), dtype=nint)
    
    if sum(aMonomList) == 0:
        thisList = [(0,coefMat0)]
        thisList[0][1][0,0] += 1
    else:
        thisList = [[0,dp(coefMat0)]] #Start
        for i,aExp in enumerate(aMonomList):
            #For each exponent multiply all existing monomials with the new linear combination
            #corresponding to x[i,0]
            for _ in range(aExp):
                thisListOld = thisList
                thisList = []
                for j in range(nDims):
                    # Single exponent as int
                    sExpInt = 10**(digits*(nDims-j-1)) # exponent of the jth new dim as integer
                    for oldExp,oldValue in thisListOld:
                        newExp = oldExp+sExpInt
                        newValue = cp(oldValue)
                        newValue[i+1,j+1] += 1 # i+1,j+1 necessary due to the addition of the constant propagation
                        thisList.append([newExp,newValue])
                        
    return thisList
    
def linChangeList2Dict(linChangeList):
    
    thisDict = {}
    
    for origMonom, aVal in linChangeList:
        for newMonom, aIdx in aVal:
            try:
                thisDict[newMonom].append((origMonom, aIdx))
            except KeyError:
                thisDict[newMonom] = [(origMonom,aIdx)]
    
def List2Str(nDims, maxDeg, digits, monom2num, num2monom, linChangeDict, A='A', c='cIdx', cp='cp'):
    
    from os import path
    
    fileAsList = ["from numba import njit", "", "@njit", "def linCoordChange({0},{1}".format(c,cp)]
    
    for k in range(0,maxDeg+1):
        fileAsList[-1] += ", {0}{1:d}".format(A,k)
    fileAsList[-1] += "):"
    fileAsList.append("")
    
    # TODO
    # Combine valuations to increase efficiency -> First build up a dict and count occurences
    # Even better -> this is surely following some binomial thingy -> get directly this
    
    newStyleDict = {'firstIdxPerMonomial':[0], 'cIdx':[], 'cpIdx':np.zeros((0,), dtype=nintu), 'multiplier':np.zeros((0,), dtype=nintu), 'numElem':np.zeros((0,), dtype=nintu), 'idxList':np.zeros((maxDeg,0), dtype=nintu)}
    idxMat = np.arange((nDims+1)**2).reshape((nDims+1,nDims+1)).astype(nintu)
    
    for monomOrig, newVal in linChangeDict.items():
        #Start new if-clause for each monomial to check if 0.
        fileAsList.append("\ttempVal={0}[{1}] # coef of {2}".format(c, monom2num[monomOrig], monomOrig))
        fileAsList.append("\tif tempVal!=0.:")
        thisCounterDict = {}
        for monomNew, coeffs in newVal:
            monomNewInt = monom2num[monomNew]
            try:
                isinstance(thisCounterDict[monomNewInt], dict)  # Test if existing
            except KeyError:
                thisCounterDict[monomNewInt] = {}
            thisSubDict = thisCounterDict[monomNewInt]
            #Loop over idx and build  powerMat which can be used as key
            powerMat = nzeros((nDims+1, nDims+1),dtype=nint)
            for idx,power in coeffs.items():
                powerMat[idx[0],idx[1]] += power
            powerKey = tuple(powerMat.flatten())
            try:
                thisSubDict[powerKey] += 1
            except KeyError:
                thisSubDict[powerKey] = 1
            
        # Now make it a string
        for monomNewInt, thisSubDict in thisCounterDict.items():
            for powerKey, counter in thisSubDict.items():
                fileAsList.append("\t\t{0}[{1}] += tempVal*{2:.1f}*(".format(cp,monomNewInt,counter))
                powerMat = narray(powerKey, dtype=nint).reshape((nDims+1,nDims+1))
                for i in range(nDims+1):
                    for j in range(nDims+1):
                        if powerMat[i,j] != 0:
                            fileAsList[-1] += "{0}{1:d}[{2:d},{3:d}]*".format(A,powerMat[i,j],i,j)
                # Cut last *
                fileAsList[-1] = fileAsList[-1][:-1]
                fileAsList[-1] += ") #{0}".format(num2monom[monomNewInt])
        #Also convert it to the new style -> drop exponentiation beforehand
        thisLen = 0
        for _,thisSubDict in thisCounterDict.items():
            thisLen += len(thisSubDict.keys())
            
        newStyleDict['firstIdxPerMonomial'].append(newStyleDict['firstIdxPerMonomial'][-1]+thisLen)
        newStyleDict['cIdx'].append(monom2num[monomOrig])
        #new Matrices
        newCP = nzeros((thisLen,), dtype=nintu)
        newMultiplier = nzeros((thisLen,), dtype=nintu)
        newNumElem = nzeros((thisLen,), dtype=nintu)
        newIdxList = nzeros((maxDeg,thisLen), dtype=nintu)
        
        k=0
        for monomNewInt, thisSubDict in thisCounterDict.items():
            for powerKey, counter in thisSubDict.items():
                newCP[k] = monomNewInt
                newMultiplier[k] = counter
                powerMat = narray(powerKey,dtype=nint).reshape((nDims+1,nDims+1))
                for i in range(nDims+1):
                    for j in range(nDims+1):
                        for _ in range(powerMat[i,j]):
                            newIdxList[newNumElem[k], k] = idxMat[i,j]
                            newNumElem[k] += 1
                k+=1

        newStyleDict['cpIdx'] = np.hstack([newStyleDict['cpIdx'], newCP])
        newStyleDict['multiplier'] = np.hstack([newStyleDict['multiplier'],newMultiplier])
        newStyleDict['numElem'] = np.hstack([newStyleDict['numElem'],newNumElem])
        newStyleDict['idxList'] = np.hstack([newStyleDict['idxList'],newIdxList])
    
    fileAsList.append("\treturn {0}".format(cp))
    
    pyFileName = "helper_{0:d}_{1:d}_{2:d}_numba".format(nDims, maxDeg, digits)
    base, _ = path.split(__file__)
    
    with open(path.join(base, pyFileName+".py"), "w+") as pyFile:
        pyFile.write("\n".join(fileAsList))
    
    newStyleDict['firstIdxPerMonomial'] = narray(newStyleDict['firstIdxPerMonomial'][:-1], dtype=nintu)
    newStyleDict['cIdx'] = narray(newStyleDict['cIdx'],dtype=nintu)
        
    return fileAsList, pyFileName, newStyleDict



class polynomialRepr():
    def __init__(self, nDims:int=None, maxDegree:int=None, nbrVar0:int=0, file:str=None, digits:int=1, emptyClass:bool=False):
        
        if not emptyClass:
            if file is not None:
                self.fileStr = file
                self.loadFile(file)
            else:
                self.nDims = nDims
                self.maxDeg = maxDegree
                self.digits = digits
                self.__nbrVar0 = nbrVar0
                
                self.linChangeVar = ['A','c','cp']
    
                self.linCoordChangeFun = None
                
                self.compute()
                
                self.fileStr = None #TBD
    
    @property
    def nbrVar0(self):
        return self.__nbrVar0
    
    @nbrVar0.setter
    def nbrVar0(self, new0:int):
        assert isinstance(new0,int)
        assert new0>=0
        
        self.generateDict()
        return None
    
    def __copy__(self):
        #Copies share monoms but not nbrVars
        new = polynomialRepr(emptyClass=True)
        new.__dict__.update(self.__dict__)
        return new
    
    def __deepcopy__(self, memodict={}):
        return polynomialRepr(self.nDims, self.maxDegree, self.nbrVar0, self.fileStr, self.digits)
        
    
    def generateDict(self):
        self.nMonoms = len(self.listOfMonomials)

        self.varNums = np.arange(self.nbrVar0,self.nbrVar0+self.nMonoms,dtype=nint)

        # Setup fast access.
        # Fastest (by far) method seems to be using standard dicts...

        self.monom2num = dict(zip(self.listOfMonomialsAsInt,self.varNums))
        self.num2monom = dict(zip(self.varNums,self.listOfMonomialsAsInt))
        
        return None
    
    def compute(self):
        
        self.listOfMonomialsPerDeg = getListOfMonoms(self.nDims,self.maxDeg,self.digits)
        self.listOfMonomials = []
        for aList in self.listOfMonomialsPerDeg:
            self.listOfMonomials.extend(aList)
        
        # get the integer reps
        self.listOfMonomialsAsInt = narray(lmap(list2int,self.listOfMonomials)).astype(nint)
        
        self.generateDict()
        
        self.precompLinearChange()
        
        return None
    
    def precompLinearChange(self):
        
        self.linChangeList = {}
        
        for aMonomInt, aMonomList in zip(self.listOfMonomialsAsInt,self.listOfMonomials):
            self.linChangeList[aMonomInt] = compLinChangeList(self.nDims,self.digits,aMonomList)

        self.linChangeListStr,self.linChangeModName, self.newStyleDict = List2Str(self.nDims,self.maxDeg,self.digits,self.monom2num, self.num2monom,self.linChangeList,*self.linChangeVar)
        
        #self.linCoordChangeFun = importDynFunc("polynomial."+self.linChangeModName)
        
        return None
    
    def doLinCoordChange(self, c:np.ndarray, A:np.ndarray):
        
        # dtype
        A = nrequire(A, dtype=nfloat)
        c = nrequire(c, dtype=nfloat)
        
        #Ensure size
        # adjust for constant term
        if A.shape==(self.nDims,self.nDims):
            Ahat = nidentity(self.nDims+1)
            Ahat[1:,1:] = A
        elif A.shape==(self.nDims+1,self.nDims+1):
            Ahat=A
        else:
            raise TypeError
        
        c.resize((self.nMonoms,))
        cp = nzeros((self.nMonoms,), dtype=nfloat)
        
        #Build up
        #AhatL = [np.power(Ahat,k) for k in range(self.maxDeg+1)] #Attention if A**2 -> A[i,j] = A[i,j]**2 if array, A**2 = A*A if A is matrix
        #call
        #cp = self.linCoordChangeFun(np.copy(c),cp,*AhatL)

        Ahatflat = Ahat.flatten()
        cp = linChangeNumba(self.newStyleDict['firstIdxPerMonomial'],self.newStyleDict["cIdx"],self.newStyleDict["cpIdx"],self.newStyleDict["multiplier"],self.newStyleDict["numElem"],self.newStyleDict["idxList"],c,cp,Ahatflat)
        return cp
        
        
        
        
            
        
    
    
if __name__ == "_main_":
    print("Polynomial utils")
