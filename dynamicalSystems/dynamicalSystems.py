from coreUtils import *
from polynomial import polynomialRepr

from polynomial.utils_numba import getIdxAndParent

import string

from dynamicalSystems.derivUtils import getInverseTaylorStrings

class dynamicalSystem:
    
    def __init__(self, repr:polynomialRepr, q, u, maxTaylorDegree):
        
        if __debug__:
            assert q.shape[1] == 1
            assert u.shape[1] == 1
            assert maxTaylorDegree <= repr.maxDeg
            assert repr.nDims == q.shape[0]
        
        self.repr = repr
        
        self.q = q
        self.u = u
        
        self.nq = int(q.shape[0])
        self.nu = int(u.shape[0])
        
        self.maxTaylorDeg=maxTaylorDegree
        
    
    def getTaylorApprox(self, x:np.ndarray, maxDeg:int)->Tuple:
        raise NotImplementedError
    
    def precompute(self):
        raise NotImplementedError
    
    def __call__(self, x:np.ndarray, u:np.ndarray, mode:str='OO', x0:np.ndarray=None):
        raise NotImplementedError

class secondOrderSys(dynamicalSystem):
    
    def __init__(self, repr:polynomialRepr, massMat:sy.Matrix, f:sy.Matrix, g:sy.Matrix, q:"symbols", u:"symbols", maxTaylorDegree:int=3, file:str=None):
        """
        represents a system of the form
        q is composed of (position,velocity) -> (q_p,q_v)
        dq -> (dq_p,dq_v) = (q_v,q_a)
        q_v = dq_p
        massMat(q_p).q_a = f(q) + g(q).u
        :param repr:
        :param massMat:
        :param f:
        :param g:
        :param q:
        :param u:
        """
        
        super(secondOrderSys,self).__init__(repr,q,u,maxTaylorDegree)
        
        self.massMat = massMat
        self.f = f
        self.g = g
        
        self.nqv = self.nq//2
        
        if file == None:
            self.compTaylor()
        else:
            self.fromFile(file)
    
    def compTaylor(self):
        self.compTaylorF()
        self.compTaylorG()
        self.compTaylorM()
        if self.maxTaylorDeg<=3:
            derivStrings = string.ascii_lowercase[23:23+self.maxTaylorDeg]
        else:
            derivStrings = string.ascii_lowercase[23:23+self.maxTaylorDeg]
        self.inversionTaylor = variableStruct(**getInverseTaylorStrings('M','Mi','W',derivStrings))
        
    def compTaylorF(self):
        
        taylorFD ={ 'f0':sy.Matrix(nzeros((self.nqv,1))) }
        
        taylorFD['f0'] += self.f
        
        lastMat = self.f
        for k in range(1,self.maxTaylorDeg+1):
            thisMat = sy.Matrix(nzeros((self.nqv,len(self.repr.listOfMonomialsPerDeg[k]))))

            #Get the derivative corresponding to each monomial
            for j,aMonom in enumerate(self.repr.listOfMonomialsPerDegAsInt[k]):
                idxDeriv, idxParent = getIdxAndParent(aMonom, self.repr.listOfMonomialsPerDegAsInt[k-1], self.nq, self.repr.digits)
                
                # Derive the (sliced) vector
                thisMat[:,j] = sy.diff(lastMat[:,idxParent], self.q[idxDeriv,0])/float(k)
            
            #save
            taylorFD["f{0:d}".format(k)] = thisMat
            #Iterate
            lastMat = thisMat
            
        
        #for all 0-N
        totCols = sum([len(aList) for aList in self.repr.listOfMonomialsPerDeg[:self.maxTaylorDeg+1]])
        taylorFD["fTaylor"] = sy.Matrix(nzeros((self.nqv,totCols)))
        cCols=0
        for k in range(0,self.maxTaylorDeg+1):
            aKey = "f{0:d}".format(k)
            aVal = taylorFD[aKey]
            
            taylorFD["fTaylor"][:,cCols:cCols+aVal.shape[1]] = aVal
            cCols += aVal.shape[1]
        
        #array2mat = [{'ImmutableDenseMatrix':np.matrix},'numpy']
        array2mat = ['numpy']
        
        tempDict=dict()

        for aKey, aVal in taylorFD.items():
            tempDict[aKey+"_eval"] = sy.lambdify(self.q, aVal, modules=array2mat)
        
        taylorFD.update(tempDict)
        del tempDict
        
        self.taylorF = variableStruct(**taylorFD)
    
    def compTaylorM(self):
        """
        Taylor expansion of the mass matrix
        
        :return:
        """
        taylorMD = {'M0':sy.Matrix(nzeros((self.nqv,self.nqv)))}
    
        taylorMD['M0'] += self.massMat
        
        lastList = [self.massMat]

        for k in range(1,self.maxTaylorDeg+1):
            thisList = [sy.Matrix(nzeros((self.nqv,self.nqv))) for _ in range(self.repr.listOfMonomialsAsInt.size)]
    
            #Get the derivative corresponding to each monomial
            for j,aMonom in enumerate(self.repr.listOfMonomialsPerDegAsInt[k]):
                idxDeriv,idxParent = getIdxAndParent(aMonom,self.repr.listOfMonomialsPerDegAsInt[k-1],self.nq,self.repr.digits)
                
                thisList[j] = sy.diff(lastList[idxParent], self.q[idxDeriv,0])/float(k)
            
            #save
            taylorMD["M{0:d}".format(k)] = thisList
            #Iterate
            lastList = thisList

        taylorMD["MTaylor"] = []
        for k in range(0,self.maxTaylorDeg+1):
            aKey = "M{0:d}".format(k)
            aVal = taylorMD[aKey]
    
            taylorMD["MTaylor"].extend(aVal)

        array2mat = ['numpy']
    
        tempDict = {}
        for aKey, aVal in taylorMD.items():
            tempDict[aKey+"_eval"] = [sy.lambdify(self.q, aMat, modules=array2mat) for aMat in aVal]
        taylorMD.update(tempDict)

        self.taylorM = variableStruct(**taylorMD)

    def compTaylorG(self):
        """
        Taylor expansion of the input matrix

        :return:
        """
        taylorGD = {'G0':sy.Matrix(nzeros((self.nqv,self.nu)))}
    
        taylorGD['G0'] += self.g
    
        lastList = [self.g]
    
        for k in range(1,self.maxTaylorDeg+1):
            thisList = [sy.Matrix(nzeros((self.nqv,self.nu))) for _ in range(self.repr.listOfMonomialsAsInt.size)]

            # Get the derivative corresponding to each monomial
            for j,aMonom in enumerate(self.repr.listOfMonomialsPerDegAsInt[k]):
                idxDeriv,idxParent = getIdxAndParent(aMonom,self.repr.listOfMonomialsPerDegAsInt[k-1],self.nq,self.repr.digits)
            
                thisList[j] = sy.diff(lastList[idxParent],self.q[idxDeriv,0])/float(k)

            # save
            taylorGD["G{0:d}".format(k)] = thisList
            # Iterate
            lastList = thisList
    
        taylorGD["GTaylor"] = []
        for k in range(0,self.maxTaylorDeg+1):
            aKey = "G{0:d}".format(k)
            aVal = taylorGD[aKey]
        
            taylorGD["GTaylor"].extend(aVal)
    
        array2mat = ['numpy']
        
        tempDict = {}
        for aKey,aVal in taylorGD.items():
            tempDict[aKey+"_eval"] = [sy.lambdify(self.q,aMat,modules=array2mat) for aMat in aVal]
        taylorGD.update(tempDict)
    
        self.taylorG = variableStruct(**taylorGD)
    
    def evalTaylor(self, x:np.ndarray, maxDeg:int=None):
        
        # TODO this is a naive implementation
        
        if __debug__:
            assert (maxDeg is None) or (maxDeg <= self.maxTaylorDeg)
        
        maxDeg = self.maxTaylorDeg if maxDeg is None else maxDeg
        
        # return Value
        MifTaylor = nzeros((self.nq, self.repr.nMonoms), dtype=nfloat)
        MigTaylor = nzeros((self.repr.nMonoms,self.nq,self.nu), dtype=nfloat)
        
        # The integration of the velocity
        MifTaylor[:self.nqv, 1:1+self.nqv] = nidentity(self.nqv, dtype=nfloat)
        
        # Setup the inputs
        xList = list(x)
        # First get all taylor series of non-inversed
        fTaylor = nmatrix(self.taylorF.fTaylor(*xList)) # Pure np.matrix
        MTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorM.MTaylor] #List of matrices
        GTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorG.GTaylor] #List of matrices
        
        # Inverse the inertia matrix at the current point
        Mi = inv(MTaylor[0])
        
        # Now loop over all
        nVar = list(range(self.nq))
        nameIdxList = []
        for k,aMonom in enumerate(self.repr.listOfMonomials):
            
            for i,(aStr,aExp) in enumerate(zip(self.inversionTaylor.derivStr,aMonom)):
                for _ in range(aExp):
                    thisNameIdxList.append(aStr,i)
            # Get
            allDependentList =
        
        
        
        
        


                    
        
        
        
        
    
    
    
    
    
    
    
    