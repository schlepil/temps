doThreading_ = False # TODO check why exploration is so different and more costly when threading
nThreads_ = 4 # Nbr of threads

nCstrMax_ = 20+1+1 #Maximal number of constraints ( here 20 inputs, spherebounded, baserelaxation )
lenBuffer_ = 1000

unifiedQueues_ = False

solver_ = 'cvxopt'

useSharedMem_ = False

printProbNSol_ = True