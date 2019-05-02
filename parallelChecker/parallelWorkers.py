from parallelChecker.parallelChecker import *

# Set up the queues and workers
if unifiedQueues_:
    probQueues = [Queue()]
    solQueues = [Queue()]
    if doThreading_:
        allWorkers = []
        for k in range(nThreads_):
            allWorkers.append(Process(target=workerSolve, args=(probQueues[0], solQueues[0])))
            allWorkers[-1].deamon = True
            allWorkers[-1].start() #Commented for testing
else:
    probQueues = [Queue() for _ in range(nThreads_)]
    solQueues = [Queue() for _ in range(nThreads_)]
    if doThreading_:
        allWorkers = []
        for k in range(nThreads_):
            allWorkers.append( Process(target=workerSolve, args=(probQueues[k], solQueues[k])) )
            allWorkers[-1].deamon = True
            allWorkers[-1].start() #Commented for testing


if doThreading_:
    distributor = workDistributor(probQueues, solQueues)
else:
    distributor = workDistributorNoThread(probQueues, solQueues)