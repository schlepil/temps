from parallelChecker.parallelChecker import *

# Set up the queues and workers
if unifiedQueues_:
    probQueues = [Queue()]
    solQueues = [Queue()]
    allWorkers = []
    for k in range(nThreads_):
        allWorkers.append(Process(target=workerSolve, args=(probQueues[0], solQueues[0])))
        allWorkers[-1].deamon = True
        allWorkers[-1].start() #Commented for testing
else:
    probQueues = [Queue() for _ in range(nThreads_)]
    solQueues = [Queue() for _ in range(nThreads_)]
    allWorkers = []
    for k in range(nThreads_):
        allWorkers.append( Process(target=workerSolve, args=(probQueues[k], solQueues[k])) )
        allWorkers[-1].deamon = True
        allWorkers[-1].start() #Commented for testing


distributor = workDistributor(probQueues, solQueues)