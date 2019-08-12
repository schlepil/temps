from coreUtils import *
import XXX

def ffun():
    print('a')

if __name__ == "__main__":
    #import parallelChecker.parallelWorkers
    #from parallelChecker.parallelChecker import workerSolve
    #from multiprocessing import Process, Queue
    a = XXX.getP()
    a.start()
    b = np.random.rand(10)
    print(a)
    a.join()
    