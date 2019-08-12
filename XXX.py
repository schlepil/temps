from coreUtils import *

def ffun():
    for _ in range(100000):
        print('a')

def getP():
    from multiprocessing import Process, Queue
    a = Process(target=ffun)
    return a

if __name__ == "__main__":
    #import parallelChecker.parallelWorkers
    #import parallelChecker.parallelChecker
    b = np.random.rand(10)
    print(b)
    