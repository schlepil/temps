from multiprocessing import Queue, Process
import os
import numpy as np

def replyFun(inQueue, outQueue):
    while True:
        inDict = inQueue.get(block=True)
        if (isinstance(inDict, str)) and (inDict == "term"):
            break
        try:
            inDict['pid'] += os.getpid()
        except KeyError:
            inDict['pid'] = os.getpid()
        outQueue.put(inDict)
    

if __name__ == "__main__":
    from time import time
    inputQueue = Queue()
    outputQueue = Queue()
    allProcesses = []
    for k in range(4):
        allProcesses.append(Process(target=replyFun, args=(inputQueue, outputQueue), daemon=True))
        allProcesses[-1].start()
    
    baseDict = dict( [[i, np.random.rand(100,)] for i in range(20)] )
    
    outList = [baseDict.copy() for _ in range(4)]
    T = time()
    for _ in range(100):
        for k in range(4):
            inputQueue.put(outList[k], timeout=5)
        for k in range(4):
            outList[k] = outputQueue.get(timeout=5)
    T = time()-T
    for k in range(4):
        print(outList[k])
        inputQueue.put("term", timeout=5)
    print(f"It took {T/400.} seconds per transmission loop with partial block")
    
    for aProc in allProcesses:
        aProc.join()
        aProc.close()
    
    