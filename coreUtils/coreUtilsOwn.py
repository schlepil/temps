from coreUtils.coreUtilsImport import *

def lmap(f:Callable, l:Iterable)->List:
    return list(map(f,l))

class variableStruct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return self.__str__()
