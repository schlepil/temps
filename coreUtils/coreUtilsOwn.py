from coreUtils.coreUtilsImport import Callable, Iterable, List

def lmap(f:Callable, l:Iterable)->List:
    return list(map(f,l))

class variableStruct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return self.__str__()


def recursiveExclusiveUpdate(dict0:dict, dict1:dict):
    """
    Updates recursively all key/value pairs in dict0 with those in dict1 without creating new ones
    :param dict0:
    :param dict1:
    :return:
    """

    keys1 = dict1.keys()

    for aKey0,aVal0 in dict0.items():
        if aKey0 in keys1:
            if (isinstance(aVal0, dict) and isinstance(dict1[aKey0], dict)):
                recursiveExclusiveUpdate(dict0[aKey0], dict1[aKey0])
            else:
                dict0[aKey0] = dict1[aKey0]

    return None
