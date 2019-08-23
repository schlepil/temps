from coreUtils.coreUtilsImport import Callable, Iterable, List
from coreUtils.dbg import *
from copy import deepcopy as dp



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

def rescaleLinCtrlDict(ctrlDict:dict, scale=1., doDeepCopy=True):
    """
    if scale is a scalar -> rescale all linear control laws with this
    if scale is a array -> rescale individually
    :param ctrlDict:
    :param scale:
    :param doDeepCopy:
    :return:
    """


    if doDeepCopy:
        ctrlDict = dp(ctrlDict)

    isArrayOrDict = hasattr(scale, "__getitem__")

        
    for aInKey, aCoeffDict in ctrlDict.items():
        try:
            # Rescale the linear input
            # Attention only correct of 'projection' is 'sphere'
            int(aInKey)  # Ensure that it is dynamics
            assert isinstance(aCoeffDict, dict)
            if isArrayOrDict:
                aCoeffDict[2] *= scale[aInKey]
            else:
                aCoeffDict[2] *= scale
        except (KeyError, AssertionError, TypeError, ValueError):
            if dbg__1:
                print(f"Passed on {aInKey} for rescaling")
        except IndexError:
            print('No Indexable features should be addressed')
            assert 0
    
    return ctrlDict