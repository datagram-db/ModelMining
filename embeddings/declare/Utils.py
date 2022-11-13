from abc import ABC, abstractmethod
from copy import copy, deepcopy

import pandas as pd

from dataloading.Log import TracePositional, Log

class DeclareConstruct(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, arg):
        pass

    def toArray(self, log1, log2):
        assert isinstance(log1, Log)
        assert isinstance(log2, Log)
        ls = []
        for i in range(log1.getNTraces()):
           ls.append( self.__call__(log1.getIthTrace(i)))
        for i in range(log2.getNTraces()):
           ls.append( self.__call__(log2.getIthTrace(i)))
        return pd.array(ls, dtype=pd.SparseDtype("int", 0))

class DeclareUnary(DeclareConstruct):
    def __init__(self, name, n, arg1):
        super().__init__(name)
        self.n = n
        self.arg1 = arg1

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def instantiateWith(self, dc):
        assert isinstance(dc, DeclareCandidate)
        this = deepcopy(self)
        this.arg1 = dc.args[0]
        return this

    @abstractmethod
    def __call__(self, arg):
        pass

    def __str__(self):
        return self.name + str(self.n) + "(" + self.arg1 + ")"


class DeclareBinary(DeclareConstruct):
    def __init__(self, name, arg1, arg2):
        super().__init__(name)
        self.arg1 = arg1
        self.arg2 = arg2

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def instantiateWith(self, dc):
        assert isinstance(dc, DeclareCandidate)
        this = deepcopy(self)
        this.arg1 = dc.args[0]
        this.arg2 = dc.args[1]
        return this

    @abstractmethod
    def __call__(self, arg):
        pass

    def __str__(self):
        return self.name + "(" + self.arg1 + "," + self.arg2 + ")"

class DeclareCandidate:
    def __init__(self, args):
        self.args = list(args)

    def __hash__(self):
        return hash(tuple(self.args))

    def __len__(self):
        return len(self.args)
