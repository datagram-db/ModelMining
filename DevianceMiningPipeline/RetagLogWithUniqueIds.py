"""
Changing the dataset that were produced by ensuring unique ids to each trace

@author Giacomo Bergami
"""
from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, XAttributeContinuous
from opyenxes.factory.XFactory import XFactory
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
defaults = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnoprstuvwxyz0123456789"

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def generateStringFromNumber(n):
    return "".join(map(lambda x: defaults[x], numberToBase(n, len(defaults))))

def changeLog(logFileName):
    with open(logFileName) as file:
        log = XUniversalParser().parse(file)[0]
        i = 0
        for trace in log:
            trace.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name", generateStringFromNumber(i))
            i = i+1
        with open(logFileName+"_unique.xes", "w") as file2:
            XesXmlSerializer().serialize(log, file2)
            file2.close()
        with open(logFileName+"_unique.xes") as file2:
            log = XUniversalParser().parse(file2)[0]
            l = list(map(lambda x: x.get_attributes()["concept:name"].get_value(), log))
            assert(len(l) == len(set(l)))
            file2.close()
        file.close()