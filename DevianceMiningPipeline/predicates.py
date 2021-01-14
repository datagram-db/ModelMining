from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, XAttributeContinuous
from opyenxes.factory.XFactory import XFactory

def hasAttributeWithValue(x, attribute, value):
    if attribute in x.get_attributes():
        return x.get_attributes()[attribute].get_value() == value
    else:
        return False

def compileAttributeWithValue(attribute, value):
    return lambda x: hasAttributeWithValue(x, attribute, value)

def extractAttributeValues(x, attribute):
    if attribute in x.get_attributes():
        return x.get_attributes()[attribute].get_value()
    else:
        return None

def logTagger(log, predicate):
    for trace in log:
        trace.get_attributes()["label"] = XFactory.create_attribute_literal("label","1" if predicate(trace) else "0")