"""

Defines high-level function predicates for the log

@author: Giacomo Bergami <bergamigiacomo@gmail.com>
"""
from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, XAttributeContinuous
from opyenxes.factory.XFactory import XFactory
from enum import Enum

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

class SatCases(Enum):
        VacuitySat = 0
        NotSat = 1
        NotVacuitySat = 2
        Sat = 3

checkSatSwitch = {'VacuitySat': lambda x: x == 0, 'NotSat': lambda x: x<0, 'NotVacuitySat': lambda x : x>0, 'Sat': lambda x: x>=0}
def checkSat(trace, function, event_name_list, SatCheck):
    """
    Checks the satisfiability of the condition using all the parameters

    :param trace:               Trace to be tested with a Joonas' predicate
    :param function:            Function to be called for returning the predicate's value
    :param event_name_list:     Event names to be used to instantiate the predicate
    :param SatCheck:            The satisfiability condition to be checked for the predicate
    :return:
    """
    out, _  = function(trace, event_name_list)
    return checkSatSwitch[SatCheck.name](out)

class SatProp:
    """
    This class makes each Joonas' predicate as a true predicate to be called, so to be uniform
    with a 1) intuitive concept of a predicate to be checked 2) separating the parameter instantiation
    from its check 3) Provide a compact representation of the predicate as a class
    """
    def __init__(selbst, function, event_name_list, SatCheck):
        """
        :param function:            Joonas' function to be checked
        :param event_name_list:     List for instantiating the predicate with the events' names
        :param SatCheck:            Satisfiability check for the predicate's outcome
        """
        assert(callable(function))
        selbst.function = function
        assert(isinstance(event_name_list, list))
        selbst.event_name_list = event_name_list
        assert (isinstance(SatCheck, SatCases))
        selbst.SatCheck = SatCheck

    def __call__(self, trace):
        """
        Checking the predicate via a trace
        :param trace:
        :return:        Whether the predicate was satisfied using a specific satisfiability condition
        """
        return checkSat(trace, self.function, self.event_name_list, self.SatCheck)

class SatAllProp:
    """
    Definition of a class checking that all the Satisfiability predicates were satisfied
    """
    def __init__(self, props):
        for x in props:
            assert(isinstance(x, SatProp))
        self.props = props

    def __call__(self, trace):
        return all(x(trace) for x in self.props)

def logTagger(log, predicate):
    """
    Tags the log using the predicate's satisfiability
    :param log:         Log to be tagget at each trace
    :param predicate:   Predicate used to tag the sequence
    :return:
    """
    for trace in log:
        trace.get_attributes()["Label"] = XFactory.create_attribute_literal("Label","1" if predicate(trace) else "0")

def tagLogWithValueEqOverAttn(log, attn, val):
    logTagger(log, compileAttributeWithValue(attn, val))

def tagLogWithSatAllProp(log, functionEventNamesList, SatCheck):
    logTagger(log, SatAllProp([SatProp(x,y, SatCheck) for (x,y) in functionEventNamesList]))