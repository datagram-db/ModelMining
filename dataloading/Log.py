from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, XAttributeContinuous
import dateparser

def get_attribute_type(val):
    if isinstance(val, XAttributeLiteral.XAttributeLiteral):
        return "literal"
    elif isinstance(val, XAttributeBoolean.XAttributeBoolean):
        return "boolean"
    elif isinstance(val, XAttributeDiscrete.XAttributeDiscrete):
        return "discrete"
    elif isinstance(val, XAttributeTimestamp.XAttributeTimestamp):
        return "timestamp"
    elif isinstance(val, XAttributeContinuous.XAttributeContinuous):
        return "continuous"

def extract_attributes(event, attribs=None):
    if not attribs:
        #attribs = ["concept:name", "lifecycle:transition"]
        attribs = ["concept:name"]
    extracted = {}
    event_attribs = event.get_attributes()
    for att in attribs:
        extracted[att] = event_attribs[att].get_value()
    return extracted


class EventPayload:
    def __init__(self, attributes, toExclude=None):
        self.attribute_type = {}
        self.trace_data = {}
        self.size = 0
        if toExclude is None:
            toExclude = {"concept:name", "Label", "lifecycle:transition"}
        for key, val in attributes:
            if key not in toExclude:
                if (key == "time:timestamp"):
                    self.size = self.size + 1
                    self.attribute_type[key] = "continuous"
                    self.trace_data[key] = str(dateparser.parse(str(val)).timestamp())
                else:
                    self.size = self.size + 1
                    self.attribute_type[key] = get_attribute_type(val)
                    self.trace_data[key] = str(val)


class Event:
    def __init__(self, activityLabel, eventpayload):
        self.eventpayload = eventpayload
        self.activityLabel = activityLabel

class TracePositional:
    def __init__(self, trace, withData=False):
        trace_attribs = trace.get_attributes()
        self.trace_name = trace_attribs["concept:name"].get_value()
        self.positional_events = {}
        self.events = []
        self.length = 0
        if withData:
            payload = EventPayload(trace_attribs.items())
            event_name = "__trace_payload"
            self.events.append(Event(event_name, payload))
        for pos, event in enumerate(trace):
            self.length = self.length + 1
            event_attribs = extract_attributes(event)
            event_name = event_attribs["concept:name"]
            self.addEventAtPositionalTrace(event_name, pos)
            if withData:
                payload = EventPayload(event.get_attributes().items())
                self.events.append(Event(event_name, payload))

    def __contains__(self, key):
        return self.hasEvent(key)

    def getStringTrace(self):
        l = [None] * self.length
        for label in self.positional_events:
            for i in self.positional_events[label]:
                l[i] = label
        return l

    def getEventsInPositionalTrace(self, label):
        if label not in self.positional_events:
            return list()
        else:
            return self.positional_events[label]

    def hasPos(self, label, pos):
        if label not in self.positional_events:
            return False
        else:
            return pos in self.positional_events[label]

    def addEventAtPositionalTrace(self, label, pos):
        if label not in self.positional_events:
            self.positional_events[label] = list()
        self.positional_events[label].append(pos)

    def hasEvent(self, label):
        if label not in self.positional_events:
            return False
        else:
            return len(self.positional_events[label])>0

    def eventCount(self, label):
        if label not in self.positional_events:
            return 0
        else:
            return len(self.positional_events[label])

class Log:
    def __init__(self, path, id=0, withData=False):
        self.path = path
        log = None
        self.traces = []
        self.max_length = -1
        with open(self.path) as log_file:
            log= XUniversalParser().parse(log_file)[id]
        self.unique_events = set()
        for trace in log:
            tp = TracePositional(trace, withData=withData)
            self.max_length = max(self.max_length, tp.length)
            self.traces.append(tp)
            for event in trace:
                    self.unique_events.add(extract_attributes(event)["concept:name"])

    def getEventSet(self):
        return self.unique_events

    def getTraces(self):
        return self.traces

    def getNTraces(self):
        return len(self.traces)

    def getIthTrace(self, i):
        return self.traces[i]


from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from opyenxes.factory.XFactory import XFactory
import os

def legacy_read_XES_log(path, ithLog = 0):
    with open(path) as log_file:
        return XUniversalParser().parse(log_file)[ithLog]

def legacy_mkdirs(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

def legacy_extractLogCopy(log):
    new_log = XFactory.create_log(log.get_attributes().clone())
    new_log.get_extensions().update(log.get_extensions())
    new_log.__globalTraceAttributes = []
    new_log.__globalTraceAttributes.extend(log.get_global_trace_attributes())
    new_log.__globalEventAttributes = []
    new_log.__globalEventAttributes.extend(log.get_global_event_attributes())
    return new_log

def legacy_split_log(readPath,log_file_tagged,outputPath):
    toRead = os.path.join(readPath, log_file_tagged)
    assert os.path.isfile(toRead)
    log = legacy_read_XES_log(toRead)
    negLog = legacy_extractLogCopy(log)
    posLog = legacy_extractLogCopy(log)
    for trace in log:
        if trace.get_attributes()["Label"].get_value() == "1":
            posLog.append(trace)
        elif trace.get_attributes()["Label"].get_value() == "0":
            negLog.append(trace)
        else:
            assert False
    output_file = os.path.join(outputPath, )
    legacy_mkdirs(output_file)
    with open(os.path.join(output_file, log_file_tagged[:-4]+"_true_true.xes"), "w") as file:
        XesXmlSerializer().serialize(posLog, file)
    with open(os.path.join(output_file, log_file_tagged[:-4]+"_false_false.xes"), "w") as file:
        XesXmlSerializer().serialize(negLog, file)