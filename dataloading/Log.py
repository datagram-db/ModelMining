from copy import deepcopy
from datetime import datetime
from collections import Counter
import ciso8601 as ciso8601
import pandas as pd
from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, XAttributeContinuous
import dateparser

types = {"literal", "boolean", "discrete", "timestamp", "continuous"}
TRACE_LENGTH = "@trace_len"

def dict_union(d1 : dict, d2 : dict):
    K = set.union(set(d1.keys()), set(d2.keys()))
    result = {k : set() for k in K}
    for k in K:
        if k in d1 and k in d2:
            result[k] = set.union(d1[k], d2[k])
        elif k in d1:
            result[k] = d1[k]
        else:
            result[k] = d2[k]
    return result

def typecast(type : str, val):
    if type == "literal":
        if val is None:
            return ""
        return str(val)
    elif type == "boolean":
        if val is None:
            return False
        return bool(val)
    elif type == "discrete":
        if val is None:
            return 0
        return int(val)
    elif type == "timestamp":
        if val is None or val =="":
            return datetime.MINYEAR
        return ciso8601.parse_datetime(str(val))
    elif type == "continuous":
        if val is None:
            return 0.0
        return float(val)

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
        self.keys = set(self.trace_data.keys())

    def payloadKeySet(self):
        return self.keys


class Event:
    def __init__(self, activityLabel : str, eventpayload : EventPayload):
        self.eventpayload = eventpayload
        self.activityLabel = activityLabel

    def payloadKeySet(self):
        if self.eventpayload is not None:
            return self.eventpayload.payloadKeySet()
        else:
            return set()

    def getActivityLabel(self):
        return self.activityLabel

    def getValueType(self, key):
        if self.eventpayload is None:
            return None
        elif key in self.eventpayload.trace_data:
            return self.eventpayload.attribute_type[key]
        else:
            return None

    def getValue(self, key):
        if self.eventpayload is None:
            return None
        elif key in self.eventpayload.trace_data:
            return self.eventpayload.trace_data[key]
        else:
            return None

    def hasKey(self, k):
        if self.eventpayload is None:
            return False
        if k not in self.eventpayload.keys:
            return False
        return True


class TracePositional:
    def __init__(self, trace, withData=False):
        self.declareTypeOf = {}
        trace_attribs = trace.get_attributes()
        if "concept:name" in trace_attribs:
            self.trace_name = trace_attribs["concept:name"].get_value()
        else:
            self.trace_name = "None"
        self.positional_events = {}
        self.events = []
        self.length = 0
        self.keyType = dict()
        if withData:
            payload = EventPayload(trace_attribs.items())
            event_name = "__trace_payload"
            e = Event(event_name, payload)
            self.events.append(e)
            self.keys = e.payloadKeySet()
        else:
            self.keys = set()
        for pos, event in enumerate(trace):
            self.length = self.length + 1
            event_attribs = extract_attributes(event)
            event_name = event_attribs["concept:name"]
            self.addEventAtPositionalTrace(event_name, pos)
            if withData:
                payload = EventPayload(event.get_attributes().items())
                e = Event(event_name, payload)
                self.keys = set.union(self.keys, e.payloadKeySet())
                self.events.append(e)
        for k in self.keys:
            typeInferOf = {type: 0 for type in types}
            for e in self.events:
                if e.getValueType(k) is not None:
                    typeInferOf[e.getValueType(k)] = typeInferOf[e.getValueType(k)] + 1
            self.keyType[k] =  max(typeInferOf, key=typeInferOf.get)

    def payloadKeySet(self):
        if self.keys is None:
            return set()
        else:
            return self.keys

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

    def projectWith(self, lstEvents : list[int]):
        this = deepcopy(self)
        eS = set(lstEvents)
        removeK = set()
        for k in this.positional_events:
            kS = set(this.positional_events[k]) - eS
            if len(kS) == 0:
                removeK.add(k)
            else:
                this.positional_events = list(kS)
        for k in removeK:
            this.positional_events.pop(k)

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

    def getValueType(self, k):
        if k not in self.keyType:
            return None
        else:
            return self.keyType[k]

    def collectDistinctValues(self, withTypeCast : dict[str,str], keys = None):
        if keys is None:
            keys = self.keys
        result = dict()
        for k in keys:
            values = set()
            for idx, e in enumerate(self.events):
                if e.hasKey(k):
                    val = typecast(withTypeCast[k], e.getValue(k))
                    values.add(val)
            result[k] = values
        return result

    def collectValuesForPayloadEmbedding(self, withTypeCast : dict[str,str],
                                         keys=None,
                                         occurrence = None,
                                         preserveEvents = None,
                                         keepTraceLenth = True):
        d = dict()
        if keys is None:
            keys = self.keys
        if preserveEvents is None:
            preserveEvents = set(range(self.length))
        if keepTraceLenth:
            d[TRACE_LENGTH] = self.length
        for k in keys:
            values = list()
            N = 0
            if self.events is not None:
                N = len(self.events)
            for idx, e in enumerate(self.events):
                if idx not in preserveEvents:
                    continue
                if e.hasKey(k):
                    val = typecast(withTypeCast[k], e.getValue(k))
                    d["@"+e.activityLabel+"."+k] = val
                    if idx == 0:
                        d["@first("+k+")"] = val
                    values.append(val)
                    if idx == N:
                        d["@last("+k+")"] = val
            if len(values)>0:
                d["@min("+k+")"] = min(values)
                d["@max("+k+")"] = max(values)
            counter = Counter(values)
            if keepTraceLenth:
                if occurrence is None or k not in occurrence:
                    for instance in counter:
                        d["@count("+k+"="+str(instance)+")"] = counter[instance]
                elif k in occurrence:
                    values = set.union(occurrence[k], set(counter.keys()))
                    for value in values:
                        if value in occurrence:
                            d["@count(" + k + "=" + str(value) + ")"] = counter[value]
                        else:
                            d["@count(" + k + "=" + str(value) + ")"] = 0
        return d




class Log:
    def __init__(self, path, id=0, withData=False):
        self.path = path
        log = None
        self.traces = []
        self.max_length = -1
        with open(self.path) as log_file:
            log= XUniversalParser().parse(log_file)[id]
        self.unique_events = set()
        self.keys = set()
        self.keyType = dict()
        for trace in log:
            tp = TracePositional(trace, withData=withData)
            self.keys = set.union(self.keys, tp.payloadKeySet())
            self.max_length = max(self.max_length, tp.length)
            self.traces.append(tp)
            for event in trace:
                    self.unique_events.add(extract_attributes(event)["concept:name"])
        for k in self.keys:
            typeInferOf = {type: 0 for type in types}
            for e in self.traces:
                typeInferOf[e.getValueType(k)] = typeInferOf[e.getValueType(k)] + 1
            self.keyType[k] = max(typeInferOf, key=typeInferOf.get)

    def getEventSet(self):
        return self.unique_events

    def getTraces(self):
        return self.traces

    def getNTraces(self):
        return len(self.traces)

    def getIthTrace(self, i):
        return self.traces[i]

    def payloadKeySet(self):
        return self.keys

    def getValueType(self, k):
        if k not in self.keyType:
            return None
        else:
            return self.keyType[k]

    def resolvePayload(self, key, value):
        return typecast(self.getValueType(key), value)

    def resolvePayload(self, key : str,  event : Event):
        return self.resolvePayload(key, event.getValue(key))

    def collectDistinctValues(self, ignoreKeys = None):
        if ignoreKeys is None:
            ignoreKeys = set()
        d = {k : set() for k in self.keys if k not in ignoreKeys}
        for trace in self.traces:
            d = dict_union(d, trace.collectDistinctValues(self.keyType, self.keys))
        return d

    def collectValuesForPayloadEmbedding(self,
                                         distinct_values : dict,
                                         ignoreKeys = None,
                                         preserveEvents=None,
                                         keepTraceLen = True,
                                         setNAToZero = True
                                         ):
        if ignoreKeys is None:
            ignoreKeys = set()
        if distinct_values is None or (len(distinct_values) == 0):
            distinct_values = self.collectDistinctValues(ignoreKeys)
        if preserveEvents is None:
            preserveEvents = [list(range(x.length)) for x in self.traces]
        keys = self.keys - ignoreKeys
        traceToEventsToPreserve = zip(self.traces, preserveEvents)
        df = pd.DataFrame(map(lambda x: x[0].collectValuesForPayloadEmbedding(self.keyType, keys, distinct_values, x[1], keepTraceLen), traceToEventsToPreserve))
        if setNAToZero:
            df = df.fillna(0)
        return df


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


