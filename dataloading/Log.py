import numpy as np
import pandas as pd
from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, XAttributeContinuous
import dateparser
import datetime
import itertools

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

    def addEventAtPositionalTrace(self, label, pos):
        if label not in self.positional_events:
            self.positional_events[label] = list()
        self.positional_events[label].append(pos)

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

    def IA_baseline_embedding(self, shared_activities=None, label=0):
        if shared_activities is None:
            shared_activities = self.unique_events
        train_data = list()
        clazz = [label] * self.getNTraces()
        for i in range(self.getNTraces()):
            trace = self.getIthTrace(i)
            clazz.append(label)
            train_data.append(list(map(lambda x : trace.eventCount(x), shared_activities)))
        np_train_data = np.array(train_data)
        train_df = pd.DataFrame(np_train_data)
        train_df.columns = shared_activities
        train_df["Class"] = clazz
        return train_df

    def DatalessTracewise_embedding(self, maxTraceLength=-1, shared_activities=None, label=0):
        if shared_activities is None:
            shared_activities = self.unique_events
        if maxTraceLength == -1:
            maxTraceLength = self.max_length
        beta = dict()
        train_data = list()
        clazz = [label] * self.getNTraces()
        for idx, x in enumerate(shared_activities):
            beta[x] = idx
        for i in range(self.getNTraces()):
            space = [0] * len(maxTraceLength)
            for idx, x in enumerate(self.getIthTrace(i).getStringTrace()):
                space[idx] = beta[x]
            train_data.append(space)
        np_train_data = np.array(train_data)
        train_df = pd.DataFrame(np_train_data)
        train_df.columns = list(map(lambda x: str(x), range(maxTraceLength)))
        train_df["Class"] = clazz
        return train_df

    def Correlation_embedding(self, shared_activities=None,  lambda_=0.9,label=0):
        if shared_activities is None:
            shared_activities = self.unique_events
        embedding_space = list(shared_activities) + list(map(lambda x: x[0]+"-"+x[1], itertools.product(shared_activities, shared_activities)))
        embedding_name_to_offset = dict()
        train_data = list()
        clazz = [label] * self.getNTraces()
        for idx, x in enumerate(embedding_space):
            embedding_name_to_offset[x] = idx
        for i in range(self.getNTraces()):
            space = [0] * len(embedding_space)
            trace = self.getIthTrace(i).getStringTrace()
            space[embedding_name_to_offset[trace[0]]] = 1
            for j, y in enumerate(trace):
                k = j
                for z in trace[j:]:
                    spaceIdx = embedding_name_to_offset[y+"-"+z]
                    space[spaceIdx] = space[spaceIdx] + pow(lambda_, k-j+1)
                    k = k+1
            train_data.append(space)
        np_train_data = np.array(train_data)
        train_df = pd.DataFrame(np_train_data)
        train_df.columns = embedding_space
        train_df["Class"] = clazz
        return train_df







