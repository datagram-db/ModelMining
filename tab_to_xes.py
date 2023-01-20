import sys
from opyenxes.factory.XFactory import XFactory
from opyenxes.id.XIDFactory import XIDFactory
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
import random

def convert(x,y):
    file1 = open(x, 'r')

    log = XFactory.create_log()
    count = 0

    # Using for loop
    print("Using for loop")
    for t in file1:
        trace = XFactory.create_trace()
        for activity_label in t[:-1].split("\t"):
            event = XFactory.create_event()
            attribute = XFactory.create_attribute_literal("concept:name", activity_label)
            event.get_attributes()["concept:name"] = attribute
            trace.append(event)
        log.append(trace)

    with open(y, "w") as f:
        XesXmlSerializer().serialize(log, f)

if __name__ == "__main__":
    for x in ["10_10_1000.","15_15_1000.","20_20_1000.","25_25_1000.","30_30_1000."]:
        convert(x+"tab",x+"xes")
