import os
import sys

from opyenxes.data_in.XUniversalParser import XUniversalParser

from dataloading.Log import extract_attributes

if __name__ == "__main__":
    with open(sys.argv[1], "r") as r:
        log = XUniversalParser().parse(r)[0]
        with open(sys.argv[1]+".tab","w") as f:
            N = len(log)
            for ntrace, trace in enumerate(log):
                trace_attribs = trace.get_attributes()
                tab_trace = []
                for pos, event in enumerate(trace):
                    event_attribs = extract_attributes(event)
                    event_name = event_attribs["concept:name"]
                    tab_trace.append(event_name)
                f.write("\t".join(tab_trace))
                if (ntrace<N-1):
                    f.write(os.linesep)