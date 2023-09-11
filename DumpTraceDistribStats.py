import os
import sys

from opyenxes.data_in.XUniversalParser import XUniversalParser


if __name__ == '__main__':
    log = None
    args = sys.argv[1:]
    ntrace = 0.0
    hist = dict()
    log = []
    with open(args[0], 'r') as file1:
        for t in file1:
            log.append(t[:-1].split("\t"))
    # with open(args[0]) as log_file:
    #     log = XUniversalParser().parse(log_file)[0]
    with open(args[2], "a") as dump_file:
        for trace in log:
            n = len(trace)
            dump_file.write(args[1]+","+str(len(trace))+os.linesep)
