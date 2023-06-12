import os.path
import sys

from dataloading.Log import *

if __name__ == '__main__':
    folder = "/home/giacomo/projects/knobab2/data/benchmarking/multimodel_mining/"
    pos = os.path.join(folder, "positive.powerdecl.xes")
    neg = os.path.join(folder, "negative.powerdecl.xes")
    logP = legacy_read_XES_log(pos)
    logN = legacy_read_XES_log(neg)
    training_set_pos = legacy_extractLogCopy(logP)
    training_set_neg = legacy_extractLogCopy(logN)
    testing_set_pos = legacy_extractLogCopy(logP)
    testing_set_neg = legacy_extractLogCopy(logN)
    testing_set_knobab = legacy_extractLogCopy(logP)
    list_class = []
    legacy_split_log(logP, training_set_pos, testing_set_pos, 1, None, list_class, 0.7)
    legacy_split_log(logN, training_set_neg, testing_set_neg, 0, None, list_class, 0.7)
    for trace in testing_set_pos:
        testing_set_knobab.append(trace)
    for trace in testing_set_neg:
        testing_set_knobab.append(trace)
    legagy_dump_log(training_set_pos, os.path.join(folder, "1_training_positive.xes"))
    legagy_dump_log(training_set_neg, os.path.join(folder, "0_training_negative.xes"))
    legagy_dump_log(testing_set_knobab, os.path.join(folder, "testing_knobab.xes"))
    legagy_dump_log(testing_set_pos, os.path.join(folder, "1_testing_positive.xes"))
    legagy_dump_log(testing_set_neg, os.path.join(folder, "0_testing_negative.xes"))
    with open(os.path.join(folder, "testing_classes_knobab.txt"),'w') as file:
        file.write('\n'.join(str(year) for year in list_class))