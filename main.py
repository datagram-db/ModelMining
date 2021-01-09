# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import LogGeneration
import RulesExtraction
import Other
import jsonpickle

LOGS_FOLDER="data/logs"
DATA_EXP="data/experiments"
ranges=(5, 15, 25)

to_describe = {
    #"sepsis_constr": "sepsis_constraint_tagged.xes", --> missing
    #"sepsis_seq": "sepsis_sequence_tagged.xes",      --> missing
    "sepsis_er": "sepsis_tagged_er.xes",
    "SYNTH_MR2": "synth_mr2_tagged.xes",
    "SYNTH_TR": "synth_tr_tagged.xes",
    #"BPI2011_M16": "EnglishBPIChallenge2011_tagged_m16.xes",
    #"BPI2011_T101": "EnglishBPIChallenge2011_tagged_t101.xes",
    #"BPI2011_M13": "EnglishBPIChallenge2011_tagged_M13.xes",
    #"BPI2011_CC": "EnglishBPIChallenge2011_tagged_cc.xes",
    #"XRAY": "merged_xray.xes"
}

def write_logs_01():
    LogGeneration.write_xeses(LOGS_FOLDER)

def do_benchmarks():
    pass


def describe_logs():
    for key,value in to_describe.items():
        Other.describe(key, LOGS_FOLDER, value)

def run_complete_configuration_and_run(conf_file):
    jsonpickle.decode(open(conf_file).read()).run(LOGS_FOLDER, DATA_EXP, ranges, True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_complete_configuration_and_run("sepsis_er.json")
    #RulesExtraction.get_dtrules()               ## Requires: snapshots
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
