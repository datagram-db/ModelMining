# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import LogGeneration
import RulesExtraction
import Other
import jsonpickle
import sys
import os

from DevianceMiningPipeline.deviancecommon import read_XES_log
import DevianceMiningPipeline.predicates
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer

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

def do_benchmarks():
    pass


def describe_logs():
    for key,value in to_describe.items():
        Other.describe(key, LOGS_FOLDER, value)

def run_complete_configuration_and_run(conf_file, doNr0 = True):
    jsonpickle.decode(open(conf_file).read()).run(LOGS_FOLDER, DATA_EXP, ranges, doNr0)

# from DevianceMiningPipeline import ConfigurationFile
# cf = ConfigurationFile()
# cf.setExperimentName("output_pos_neg_data")
# cf.setLogName("output_pos_and_neg.xes")
# cf.setOutputFolder("output_pos_neg_res")
# cf.setMaxDepth(10)
# cf.setMinLeaf(10)
# cf.setSequenceThreshold(5)
# cf.dump("output_pos_neg_data.json")

def write_log_file(log, filen):
    with open(filen, "w") as file:
        XesXmlSerializer().serialize(log, file)

def write_log_file_with_label_cond(log, filen, attn, val):
    if not os.path.isfile(filen):
        print("Writing: "+filen)
        DevianceMiningPipeline.predicates.logTagger(log,
                                                    DevianceMiningPipeline.predicates.compileAttributeWithValue(
                                                        attn, val))
        write_log_file(log, filen)


def generateTagging():
    assert os.path.isfile("data/logs/bpi2011.xes")
    if (not os.path.isfile("data/logs/bpi2011_dCC.xes")) or (not os.path.isfile("data/logs/bpi2011_t101.xes")) or (not os.path.isfile("data/logs/bpi2011_m13.xes")) or (not os.path.isfile("data/logs/bpi2011_m16.xes")):
        log = read_XES_log("data/logs/bpi2011.xes")
        write_log_file_with_label_cond(log, "data/logs/bpi2011_dCC.xes", "Diagnosis", "maligniteit cervix")
        write_log_file_with_label_cond(log, "data/logs/bpi2011_t101.xes", "Treatment code", 101)
        write_log_file_with_label_cond(log, "data/logs/bpi2011_m13.xes", "Diagnosis code", "M13")
        write_log_file_with_label_cond(log, "data/logs/bpi2011_m16.xes", "Diagnosis code", "M16")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    LogGeneration.write_xeses(LOGS_FOLDER)
    generateTagging()
    conf_file = "synth_xray.json"
    if len(sys.argv)>1:
        conf_file = sys.argv[1]
    preprocess = True
    if len(sys.argv)>2:
        test = sys.argv[2]
        preprocess = not (test == "skipPreprocessing")
    run_complete_configuration_and_run(conf_file, preprocess)

