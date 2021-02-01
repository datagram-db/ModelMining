# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import LogGeneration
import RulesExtraction
import Other
import jsonpickle
import sys
import os

from sklearn.model_selection import StratifiedKFold
from DevianceMiningPipeline.deviancecommon import read_XES_log
import DevianceMiningPipeline.predicates
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from DevianceMiningPipeline.declaretemplates import template_response, template_precedence
from DevianceMiningPipeline.predicates import SatAllProp, SatProp, SatCases

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
    #jsonpickle.decode(open(conf_file).read()).complete_embedding_generation(LOGS_FOLDER, DATA_EXP)
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


def write_log_file_with_cond(log, filen, f):
    if not os.path.isfile(filen):
        print("Writing: "+filen)
        f(log)
        write_log_file(log, filen)


def write_log_file_with_label_cond(log, filen, attn, val):
    """
    This function updates the log by labelling it as 1 if there exists a trace with attribute attn and associated value value

    :param log:         XLog to analyse
    :param filen:       File where to serialize the labelled log
    :param attn:        Attribute to be found within the trace
    :param val:         Value associated to the attribute to be found.
    """
    write_log_file_with_cond(log, filen, lambda x : DevianceMiningPipeline.predicates.tagLogWithValueEqOverTraceAttn(x, attn, val))

def generateTagging():
    #assert os.path.isfile("data/logs/EnglishBPIChallenge2011.xes")
    #if (not os.path.isfile("data/logs/bpi2011_dCC.xes")) or (not os.path.isfile("data/logs/bpi2011_t101.xes")) or (not os.path.isfile("data/logs/bpi2011_m13.xes")) or (not os.path.isfile("data/logs/bpi2011_m16.xes")):
    #    log = read_XES_log("data/logs/EnglishBPIChallenge2011.xes")
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_dCC.xes", "Diagnosis", "maligniteit cervix")
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_t101.xes", "Treatment code", 101)
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_m13.xes", "Diagnosis code", "M13")
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_m16.xes", "Diagnosis code", "M16")
    assert os.path.isfile("data/logs/sepsis.xes")
    if ((not os.path.isfile("data/logs/sepsis_payload.xes")) or (not os.path.isfile("data/logs/sepsis_proc.xes")) or (not os.path.isfile("data/logs/sepsis_decl.xes")) or (not os.path.isfile("data/logs/sepsis_mr_tr.xes")) or (not os.path.isfile("data/logs/sepsis_mra_tra.xes"))):
        log = read_XES_log("data/logs/sepsis.xes")
        write_log_file_with_cond(log, "data/logs/sepsis_proc.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactSubsequence(x, ["Admission NC","Leucocytes", "CRP"]))
        write_log_file_with_cond(log, "data/logs/sepsis_decl.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithSatAllProp(x, [(template_response, ["IV Antibiotics", "Leucocytes"]),
                                                                                                                                        (template_response, ["LacticAcid", "IV Antibiotics"]),
                                                                                                                                        (template_response, ["ER Triage", "CRP"])], SatCases.NotVacuitySat))
        write_log_file_with_cond(log, "data/logs/sepsis_mr_tr.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactOccurrence(x, ["IV Liquid", "LacticAcid", "Leucocytes"], 2))
        write_log_file_with_cond(log, "data/logs/sepsis_mra_tra.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithOccurrence(x, ["IV Liquid", "LacticAcid", "Leucocytes"], 2))
        write_log_file_with_cond(log, "data/logs/sepsis_payload.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithValueEqOverIthEventAttn(x, "DisfuncOrg", True, 0))
    assert os.path.isfile("data/logs/merged_xray.xes")
    if ((not os.path.isfile("data/logs/xray_proc.xes")) or (not os.path.isfile("data/logs/xray_decl.xes")) or (not os.path.isfile("data/logs/xray_mr_tr.xes")) or (not os.path.isfile("data/logs/xray_mra_tra.xes"))):
        log = read_XES_log("data/logs/merged_xray.xes")
        write_log_file_with_cond(log, "data/logs/xray_proc.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactSubsequence(x, ["check_X_ray_risk", "examine_patient", "perform_surgery"]))
        write_log_file_with_cond(log, "data/logs/xray_decl.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithSatAllProp(x, [(template_precedence, ["perform_reposition", "perform_X_ray"]),
                                                                                                                                        (template_precedence, ["apply_cast", "perform_X_ray"]),
                                                                                                                                       (template_precedence, ["remove_cast", "apply_cast"])], SatCases.NotVacuitySat))
        write_log_file_with_cond(log, "data/logs/xray_mr_tr.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 2))
        write_log_file_with_cond(log, "data/logs/xray_mra_tra.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 2))
        write_log_file_with_cond(log, "data/logs/xray_payload.xes", lambda x: DevianceMiningPipeline.predicates.logRandomTagger(x, 0, 1, 0.1))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    LogGeneration.write_xeses(LOGS_FOLDER)
    generateTagging()
    #conf_file = "BPI2011_dCC.json"
    #if len(sys.argv)>1:
    #    conf_file = sys.argv[1]
    #preprocess = False
    #if len(sys.argv)>2:
    #    test = sys.argv[2]
    #    preprocess = not (test == "skipPreprocessing")
    #run_complete_configuration_and_run(conf_file, preprocess)

