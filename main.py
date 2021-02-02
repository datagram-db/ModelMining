# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import LogGeneration
import Other
import jsonpickle
import sys
import os

from DevianceMiningPipeline.deviancecommon import read_XES_log
import DevianceMiningPipeline.predicates
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from DevianceMiningPipeline.declaretemplates_new import template_response, template_precedence, template_init, \
    template_alternate_precedence
from DevianceMiningPipeline.predicates import SatCases
from DevianceMiningPipeline import ConfigurationFile

LOGS_FOLDER="data/logs"
DATA_EXP="data/experiments"
ranges=[5, 10, 15, 20, 25, 30]#, 10, 15, 20, 25, 30, 35, 40, 45, 50)

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
    conf = jsonpickle.decode(open(conf_file).read())
    assert (isinstance(conf, ConfigurationFile))
    #conf.complete_embedding_generation(LOGS_FOLDER, DATA_EXP)
    conf.run(LOGS_FOLDER, DATA_EXP, ranges, doNr0)

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
        file.close()


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
    assert os.path.isfile("sepsis_er.json")
    if ((not os.path.isfile("data/logs/sepsis_payload.xes")) or (not os.path.isfile("data/logs/sepsis_proc.xes")) or (not os.path.isfile("data/logs/sepsis_decl.xes")) or (not os.path.isfile("data/logs/sepsis_mr_tr.xes")) or (not os.path.isfile("data/logs/sepsis_mra_tra.xes"))):
        log = read_XES_log("data/logs/sepsis.xes")
        conf = jsonpickle.decode(open("sepsis_er.json").read())
        assert(isinstance(conf, ConfigurationFile))

        write_log_file_with_cond(log, "data/logs/sepsis_proc.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactSubsequence(x, ["Admission NC","Leucocytes", "CRP"]))
        conf.setLogName("sepsis_proc.xes")
        conf.setOutputFolder("sepsis_proc_out")
        conf.setExperimentName("sepsis_proc")
        conf.dump("sepsis_proc.json")

        write_log_file_with_cond(log, "data/logs/sepsis_decl.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithSatAllProp(x, [(template_response, ["IV Antibiotics", "Leucocytes"]),
                                                                                                                                        (template_response, ["LacticAcid", "IV Antibiotics"]),
                                                                                                                                        (template_response, ["ER Triage", "CRP"])], SatCases.NotVacuitySat))
        conf.setLogName("sepsis_decl.xes")
        conf.setOutputFolder("sepsis_decl_out")
        conf.setExperimentName("sepsis_decl")
        conf.dump("sepsis_decl.json")

        write_log_file_with_cond(log, "data/logs/sepsis_mr_tr.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactOccurrence(x, ["IV Liquid", "LacticAcid", "Leucocytes"], 1))
        conf.setLogName("sepsis_mr_tr.xes")
        conf.setOutputFolder("sepsis_mr_tr_out")
        conf.setExperimentName("sepsis_mr_tr")
        conf.dump("sepsis_mr_tr.json")

        write_log_file_with_cond(log, "data/logs/sepsis_mra_tra.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithOccurrence(x, ["IV Liquid", "LacticAcid", "Leucocytes"], 2))
        conf.setLogName("sepsis_mra_tra.xes")
        conf.setOutputFolder("sepsis_mra_tra_out")
        conf.setExperimentName("sepsis_mra_tra")
        conf.dump("sepsis_mra_tra.json")

        write_log_file_with_cond(log, "data/logs/sepsis_payload.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithValueEqOverIthEventAttn(x, "DisfuncOrg", True, 0))
        conf.setLogName("sepsis_payload.xes")
        conf.setOutputFolder("sepsis_payload_out")
        conf.setExperimentName("sepsis_payload")
        conf.dump("sepsis_payload.json")

    assert os.path.isfile("data/logs/merged_xray.xes")
    assert os.path.isfile("synth_xray.json")
    if ((not os.path.isfile("data/logs/xray_proc.xes")) or (not os.path.isfile("data/logs/xray_decl.xes")) or (not os.path.isfile("data/logs/xray_mr_tr.xes")) or (not os.path.isfile("data/logs/xray_mra_tra.xes"))):
        log = read_XES_log("data/logs/merged_xray.xes")
        conf = jsonpickle.decode(open("synth_xray.json").read())
        assert(isinstance(conf, ConfigurationFile))

        write_log_file_with_cond(log, "data/logs/xray_proc.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactSubsequence(x, ["check_X_ray_risk", "examine_patient", "perform_surgery"]))
        conf.setLogName("xray_proc.xes")
        conf.setOutputFolder("xray_proc_out")
        conf.setExperimentName("xray_proc")
        conf.dump("xray_proc.json")

        write_log_file_with_cond(log, "data/logs/xray_decl.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithSatAnyProp(x, [(template_init, ["check_X_ray_risk"])], SatCases.NotVacuitySat))
        conf.setLogName("xray_decl.xes")
        conf.setOutputFolder("xray_decl_out")
        conf.setExperimentName("xray_decl")
        conf.dump("xray_decl.json")

        write_log_file_with_cond(log, "data/logs/xray_mr_tr.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithExactOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 1))
        conf.setLogName("xray_mr_tr.xes")
        conf.setOutputFolder("xray_mr_tr_out")
        conf.setExperimentName("xray_mr_tr")
        conf.dump("xray_mr_tr.json")

        write_log_file_with_cond(log, "data/logs/xray_mra_tra.xes", lambda x: DevianceMiningPipeline.predicates.tagLogWithOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 2))
        conf.setLogName("xray_mra_tra.xes")
        conf.setOutputFolder("xray_mra_tra_out")
        conf.setExperimentName("xray_mra_tra")
        conf.dump("xray_mra_tra.json")

        write_log_file_with_cond(log, "data/logs/xray_payload.xes", lambda x: DevianceMiningPipeline.predicates.logRandomTagger(x, 0, 1, 0.1))
        conf.setLogName("xray_payload.xes")
        conf.setOutputFolder("xray_payload_out")
        conf.setExperimentName("xray_payload")
        conf.dump("xray_payload.json")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    LogGeneration.write_xeses(LOGS_FOLDER)
    generateTagging()
    #conf_file = "sepsis_proc.json"
    #if len(sys.argv)>1:
    #    conf_file = sys.argv[1]
    #preprocess = True
    #if len(sys.argv)>2:
    #    test = sys.argv[2]
    #    preprocess = not (test == "skipPreprocessing")
    for conf_file in ["sepsis_proc.json"]:#, "sepsis_decl.json", "sepsis_mr_tr.json", "sepsis_mra_tra.json", "sepsis_payload.json", "xray_proc.json", "xray_decl.json", "xray_mr_tr.json", "xray_mra_tra.json", "xray_payload.json"]:
        print("Now running: "+conf_file)
        run_complete_configuration_and_run(conf_file, False)


