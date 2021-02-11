import LogGeneration
import Other
import jsonpickle
import os

from DevianceMiningPipeline.DataPreparation.RunWholeStrategy import RunWholeStrategy
from DevianceMiningPipeline.deviancecommon import read_XES_log
import DevianceMiningPipeline.LogTaggingViaPredicates
from DevianceMiningPipeline.DataPreparation.TaggingStrategy import TaggingStrategy
from DevianceMiningPipeline.declaretemplates_new import template_response, template_init
from DevianceMiningPipeline.LogTaggingViaPredicates import SatCases
from DevianceMiningPipeline import ConfigurationFile, PayloadType
from DevianceMiningPipeline.DataPreparation.RetagLogWithUniqueIds import changeLog

LOGS_FOLDER = "data/logs"
DATA_EXP = "data/experiments"
ranges = [5, 10, 15, 20, 25, 30]  # , 10, 15, 20, 25, 30, 35, 40, 45, 50)

to_describe = {
    # "sepsis_constr": "sepsis_constraint_tagged.xes", --> missing
    # "sepsis_seq": "sepsis_sequence_tagged.xes",      --> missing
    "sepsis_er": "sepsis_tagged_er.xes",
    "SYNTH_MR2": "synth_mr2_tagged.xes",
    "SYNTH_TR": "synth_tr_tagged.xes",
    # "BPI2011_M16": "EnglishBPIChallenge2011_tagged_m16.xes",
    # "BPI2011_T101": "EnglishBPIChallenge2011_tagged_t101.xes",
    # "BPI2011_M13": "EnglishBPIChallenge2011_tagged_M13.xes",
    # "BPI2011_CC": "EnglishBPIChallenge2011_tagged_cc.xes",
    # "XRAY": "merged_xray.xes"
}


def do_benchmarks():
    pass


def describe_logs():
    for key, value in to_describe.items():
        Other.describe(key, LOGS_FOLDER, value)


def run_complete_configuration_and_run(conf_file, doNr0=True, ranges=None, max_splits=5):
    if ranges is None:
        ranges = [5, 10, 15, 20, 25, 30]
    conf = jsonpickle.decode(open(conf_file).read())
    assert (isinstance(conf, ConfigurationFile))
    # conf.complete_embedding_generation(LOGS_FOLDER, DATA_EXP)
    conf.run(LOGS_FOLDER, DATA_EXP, ranges, doNr0, max_splits=max_splits)


def guaranteeUniqueXes():
    # changeLog("data/logs/merged_xray.xes")
    # changeLog("data/logs/sepsis.xes")
    changeLog("data/logs/EnglishBPIChallenge2011.xes")


def generate_configuration():
    pass
    from DevianceMiningPipeline import ConfigurationFile



# def write_log_file_with_label_cond(log, filen, attn, val):
#     """
#     This function updates the log by labelling it as 1 if there exists a trace with attribute attn and associated value value
#
#     :param log:         XLog to analyse
#     :param filen:       File where to serialize the labelled log
#     :param attn:        Attribute to be found within the trace
#     :param val:         Value associated to the attribute to be found.
#     """
#     write_log_file_with_cond(log, filen, lambda x : DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, attn, val))
#
#


def runWholeConfiguration(pipeline_conf, original_log_path, conf, map_as_strategy):
    assert (isinstance(original_log_path, str))
    assert (isinstance(map_as_strategy, dict))
    assert (isinstance(pipeline_conf, RunWholeStrategy))
    assert (isinstance(conf, ConfigurationFile))

    logs_folder = pipeline_conf.getLogsFolder()
    full_path = pipeline_conf.getCompleteLogPath(original_log_path)

    printWithColor(
        "Guaranteeing that the logs used for the testing have unique trace ids (it is required for better training the dataset)")
    origLog, log = changeLog(full_path)

    for key, value in map_as_strategy.items():
        printWithColor("Running configuration: " + key)

        jsonFile = TaggingStrategy(key, value)(logs_folder, conf, log)
        assert isinstance(jsonFile, TaggingStrategy)
        pipeline_conf(jsonFile)


def generateTagging():
    # assert os.path.isfile("data/logs/EnglishBPIChallenge2011.xes")
    # if (not os.path.isfile("data/logs/bpi2011_dCC.xes")) or (not os.path.isfile("data/logs/bpi2011_t101.xes")) or (not os.path.isfile("data/logs/bpi2011_m13.xes")) or (not os.path.isfile("data/logs/bpi2011_m16.xes")):
    #    log = read_XES_log("data/logs/EnglishBPIChallenge2011.xes")
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_dCC.xes", "Diagnosis", "maligniteit cervix")
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_t101.xes", "Treatment code", 101)
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_m13.xes", "Diagnosis code", "M13")
    #    write_log_file_with_label_cond(log, "data/logs/bpi2011_m16.xes", "Diagnosis code", "M16")
    assert os.path.isfile("data/logs/sepsis.xes_unique.xes")
    assert os.path.isfile("sepsis_er.json")
    # if ((not os.path.isfile("data/logs/sepsis_payload.xes")) or (not os.path.isfile("data/logs/sepsis_proc.xes")) or (not os.path.isfile("data/logs/sepsis_decl.xes")) or (not os.path.isfile("data/logs/sepsis_mr_tr.xes")) or (not os.path.isfile("data/logs/sepsis_mra_tra.xes"))):
    log = read_XES_log("data/logs/sepsis.xes_unique.xes")
    conf = jsonpickle.decode(open("sepsis_er.json").read())
    assert (isinstance(conf, ConfigurationFile))
    #
    # TaggingStrategy("sepsis_payload", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverIthEventAttn(x, "DisfuncOrg", True, 0))(LOGS_FOLDER, conf, log)
    # TaggingStrategy("sepsis_mra_tra", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["IV Liquid", "LacticAcid", "Leucocytes"], 2))(LOGS_FOLDER, conf, log)
    # TaggingStrategy("sepsis_mr_tr", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, ["Admission NC", "CRP"], 1))(LOGS_FOLDER, conf, log)
    # TaggingStrategy("sepsis_proc", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["Admission NC", "Leucocytes", "CRP"]))(LOGS_FOLDER, conf, log)
    # TaggingStrategy("sepsis_decl", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [(template_response, ["IV Antibiotics", "Leucocytes"]),
    #                                                                                                                  (template_response, ["LacticAcid", "IV Antibiotics"]),
    #                                                                                                                  (template_response, ["ER Triage", "CRP"])], SatCases.NotVacuitySat))(LOGS_FOLDER, conf, log)

    # write_log_file_with_cond(log, "data/logs/sepsis_proc.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["Admission NC", "Leucocytes", "CRP"]))
    # conf.setLogName("sepsis_proc.xes")
    # conf.setOutputFolder("sepsis_proc_out")
    # conf.setExperimentName("sepsis_proc")
    # conf.dump("sepsis_proc.json")
    # write_log_file_with_cond(log, "data/logs/sepsis_decl.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [(template_response, ["IV Antibiotics", "Leucocytes"]),
    #                                                                                                                                              (template_response, ["LacticAcid", "IV Antibiotics"]),
    #                                                                                                                                              (template_response, ["ER Triage", "CRP"])], SatCases.NotVacuitySat))
    # conf.setLogName("sepsis_decl.xes")
    # conf.setOutputFolder("sepsis_decl_out")
    # conf.setExperimentName("sepsis_decl")
    # conf.dump("sepsis_decl.json")
    # write_log_file_with_cond(log, "data/logs/sepsis_mr_tr.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, ["Admission NC", "CRP"], 1))
    # conf.setLogName("sepsis_mr_tr.xes")
    # conf.setOutputFolder("sepsis_mr_tr_out")
    # conf.setExperimentName("sepsis_mr_tr")
    # conf.dump("sepsis_mr_tr.json")
    # write_log_file_with_cond(log, "data/logs/sepsis_mra_tra.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["IV Liquid", "LacticAcid", "Leucocytes"], 2))
    # conf.setLogName("sepsis_mra_tra.xes")
    # conf.setOutputFolder("sepsis_mra_tra_out")
    # conf.setExperimentName("sepsis_mra_tra")
    # conf.dump("sepsis_mra_tra.json")
    # write_log_file_with_cond(log, "data/logs/sepsis_payload.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverIthEventAttn(x, "DisfuncOrg", True, 0))
    # conf.setLogName("sepsis_payload.xes")
    # conf.setOutputFolder("sepsis_payload_out")
    # conf.setExperimentName("sepsis_payload")
    # conf.dump("sepsis_payload.json")


# assert os.path.isfile("data/logs/merged_xray.xes_unique.xes")
# assert os.path.isfile("synth_xray.json")
# if ((not os.path.isfile("data/logs/xray_proc.xes")) or (not os.path.isfile("data/logs/xray_decl.xes")) or (not os.path.isfile("data/logs/xray_mr_tr.xes")) or (not os.path.isfile("data/logs/xray_mra_tra.xes"))):
#     log = read_XES_log("data/logs/merged_xray.xes_unique.xes")
#     conf = jsonpickle.decode(open("synth_xray.json").read())
#     assert(isinstance(conf, ConfigurationFile))
#
#     write_log_file_with_cond(log, "data/logs/xray_proc.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["check_X_ray_risk", "examine_patient", "perform_surgery"]))
#     conf.setLogName("xray_proc.xes")
#     conf.setOutputFolder("xray_proc_out")
#     conf.setExperimentName("xray_proc")
#     conf.dump("xray_proc.json")
#
#     write_log_file_with_cond(log, "data/logs/xray_decl.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [(template_init, ["check_X_ray_risk"])], SatCases.NotVacuitySat))
#     conf.setLogName("xray_decl.xes")
#     conf.setOutputFolder("xray_decl_out")
#     conf.setExperimentName("xray_decl")
#     conf.dump("xray_decl.json")
#
#     write_log_file_with_cond(log, "data/logs/xray_mr_tr.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 1))
#     conf.setLogName("xray_mr_tr.xes")
#     conf.setOutputFolder("xray_mr_tr_out")
#     conf.setExperimentName("xray_mr_tr")
#     conf.dump("xray_mr_tr.json")
#
#     write_log_file_with_cond(log, "data/logs/xray_mra_tra.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 2))
#     conf.setLogName("xray_mra_tra.xes")
#     conf.setOutputFolder("xray_mra_tra_out")
#     conf.setExperimentName("xray_mra_tra")
#     conf.dump("xray_mra_tra.json")
#
#     write_log_file_with_cond(log, "data/logs/xray_payload.xes", lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.logRandomTagger(x, 0, 1, 0.1))
#     conf.setLogName("xray_payload.xes")
#     conf.setOutputFolder("xray_payload_out")
#     conf.setExperimentName("xray_payload")
#     conf.dump("xray_payload.json")


def printWithColor(str):
    print("\x1b[6;30;42m " + str + "\x1b[0m")

def run_xray(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)

    cf = ConfigurationFile()
    cf.setExperimentName("synth_xray")
    cf.setLogName("merged_xray.xes")
    cf.setOutputFolder("xray")
    cf.setMaxDepth(10)
    cf.setMinLeaf(10)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(["concept: name", "Label", "lifecycle: transition"])
    cf.doForceTime()
    cf.setPayloadSettings("xray_settings.cfg")
    cf.dump("synth_xray.json")

    xray_map = {
        "xray_payload": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverIthEventAttn(x, "DisfuncOrg", True, 0),
        "xray_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 2),
        "xray_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, ["apply_cast", "perform_reposition", "prescribe_rehabilitation"], 1),
        "xray_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["check_X_ray_risk", "examine_patient", "perform_surgery"]),
        "xray_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [(template_init, ["check_X_ray_risk"])], SatCases.NotVacuitySat)
    }

    runWholeConfiguration(pipeline, "merged_xray.xes", cf, xray_map)

def run_sepsis(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)

    cf = ConfigurationFile()
    cf.setExperimentName("sepsis_er")
    cf.setLogName("sepsis.xes")
    cf.setOutputFolder("SepsisDWD")
    cf.setMaxDepth(5)
    cf.setMinLeaf(5)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(["Diagnosis", "Diagnose", "time:timestamp", "concept: name", "Label", "lifecycle: transition", "org:group", "org: group"])
    cf.setPayloadSettings("sepsis_settings.cfg")
    cf.dump("sepsis_er.json")

    sepsis_map = {
        "sepsis_payload": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverIthEventAttn(x,
                                                                                                                     "DisfuncOrg",
                                                                                                                     True,
                                                                                                                     0),
        "sepsis_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["IV Liquid",
                                                                                                            "LacticAcid",
                                                                                                            "Leucocytes"],
                                                                                                        2),
        "sepsis_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, [
            "Admission NC", "CRP"], 1),
        "sepsis_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
            "Admission NC", "Leucocytes", "CRP"]),
        "sepsis_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
            (template_response, ["IV Antibiotics", "Leucocytes"]),
            (template_response, ["LacticAcid", "IV Antibiotics"]),
            (template_response, ["ER Triage", "CRP"])], SatCases.NotVacuitySat)
    }

    runWholeConfiguration(pipeline, "sepsis.xes", cf, sepsis_map)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pipeline = RunWholeStrategy(os.path.join("data","logs"),
                                os.path.join("data","experiments"),
                                True,
                                [5, 10, 15, 20, 25, 30],
                                4)
    run_sepsis(pipeline)