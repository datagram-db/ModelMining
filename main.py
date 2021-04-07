import jsonpickle
import os
from DevianceMiningPipeline.DataPreparation.RunWholeStrategy import RunWholeStrategy
import DevianceMiningPipeline.LogTaggingViaPredicates
from DevianceMiningPipeline.DataPreparation.TaggingStrategy import TaggingStrategy
#from DevianceMiningPipeline.DataPreparation.main_pipeline_run import runWholeConfiguration
from DevianceMiningPipeline.declaretemplates_new import template_response, template_init
from DevianceMiningPipeline.LogTaggingViaPredicates import SatCases
from DevianceMiningPipeline import ConfigurationFile, PayloadType
from DevianceMiningPipeline.DataPreparation.RetagLogWithUniqueIds import changeLog
from main_pipeline_run import runWholeConfiguration


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
    "xray_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x,
                                                                                                     ["check_X_ray_risk","check_X_ray_risk"],
                                                                                                     1),
        "xray_payload": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverIthEventAttn(x,
                                                                                                                   "DisfuncOrg",
                                                                                                                   True,
                                                                                                                   0),
        "xray_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["apply_cast",
                                                                                                          "perform_reposition",
                                                                                                          "prescribe_rehabilitation"],
                                                                                                      2),
#
        "xray_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
            "check_X_ray_risk", "examine_patient", "perform_surgery"]),
        "xray_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [
            (template_init, ["check_X_ray_risk"])], SatCases.NotVacuitySat)
    }
    runWholeConfiguration(pipeline, "merged_xray.xes", cf, xray_map)
#
#
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
    cf.setAutoIgnore(
        ["Diagnosis", "Diagnose", "time:timestamp", "concept: name", "Label", "lifecycle: transition"])
    cf.setPayloadSettings("sepsis_settings.cfg")
    cf.dump("sepsis_er.json")
    sepsis_map = {
         "sepsis_payload2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverIthEventAttn(x,
                                                                                                                      "DisfuncOrg",
                                                                                                                      True,
                                                                                                                      0),
          "sepsis_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["IV Liquid",
                                                                                                              "LacticAcid",
                                                                                                              "Leucocytes"],
                                                                                                          2),
          "sepsis_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, [
              "Admission NC", "CRP", "Leucocytes"], 1),
         "sepsis_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
             "Admission NC", "Leucocytes"]),
          "sepsis_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
              (template_response, ["IV Antibiotics", "Leucocytes"]),
              (template_response, ["LacticAcid", "IV Antibiotics"]),
              (template_response, ["ER Triage", "CRP"])], SatCases.NotVacuitySat)
    }
    runWholeConfiguration(pipeline, "sepsis.xes", cf, sepsis_map)
#
#
def run_bpm11(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)
    cf = ConfigurationFile()
    cf.setExperimentName("bpi11")
    cf.setLogName("EnglishBPIChallenge2011.xes")
    cf.setOutputFolder("BPI11")
    cf.setMaxDepth(5)
    cf.setMinLeaf(5)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(
        ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
         "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"])
    cf.setPayloadSettings("bpi2011_settings.cfg")
    cf.dump("bpi11.json")
    bpm11_map = {
         "bpi11_payload_dCC": lambda
             x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Diagnosis",
                                                                                                 "maligniteit cervix"),
         "bpi11_payload_T101": lambda
             x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Treatment code",
                                                                                                 101),
         "bpi11_payload_M13": lambda
             x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Diagnosis code",
                                                                                                 "M13"),
         "bpi11_payload_M16": lambda
             x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Diagnosis code",
                                                                                                 "M16"),
         "bpi11_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, [
             "assumption laboratory", "Milk acid dehydrogenase LDH kinetic"], 2),
         "bpi11_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, [
             "assumption laboratory", "Milk acid dehydrogenase LDH kinetic"], 1),
         "bpi11_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
             "unconjugated bilirubin", "bilirubin - total", "glucose"]),
         "bpi11_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
             (template_init, ["outpatient follow-up consultation"])
        ], SatCases.NotVacuitySat)
    }
    runWholeConfiguration(pipeline, "EnglishBPIChallenge2011.xes", cf, bpm11_map)


if __name__ == '__main__':
    pipeline = RunWholeStrategy(os.path.join("data", "logs"),
                                os.path.join("data", "experiments"),
                                True,
                                [5, 10, 15, 20, 25, 30, 60, 80],
                                5)
    run_sepsis(pipeline)
    run_bpm11(pipeline)
    run_xray(pipeline)