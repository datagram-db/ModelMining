import jsonpickle
import os
from DevianceMiningPipeline.DataPreparation.RunWholeStrategy import RunWholeStrategy
import DevianceMiningPipeline.LogTaggingViaPredicates
from DevianceMiningPipeline.DataPreparation.TaggingStrategy import TaggingStrategy
from DevianceMiningPipeline.declaretemplates_new import template_response, template_init, template_absence1, \
    template_precedence, template_responded_existence, template_exist, template_absence2, template_chain_response
from DevianceMiningPipeline.LogTaggingViaPredicates import SatCases
from DevianceMiningPipeline import ConfigurationFile, PayloadType
from DevianceMiningPipeline.DataPreparation.RetagLogWithUniqueIds import changeLog
from main_pipeline_run import runWholeConfiguration
#from antlr4 import *
#
#
# def run_xrayKR(pipeline):
#     assert isinstance(pipeline, RunWholeStrategy)
#     cf = ConfigurationFile()
#     cf.setExperimentName("KR_XRAY2")
#     cf.setLogName("XRayTrain_KR.xes")
#     cf.setOutputFolder("XR_XRAY2")
#     cf.setMaxDepth(10)
#     cf.setMinLeaf(10)
#     cf.setSequenceThreshold(5)
#     cf.setPayloadType(None)
#     cf.setAutoIgnore(["concept: name", "Label", "lifecycle: transition"])
#     cf.doForceTime()
#     cf.setPayloadSettings("xray_settings.cfg")
#     cf.dump("XRayTrain_KR2.xes")
#     xray_map = {
#         "xray_KR_tagged2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.ignoreTagging()
#     }
#     runWholeConfiguration(pipeline, "XRayMerged2.xes", cf, xray_map)
#
# def run_xrayKRInv(pipeline):
#     assert isinstance(pipeline, RunWholeStrategy)
#     cf = ConfigurationFile()
#     cf.setExperimentName("KR_XRAYInv")
#     cf.setLogName("XRayTrain_KRInv.xes")
#     cf.setOutputFolder("XR_XRAYInv")
#     cf.setMaxDepth(10)
#     cf.setMinLeaf(10)
#     cf.setSequenceThreshold(5)
#     cf.setPayloadType(None)
#     cf.setAutoIgnore(["concept: name", "Label", "lifecycle: transition"])
#     cf.doForceTime()
#     cf.setPayloadSettings("xray_settings.cfg")
#     cf.dump("XRayTrain_KRInv.xes")
#     xray_map = {
#         "xray_KR_taggedInv": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.ignoreTagging()
#     }
#     runWholeConfiguration(pipeline, "XRayTrain_KRInv.xes", cf, xray_map)





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
          "xray_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["apply_cast",
                                                                                                            "perform_reposition",
                                                                                                            "prescribe_rehabilitation"],
                                                                                                        2),

         "xray_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
             "check_X_ray_risk", "examine_patient", "perform_surgery"]),
         "xray_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [
             (template_init, ["check_X_ray_risk"])], SatCases.NotVacuitySat)
     }
     runWholeConfiguration(pipeline, "merged_xray.xes", cf, xray_map)


# def run_bpm21(pipeline):
#     assert isinstance(pipeline, RunWholeStrategy)
#     cf = ConfigurationFile()
#     cf.setExperimentName("bpm21")
#     cf.setLogName("length_30.xes")
#     cf.setOutputFolder("bpm21")
#     cf.setMaxDepth(10)
#     cf.setMinLeaf(10)
#     cf.setSequenceThreshold(5)
#     cf.setPayloadType(PayloadType.both)
#     cf.setAutoIgnore(["concept: name", "Label", "lifecycle: transition"])
#     cf.doForceTime()
#     cf.setPayloadSettings("xray_settings.cfg")
#     cf.dump("bpm21.json")
#     xray_map = {
#         "bpm21_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [
#             (template_chain_response, ["activity 1", "activity 2"])], SatCases.NotVacuitySat),
#         "bpm21_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["activity 14",
#                                                                                                           "activity 12"],
#                                                                                                       2),
#         "bpm21_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
#             "activity 14", "activity 12"])
#     }
#     runWholeConfiguration(pipeline, "length_30.xes", cf, xray_map)

def run_bank(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)
    cf = ConfigurationFile()
    cf.setExperimentName("bank")
    cf.setLogName("bank.xes")
    cf.setOutputFolder("bank")
    cf.setMaxDepth(10)
    cf.setMinLeaf(10)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(
        ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
         "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"])
    cf.setPayloadSettings("bpi2017_settings.cfg")
    cf.dump("bank.json")
    xray_map = {
       # "bank_decl_tagging": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.ignoreTagging(),
        "bank_invalid_pin": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverEventAttn(x, "ValidPin", "false"),
        "bank_no_money_move": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverEventAttn(x, "Money", 0.0),
        "bank_decl1": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [
            (template_init, ["PinInsert"])], SatCases.NotVacuitySat),
        "bank_decl2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [
            (template_chain_response, ["PinInsert", "LogInBankAccount"])], SatCases.NotVacuitySat),
        "bank_decl3": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x, [
            (template_init, ["LogInBankAccount"])], SatCases.NotVacuitySat),
        "bank_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["PinInsert", "AddMoney"], 2),
        "bank_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, ["PinInsert"], 3),
        "bank_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
            "LogOff", "AddMoney", "WidthdrawMoney"])
    }
    runWholeConfiguration(pipeline, "bank.xes", cf, xray_map)

def run_synth(pipeline):
     assert isinstance(pipeline, RunWholeStrategy)
     cf = ConfigurationFile()
     cf.setExperimentName("synth_tr_tagged")
     cf.setLogName("synth_tr_tagged.xes")
     cf.setOutputFolder("synth")
     cf.setMaxDepth(10)
     cf.setMinLeaf(10)
     cf.setSequenceThreshold(5)
     cf.setPayloadType(PayloadType.both)
     cf.setAutoIgnore(["concept: name", "Label", "lifecycle: transition"])
     cf.doForceTime()
     cf.setPayloadSettings("xray_settings.cfg")
     cf.dump("synth_tr_tagged.json")
     xray_map = {
         "synth_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.ignoreTagging(),
          "synth_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["T1", "T2", "T3"], 3),
         "synth_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["T1", "T2", "T3"]),
         "synth_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAnyProp(x,
                                           [(template_response, ["r1", "r2"])],
                                           SatCases.NotVacuitySat)
     }
     runWholeConfiguration(pipeline, "synth_tr_tagged.xes", cf, xray_map)

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


def run_bpm12(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)
    cf = ConfigurationFile()
    cf.setExperimentName("bpi12")
    cf.setLogName("bpi12.xes")
    cf.setOutputFolder("BPI12")
    cf.setMaxDepth(5)
    cf.setMinLeaf(5)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(
        ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
         "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"])
    cf.setPayloadSettings("bpi2012_settings.cfg")
    cf.dump("bpi12.json")
    bpm12_map = {
        "bpi12_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["W_Afhandelen leads","W_Completeren aanvraag"], 3),
        "bpi12_payload_6500": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "AMOUNT_REQ", "6500"),
        "bpi12_payload_45000": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "AMOUNT_REQ", "45000"),
        "bpi12_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["O_SENT", "W_Completeren aanvraag", "W_Nabellen incomplete dossiers"], 1),
        "bpi12_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["W_Completeren aanvraag", "W_Afhandelen leads"]),
        "bpi12_decl1": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [(template_absence1, ["A_DECLINED"])], SatCases.NotVacuitySat),
        "bpi12_decl2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [(template_precedence, ["O_ACCEPTED", "A_APPROVED"])], SatCases.NotVacuitySat)
    }
    runWholeConfiguration(pipeline, "bpi12.xes", cf, bpm12_map)

def run_bpm12(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)
    cf = ConfigurationFile()
    cf.setExperimentName("bpi12oc")
    cf.setLogName("bpi12_oc.xes")
    cf.setOutputFolder("BPI12oc")
    cf.setMaxDepth(5)
    cf.setMinLeaf(5)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(
        ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
         "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"])
    cf.setPayloadSettings("bpi2012_settings.cfg")
    cf.dump("bpi12oc.json")
    bpm12_map = {
        "bpi12oc_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["W_Afhandelen leads","W_Completeren aanvraag"], 3),
        "bpi12oc_payload_6500": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "AMOUNT_REQ", "6500"),
        "bpi12oc_payload_45000": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "AMOUNT_REQ", "45000"),
        "bpi12oc_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["O_SENT", "W_Completeren aanvraag", "W_Nabellen incomplete dossiers"], 1),
        "bpi12oc_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["W_Completeren aanvraag", "W_Afhandelen leads"]),
        "bpi12oc_decl1": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [(template_absence1, ["A_DECLINED"])], SatCases.NotVacuitySat),
        "bpi12oc_decl2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [(template_precedence, ["O_ACCEPTED", "A_APPROVED"])], SatCases.NotVacuitySat)
    }
    runWholeConfiguration(pipeline, "bpi12_oc.xes", cf, bpm12_map)

def run_bpm17(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)
    cf = ConfigurationFile()
    cf.setExperimentName("bpi17")
    cf.setLogName("bpi17.xes")
    cf.setOutputFolder("BPI17")
    cf.setMaxDepth(5)
    cf.setMinLeaf(5)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(
        ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
         "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"])
    cf.setPayloadSettings("bpi2017_settings.cfg")
    cf.dump("bpi17.json")
    bpm17_map = {
        "bpi17_payload_S": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Selected", True),
        "bpi17_payload_A": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Accepted", True),
        "bpi17_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["O_Sent (mail and online)", "O_Returned", "O_Accepted"], 1),
        "bpi17_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["O_Created", "O_Cancelled"]),
        "bpi17_decl1": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
            (template_absence1, ["O_Cancelled"])], SatCases.Sat),
        "bpi17_decl2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
            (template_precedence, ["O_Returned", "O_Accepted"])], SatCases.NotVacuitySat)
    }
    runWholeConfiguration(pipeline, "bpi17.xes", cf, bpm17_map)


def run_bpm19(pipeline):
    assert isinstance(pipeline, RunWholeStrategy)
    cf = ConfigurationFile()
    cf.setExperimentName("bpi19")
    cf.setLogName("bpi19.xes")
    cf.setOutputFolder("BPI19")
    cf.setMaxDepth(5)
    cf.setMinLeaf(5)
    cf.setSequenceThreshold(5)
    cf.setPayloadType(PayloadType.both)
    cf.setAutoIgnore(
        ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
         "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"])
    cf.setPayloadSettings("bpi2017_settings.cfg")
    cf.dump("bpi19.json")
    bpm17_map = {
        "bpi19_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["Create Purchase Order Item","Record Goods Receipt"]),
         "bpi19_payload_GR": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Goods Receipt", False),
         "bpi19_payload_A": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverTraceAttn(x, "Item Type", "Subcontracting"),
         "bpi19_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["Create Purchase Order Item","Cancel Goods Receipt","Clear Invoice"], 1),
         "bpi19_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["Cancel Goods Receipt"], 3),
         "bpi19_decl1": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
             (template_absence1, ["SRM: Created"])], SatCases.Sat),
         "bpi19_decl2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
             (template_absence1, ["SRM: Created"])], SatCases.Sat),
         "bpi19_decl3": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
             (template_response, ["Record Invoice Receipt", "Remove Payment Block"])], SatCases.NotVacuitySat)

    }
    runWholeConfiguration(pipeline, "bpi19.xes", cf, bpm17_map)



def run_traffic2(pipeline):
        assert isinstance(pipeline, RunWholeStrategy)
        cf = ConfigurationFile()
        cf.setExperimentName("traffic")
        cf.setLogName("traffic.xes")
        cf.setOutputFolder("TRAFFIC")
        cf.setMaxDepth(5)
        cf.setMinLeaf(5)
        cf.setSequenceThreshold(5)
        cf.setPayloadType(PayloadType.both)
        cf.setAutoIgnore(
            ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
             "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"])
        cf.setPayloadSettings("bpi2017_settings.cfg")
        cf.dump("traffic.json")
        bpm17_map = {
             "traffic_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, ["Create Fine","Payment"]),
             "traffic_payload_Pay36": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverEventAttn(x, "paymentAmount", 36.0),
            "traffic_payload_Art157": lambda
                x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithValueEqOverEventAttn(x, "article", 157),
             "traffic_mra_tra": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["Create Fine","Payment"], 2),
             "traffic_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithOccurrence(x, ["Add penalty", "Payment"], 1),
            "traffic_decl1": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
               (template_absence2, ["Payment"])], SatCases.NotVacuitySat),
            "traffic_decl2": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
               (template_exist, ["Insert Fine Notification"])], SatCases.NotVacuitySat),
            "traffic_decl3": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
              (template_response, ["Insert Date Appeal to Prefecture", "Add penalty"])], SatCases.NotVacuitySat)
        }
        runWholeConfiguration(pipeline, "traffic.xes", cf, bpm17_map)

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

          "bpi11_mr_tr": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactOccurrence(x, [
               "SGOT - Asat kinetic", "SGPT - alat kinetic", "Milk acid dehydrogenase LDH kinetic", "leukocyte count electronic"], 1),

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
           "bpi11_proc": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithExactSubsequence(x, [
               "unconjugated bilirubin", "bilirubin - total", "glucose"]),
           "bpi11_decl": lambda x: DevianceMiningPipeline.LogTaggingViaPredicates.tagLogWithSatAllProp(x, [
               (template_init, ["outpatient follow-up consultation"])
         ], SatCases.NotVacuitySat)
     }
     runWholeConfiguration(pipeline, "EnglishBPIChallenge2011.xes", cf, bpm11_map)


def oldPipeline():
     split_no = 2
     pipeline = RunWholeStrategy(os.path.join("data", "logs"),
                                 os.path.join("data", "experiments"),
                                 True,
                                 [5,10,15,20,25,30,35,40,60,80],
                                 split_no)
     #run_sepsis(pipeline)
     #run_bpm11(pipeline)
     #run_xray(pipeline)
     run_bpm12(pipeline)
     #run_synth(pipeline)
     #run_bank(pipeline)

     pipeline2 = RunWholeStrategy(os.path.join("data", "logs"),
                                 os.path.join("data", "experiments"),
                                 True,
                                 [5,10,15,20,25,30,35,40,60,80],
                                 3,
                                  0.3)
     #run_bpm17(pipeline2)
     #run_traffic2(pipeline2)

     pipeline3 = RunWholeStrategy(os.path.join("data", "logs"),
                                  os.path.join("data", "experiments"),
                                  False,
                                  [5, 10, 15, 20, 25],  #,30,35,40,60,80],
                                  3,
                                  0.6)
     #run_bpm19(pipeline3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #main(["", "xray_conf.txt"])
    #main(["", "xray2_conf.txt"])
    oldPipeline()