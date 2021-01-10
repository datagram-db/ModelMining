# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import LogGeneration
import RulesExtraction
import Other
import jsonpickle
import sys

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

def output_pos_neg_test():
    INP_PATH = "logs/"
    EXP_NAME = "output_pos_neg_data"
    LOG_NAME = "output_pos_and_neg.xes"
    OUTPUTFOLDER = "payload/"
    results_folder = "output_pos_neg_res"
    log_path_seq = "output_pos_and_neg_{}.xes"
    results_file = "output_pos_neg.txt"

    payload = True
    payload_settings = "output_pos_and_neg_settings.cfg"


    for nr, i in enumerate((5, 15, 25)):
        ex = ExperimentRunner(experiment_name=EXP_NAME, output_file=results_file, results_folder=results_folder,
                              inp_path=INP_PATH, log_name=LOG_NAME, output_folder=OUTPUTFOLDER,
                              log_template=log_path_seq, dt_max_depth=10, dt_min_leaf=10,
                              selection_method="coverage", coverage_threshold=i, sequence_threshold=5,
                              payload=payload, payload_settings=payload_settings)

        with open("train_" + results_file, "a+") as f:
            f.write("\n")
        with open("test_" + results_file, "a+") as f:
            f.write("\n")

        #if nr == 0:
        #    ex.prepare_cross_validation()
        #    ex.prepare_data()
        ex.train_and_eval_benchmark()
    # ex.clean_data()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #pass
    conf_file = "sepsis_er.json"
    if len(sys.argv)>1:
        conf_file = sys.argv[1]
    preprocess = True
    if len(sys.argv)>2:
        test = sys.argv[2]
        preprocess = not (test == "skipPreprocessing")
    # run_complete_configuration_and_run("sepsis_er.json") --> Ok
    run_complete_configuration_and_run(conf_file, preprocess)
    #RulesExtraction.get_dtrules()               ## Requires: snapshots
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
