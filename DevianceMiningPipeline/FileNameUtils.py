"""
Utility Names for the files as strings

@author: Giacomo Bergami
"""
import os

def getXesName(log_path, logNr):
    return log_path.format(logNr + 1)

def embedding_path(logNr, results_folder, strategyName):
    return os.path.join(results_folder, "split" + str(logNr + 1), strategyName)

def baseline_path(logNr, results_folder):
    return embedding_path(logNr, results_folder, "baseline")

def declare_path(logNr, results_folder):
    return embedding_path(logNr, results_folder, "declare")

def declare_data_aware_path(logNr, results_folder):
    return embedding_path(logNr, results_folder, "dwd")

def payload_path(logNr, results_folder):
    return embedding_path(logNr, results_folder, "payload")