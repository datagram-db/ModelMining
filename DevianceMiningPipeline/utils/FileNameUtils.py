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

def trace_encodings(results_folder, encoding, split_nr):
        split = "split" + str(split_nr)
        file_loc = results_folder + "/" + split + "/" + encoding
        train_path = file_loc + "/" + "train_encodings.arff"
        test_path = file_loc + "/" + "test_encodings.arff"
        return {"train": os.path.abspath(train_path), "test": os.path.abspath(test_path)}