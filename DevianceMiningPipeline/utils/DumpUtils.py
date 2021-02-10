import os
import pandas as pd

from .FileNameUtils import path_generic_log
from .PandaExpress import ensureDataFrameQuality, dataframe_join_withChecks
from scipy.io import arff

def read_single_arff_dump(arff_file, csv_file, doQualityCheck = True):
    arff_pd  = pd.DataFrame(arff.loadarff(os.path.abspath((arff_file)))[0])
    arff_pd.rename(columns={"label": "Label"}, inplace=True)
    getOnlyLabels = pd.read_csv((csv_file), sep=";", index_col="Case_ID", na_filter=False, usecols=["Case_ID", "Label"])
    assert (len(arff_pd.index) == len(getOnlyLabels.index))
    assert ("Case_ID" not in arff_pd)
    arff_pd.index = getOnlyLabels.index
    arff_pd["Case_ID"] = getOnlyLabels.index
    if doQualityCheck:
        assert len(dataframe_join_withChecks(arff_pd, getOnlyLabels).index) == len(arff_pd.index)
    return arff_pd

def read_arff_embedding_dump(complete_path, dictionary, doQualityCheck = True):
    arff_training = os.path.join(complete_path, "train_encodings.arff")
    arff_testing = os.path.join(complete_path, "test_encodings.arff")
    csv_training = os.path.join(complete_path, "crosstrain.csv")
    csv_testing = os.path.join(complete_path, "crosstest.csv")

    dictionary["train"] = os.path.abspath(arff_training)
    dictionary["test"] = os.path.abspath(arff_testing)

    train_df = ensureDataFrameQuality(read_arff_embedding_dump(arff_training, csv_training, doQualityCheck))
    test_df = ensureDataFrameQuality(read_arff_embedding_dump(arff_training, csv_training, doQualityCheck))
    return train_df, test_df



def read_generic_embedding_dump(results_folder, split_nr, encoding, dictionary):
    """
    This method reads the log, that has been already serialized for a vectorial representation

    :param results_folder:  Folder from which we have to read the serialization
    :param split_nr:        Number of current fold for the k-fold
    :param encoding:        Encoding stored in the folder
    :return:
    """
    split = "split" + str(split_nr)
    file_loc = os.path.join(results_folder, split, encoding)
    train_path = os.path.join(file_loc, encoding+"_train.csv")
    test_path = os.path.join(file_loc, encoding+"_test.csv")
    dictionary["train"] = os.path.abspath(train_path)
    dictionary["test"] = os.path.abspath(test_path)
    train_df = ensureDataFrameQuality(pd.read_csv(train_path, sep=",", index_col="Case_ID", na_filter=False))
    test_df = ensureDataFrameQuality(pd.read_csv(test_path, sep=",", index_col="Case_ID", na_filter=False))
    return train_df, test_df

def dump_extended_dataframes(train_df, test_df, results_folder, split_nr, encoding):
    train_path, test_path = path_generic_log(results_folder, split_nr, encoding)
    print("Dumping extended data frames into " + train_path +" and "+test_path)
    new_cols = [col for col in train_df.columns if col != 'Label'] + ['Label']
    train_df[new_cols].to_csv(train_path, index=False)
    test_df[new_cols].to_csv(test_path, index=False)
    return (train_path, test_path)

def multidump_compact(results_folder, elements, forMultiDump, payload_test_df, payload_train_df, split_nr):
        tr_f, t_f = dump_extended_dataframes(payload_train_df, payload_test_df, results_folder, split_nr,
                                             forMultiDump)
        d = dict()
        d["train"] = os.path.abspath(tr_f)
        d["test"] = os.path.abspath(t_f)
        elements.append(d)