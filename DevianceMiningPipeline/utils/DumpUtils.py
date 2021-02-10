import os
import pandas as pd
from .PandaExpress import ensureDataFrameQuality


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