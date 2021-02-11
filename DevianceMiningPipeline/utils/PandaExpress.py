"""
Generating one single file for all the Pandas utils

@author: Giacomo Bergami
"""
import os
from functools import reduce

import pandas as pd

from DevianceMiningPipeline.utils.FileNameUtils import path_generic_log


def ensureDataFrameQuality(df):
    assert ('Case_ID' in df.columns)
    assert ('Label' in df.columns)
    return df.sort_index()

def ensureLoadedDataQuality(df):
    if "Case_ID" not in df.columns:
        df["Case_ID"] = df.index
    assert ('Label' in df.columns)
    return df.sort_index()


def serialize(df, path, index = False):
    df = ensureDataFrameQuality(df)
    if not df.empty:
        df.to_csv(path, index=index)

def ExportDFRowNamesAsSets(test_df, train_df):
    return set(train_df["Case_ID"].to_list()), set(test_df["Case_ID"].to_list())

def ExportDFRowNamesAsLists(test_df, train_df):
    return list(train_df["Case_ID"].to_list()), list(test_df["Case_ID"].to_list())

def extendDataFrameWithLabels(df, map_rowid_to_label):
    assert ('Case_ID' in df.columns)
    assert ('Label' not in df.columns)
    ls = list()
    for trace_id in df["Case_ID"]:
        ls.append(map_rowid_to_label[trace_id])
    df["Label"] = ls
    return df

def dataframe_join_withChecks(left, right):
    j = None
    idTest = ('Case_ID' in left.columns) and ('Case_ID' in right.columns)
    j = left.join(right, lsuffix='_left', rsuffix='_right')
    if idTest:
        assert ((list(map(lambda x: str(x), j["Case_ID_right"].to_list())) == list(map(lambda x: str(x), j["Case_ID_left"].to_list()))))
        assert ((list(map(lambda x: str(x), j["Case_ID_right"].to_list())) == list(map(lambda x: str(x), j.index))))
    assert ((list(map(lambda x: int(x), j["Label_right"].to_list())) == list(map(lambda x: int(x), j["Label_left"].to_list()))))
    assert ('Label_left' in j)
    assert ('Label_right' in j)
    j.drop("Label_left", axis=1, inplace=True)
    if idTest:
        assert (('Case_ID_left' in j) and ('Case_ID_right' in j))
        j.drop("Case_ID_left", axis=1, inplace=True)
        j.rename(columns={'Label_right': 'Label', 'Case_ID_right': 'Case_ID'}, inplace=True)
    else:
        assert ('Case_ID' in j)
    return j

def dataframe_multiway_equijoin(ls):
    return reduce(dataframe_join_withChecks, ls)

