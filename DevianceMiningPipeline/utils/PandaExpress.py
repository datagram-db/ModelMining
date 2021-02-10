"""
Generating one single file for all the Pandas utils

@author: Giacomo Bergami
"""
import os
import pandas as pd

def ensureDataFrameQuality(df):
    assert ('Case_ID' in df.columns)
    return df.sort_index()

def serialize(df, path, index = False):
    df = ensureDataFrameQuality(df)
    if not df.empty:
        df.to_csv(path, index=index)

def ExportDFRowNames(test_df, train_df):
    return set(train_df["Case_ID"].to_list()), set(test_df["Case_ID"].to_list())