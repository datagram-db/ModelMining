"""
Generating one single file for all the Pandas utils

@author: Giacomo Bergami
"""
import os
import pandas as pd

def ensureDataFrameQuality(df):
    return df.sort_index()

def serialize(df, path, index = False):
    df = ensureDataFrameQuality(df)
    if not df.empty:
        df.to_csv(path, index=index)