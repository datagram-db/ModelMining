import numpy as np
import pandas as pd

class embedding:
    def __init__(self, X, Y, keynames,catcols):
        self.X = X
        self.Y = Y
        self.keynames = keynames
        self.catcols= catcols

def load_embedding(file, dele=","):
    data= pd.read_csv(file, sep=dele)
    Y = data.iloc[:,-1:].values
    data = data.iloc[:, :-1]
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    categorical_columns = list(set(cols) - set(num_cols))
    key_names = list(data.columns)
    return embedding(data,Y,key_names,categorical_columns)

def one_hot_encoding(e1, e2):
    N = len(e1.X.index)
    merged = pd.concat([e1.X, e2.X], ignore_index=True)
    merged = merged.fillna(0)
    categorical_cols = e1.catcols + list(set(e2.catcols) - set(e1.catcols))
    merged = pd.get_dummies(merged, columns=categorical_cols)
    e1.X = merged.head(N).copy(deep=True)
    e2.X = merged.iloc[N:]
    return (e1,e2)