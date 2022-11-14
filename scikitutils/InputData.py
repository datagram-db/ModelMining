import numpy as np
import pandas as pd

class embedding:
    def __init__(self, X, Y, keynames,catcols):
        self.X = X
        self.Y = Y
        self.keynames = keynames
        self.catcols= catcols

    def dropna(e1, update=False):
        if isinstance(e1.X, pd.DataFrame):
            mic = e1.X.dropna(how='all')
            if update:
                e1.X = mic
            else:
                return embedding(mic,e1.Y[mic.index],e1.keynames,e1.catcols)
        else:
            mic = ~np.all(np.isnan(e1.X), axis=1)
            if update:
                e1.X = e1.X[mic]
            else:
                return embedding(e1.X[mic],e1.Y[mic],e1.keynames,e1.catcols)


def pandasToEmbedding(data : pd.DataFrame):
    Y = data.iloc[:,-1:].values
    data = data.iloc[:, :-1]
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    categorical_columns = list(set(cols) - set(num_cols))
    key_names = list(data.columns)
    return embedding(data,Y,key_names,categorical_columns)

def load_embedding(file, dele=","):
    data= pd.read_csv(file, sep=dele)
    return pandasToEmbedding(data)

def one_hot_encoding(e1, e2):
    N = len(e1.X.index)
    S = set.union(set(e1.X.columns),set(e2.X.columns))
    merged = pd.concat([e1.X, e2.X], ignore_index=True)
    merged = merged.fillna(0)
    categorical_cols = e1.catcols + list(set(e2.catcols) - set(e1.catcols))
    categorical_cols = set(filter(lambda x: x in S, categorical_cols))
    merged = pd.get_dummies(merged, columns=list(categorical_cols))
    e1.X = merged.head(N).copy(deep=True)
    e2.X = merged.iloc[N:]
    return (e1,e2)



def df_one_hot_encoding(df1, df2, catcol1, catcol2):
    N = len(df1.index)
    merged = pd.concat([df1, df2], ignore_index=True)
    S = set.union(set(df1.columns), set(df2.columns))
    merged = merged.fillna(0)
    categorical_cols = catcol1 + list(set(catcol2) - set(catcol1))
    categorical_cols = set(filter(lambda x: x in S, categorical_cols))
    merged = pd.get_dummies(merged, columns=categorical_cols)