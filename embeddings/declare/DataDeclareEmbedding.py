import numpy as np
import pandas as pd

from dataloading.Log import Log, dict_union
from embeddings.declare.Utils import DeclareConstruct
from scikitutils.dt_printer import export_text2
from scikitutils.trainer import from_hyperparameters_instantiate_model, trainer, trainer_precision


def forEachRunCandidate(runTrainClause : DeclareConstruct,
                        runTestClause : DeclareConstruct,
                        trainLogPos : Log,
                        trainLogNeg : Log,
                        testLogPos : Log,
                        testLogNeg : Log,
                        conflist : list[from_hyperparameters_instantiate_model],
                        minScore : float):
    assert runTrainClause.isRun
    assert runTestClause.isRun
    from scikitutils.InputData import one_hot_encoding
    from scikitutils.InputData import pandasToEmbedding
    # First have to find all locations of fulfillments

    values = dict_union(trainLogPos.collectDistinctValues(), trainLogNeg.collectDistinctValues())
    toKeep1 = runTrainClause.fulfillments
    toKeep2 = runTestClause.fulfillments

    ## Fitting in the training data
    trepos1 = trainLogPos.collectValuesForPayloadEmbedding(values, None, toKeep1, False, False)
    # trepos1_ = trepos1.dropna(axis=0, how='all')
    nrow_trepos1 = len(trepos1.index)
    treneg1 = trainLogNeg.collectValuesForPayloadEmbedding(values, None, toKeep1, False, False)
    # treneg1_ = treneg1.dropna(axis=0, how='all')
    nrow_treneg1 = len(treneg1.index)
    a1 = pd.concat([trepos1, treneg1], ignore_index=True)
    a1 = a1.assign(Class=[1] * nrow_trepos1+[-1] * nrow_treneg1)
    e1 = pandasToEmbedding(a1.dropna(axis=0, how='all'))

    ## Similar approach for the testing one
    trepos2 = testLogPos.collectValuesForPayloadEmbedding(values, None, toKeep2, False, False)
    # trepos2_ = trepos2.dropna(axis=0, how='all')
    nrow_trepos2 = len(trepos2.index)
    treneg2 = testLogNeg.collectValuesForPayloadEmbedding(values, None, toKeep2, False, False)
    # treneg2_ = treneg2.dropna(axis=0, how='all')
    nrow_treneg2 = len(treneg2.index)
    a2 = pd.concat([trepos2, treneg2], ignore_index=True)
    a2 = a2.assign(Class=[1] * nrow_trepos2+[-1] * nrow_treneg2)
    e2 = pandasToEmbedding(a2)    ## Using the same hot encoding on both
    na1 = e1.dropna()
    na2 = e2.dropna()
    if len(na1.X.index) == 0 or len(na2.X.index) ==0:
        return None
    e1,e2 = one_hot_encoding(e1,e2)
    na1, na2 = one_hot_encoding(na1, na2)
    # na1.X = na1.X.values
    # na2.X = na2.X.values
    t = trainer(na1,na2,conflist,False)
    t.train_all()
    classifier,score = t.getBestClassifier(trainer_precision)
    if score < minScore:
        return None

    print("Returning: " +str(runTrainClause)+":Data")
    e1.X[np.isnan(e1.X)] = 0
    e2.X[np.isnan(e2.X)] = 0
    y_predTe = classifier.predict(e1.X[list(classifier.feature_names_in_)])
    y_predTr = classifier.predict(e2.X[list(classifier.feature_names_in_)])
    for idx, x in enumerate(toKeep1):
        if len(x)==0:
            y_predTe[idx] = 0
    for idx, x in enumerate(toKeep2):
        if len(x)==0:
            y_predTr[idx] = 0
    return (y_predTe.tolist(), y_predTr.tolist(), export_text2(classifier))



