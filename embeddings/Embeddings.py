import itertools
from copy import deepcopy

import numpy as np
import pandas as pd

from dataloading.Log import Log, dict_union
from embeddings.declare.DataDeclareEmbedding import forEachRunCandidate
from scikitutils.trainer import from_hyperparameters_instantiate_model


def IA_baseline_embedding(self, shared_activities=None, label=0):
    if shared_activities is None:
        shared_activities = self.unique_events
    train_data = list()
    clazz = [label] * self.getNTraces()
    for i in range(self.getNTraces()):
        trace = self.getIthTrace(i)
        train_data.append(list(map(lambda x: trace.eventCount(x), shared_activities)))
    np_train_data = np.array(train_data)
    train_df = pd.DataFrame(np_train_data)
    train_df.columns = shared_activities
    train_df["Class"] = clazz
    train_df = train_df.fillna(0)
    return train_df


def DatalessTracewise_embedding(self, shared_activities=None, label=0, maxTraceLength=-1):
    if shared_activities is None:
        shared_activities = self.unique_events
    if maxTraceLength == -1:
        maxTraceLength = self.max_length
    beta = dict()
    train_data = list()
    clazz = [label] * self.getNTraces()
    for idx, x in enumerate(shared_activities):
        beta[x] = idx
    for i in range(self.getNTraces()):
        space = [-1] * maxTraceLength
        for idx, x in enumerate(self.getIthTrace(i).getStringTrace()):
            space[idx] = beta[x]
        train_data.append(space)
    np_train_data = np.array(train_data)
    train_df = pd.DataFrame(np_train_data)
    train_df.columns = list(map(lambda x: str(x), range(maxTraceLength)))
    train_df["Class"] = clazz
    train_df = train_df.fillna(0)
    return train_df


def Correlation_embedding(self, shared_activities=None, label=0, lambda_=0.9):
    if shared_activities is None:
        shared_activities = self.unique_events
    embedding_space = list(shared_activities) + list(
        map(lambda x: x[0] + "-" + x[1], itertools.product(shared_activities, shared_activities)))
    embedding_name_to_offset = dict()
    train_data = list()
    clazz = [label] * self.getNTraces()
    for idx, x in enumerate(embedding_space):
        embedding_name_to_offset[x] = idx
    for i in range(self.getNTraces()):
        space = [0] * len(embedding_space)
        trace = self.getIthTrace(i).getStringTrace()
        space[embedding_name_to_offset[trace[0]]] = 1
        for j, y in enumerate(trace):
            k = j
            for z in trace[j:]:
                spaceIdx = embedding_name_to_offset[y + "-" + z]
                space[spaceIdx] = space[spaceIdx] + pow(lambda_, k - j + 1)
                k = k + 1
        train_data.append(space)
    np_train_data = np.array(train_data)
    train_df = pd.DataFrame(np_train_data)
    train_df.columns = embedding_space
    train_df["Class"] = clazz
    train_df = train_df.fillna(0)
    return train_df, embedding_space

def extractPayloadEmbedding(logPos : Log,
                            logNeg : Log,
                            ignoreFields :
                            set[str], values = None):
    if values is None:
        values = dict_union(logPos.collectDistinctValues(ignoreFields), logNeg.collectDistinctValues(ignoreFields))
    dfPos = logPos.collectValuesForPayloadEmbedding(values, ignoreFields)
    dfNeg = logNeg.collectValuesForPayloadEmbedding(values, ignoreFields)
    ## Dimension referring to the trace length
    trace_lengthPos = []
    trace_lengthNeg = []
    for i in range(logPos.getNTraces()):
        trace = logPos.getIthTrace(i)
        trace_lengthPos.append(trace.length)
    for i in range(logNeg.getNTraces()):
        trace = logPos.getIthTrace(i)
        trace_lengthNeg.append(trace.length)
    dfPos["@len(trace)"] = trace_lengthPos
    dfNeg["@len(trace)"] = trace_lengthNeg
    training = pd.concat([dfPos, dfNeg], ignore_index=True)
    training = training.fillna(0)
    return training, values


class Embedding:
    def __init__(self, posLogTr:Log, negLogTr:Log, posLogTe:Log, negLogTe:Log):
        self.posTr = posLogTr
        self.negTr = negLogTr
        self.posTe = posLogTe
        self.negTe = negLogTe
        self.trainingCandidates = None
        self.testingCandidates = None
        assert isinstance(self.posTr, Log)
        assert isinstance(self.negTr, Log)

    def IA_baseline_embedding(self):
        s = set.union(self.posTr.unique_events, self.negTr.unique_events)
        df1 = IA_baseline_embedding(self.posTr, s, 1)
        df2 = IA_baseline_embedding(self.negTr, s, 0)
        training = pd.concat([df1, df2], ignore_index=True)
        df1 = IA_baseline_embedding(self.posTe, s, 1)
        df2 = IA_baseline_embedding(self.negTe, s, 0)
        testing = pd.concat([df1, df2], ignore_index=True)
        return training, testing

    def DatalessTracewise_embedding(self):
        maxTraceLength = max(self.posTr.max_length, self.negTr.max_length, self.posTe.max_length, self.negTe.max_length)
        s = set.union(self.posTr.unique_events, self.negTr.unique_events)
        df1 = DatalessTracewise_embedding(self.posTr, s, 1, maxTraceLength)
        df2 = DatalessTracewise_embedding(self.negTr, s, 0, maxTraceLength)
        training = pd.concat([df1, df2], ignore_index=True)
        df1 = DatalessTracewise_embedding(self.posTe, s, 1, maxTraceLength)
        df2 = DatalessTracewise_embedding(self.negTe, s, 0, maxTraceLength)
        testing = pd.concat([df1, df2], ignore_index=True)
        return training, testing

    def Correlation_embedding(self, lambda_=.9):
        s = set.union(self.posTr.unique_events, self.negTr.unique_events)
        df1, e1 = Correlation_embedding(self.posTr, s, 1, lambda_)
        df2, f1 = Correlation_embedding(self.negTr, s, 0, lambda_)
        assert (e1 == f1)
        training = pd.concat([df1, df2], ignore_index=True)
        df1, e = Correlation_embedding(self.posTe, s, 1, lambda_)
        df2, f = Correlation_embedding(self.negTe, s, 0, lambda_)
        assert (e == f)
        assert (e == e1)
        testing = pd.concat([df1, df2], ignore_index=True)
        return training, testing

    def DeclareDataless_embedding(self, filterCandidates=True, candidate_threshold=0.1, constraint_threshold=0.1):
        from embeddings.declare.DatalessDeclareEmbedding import DeclareDevMining
        obj = DeclareDevMining()
        training, self.trainingCandidates = obj.run(self.posTr, self.negTr, candidates=None, filterCandidates=filterCandidates,
                                                    candidate_threshold=candidate_threshold, constraint_threshold=constraint_threshold)
        testing, self.testingCandidates = obj.run(self.posTe, self.negTe, candidates=deepcopy(self.trainingCandidates), filterCandidates=filterCandidates,
                             candidate_threshold=candidate_threshold, constraint_threshold=constraint_threshold)
        return training, testing

    def Payload_embedding(self, ignoreKeys = None):
        training, candidates = extractPayloadEmbedding(self.posTr, self.negTr, ignoreKeys, None)
        testing, _ = extractPayloadEmbedding(self.posTe, self.negTe, ignoreKeys, candidates)
        return training, testing

    def DeclareWithData_embedding(self, conflist: list[from_hyperparameters_instantiate_model],
                                        filterCandidates=True,
                                        candidate_threshold=0.7,
                                        constraint_threshold=0.7,
                                        minScore = 0.8,
                                        allValues = False,
                                  ignoreKeys = None):
        dTrain = {}
        dTest = {}
        classifier = {}
        if (self.trainingCandidates is None):
            self.DeclareDataless_embedding(filterCandidates, candidate_threshold, constraint_threshold)
        for (trC, teC) in zip(self.trainingCandidates, self.testingCandidates):
            if not trC.supportFV:
                continue
            tup = forEachRunCandidate(trC, teC, self.posTr, self.negTr, self.posTe, self.negTe, conflist, minScore, allValues, ignoreKeys)
            if tup is None:
                continue
            trainClass, testClass, treeString = tup
            k = str(trC)+":"+treeString
            dTrain[k] = trainClass
            dTest[k] = testClass
            classifier[k] = treeString

        dTrain["Class"] = [1]*self.posTr.getNTraces() + [0]*self.negTr.getNTraces()
        dTest["Class"] = [1] * self.posTe.getNTraces() + [0] * self.negTe.getNTraces()
        return pd.DataFrame(dTrain), pd.DataFrame(dTest), classifier









