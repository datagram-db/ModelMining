import pandas as pd

from dataloading.Log import Log, dict_union


def extractEmbedding(logPos : Log,logNeg : Log, ignoreFields : set[str], values = None):
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
