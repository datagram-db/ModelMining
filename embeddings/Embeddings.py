from dataloading.Log import Log

def get_shared_activities(logPosTrain, logNegTrain, logPosTest, logNegTest):
    assert isinstance(logPosTrain, Log)
    assert isinstance(logNegTrain, Log)
    assert isinstance(logPosTest, Log)
    assert isinstance(logNegTest, Log)
    return set.union(logPosTrain.getEventSet(), logNegTrain.getEventSet(), logPosTest.getEventSet(), logNegTest.getEventSet())

