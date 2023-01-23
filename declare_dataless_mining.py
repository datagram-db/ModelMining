from dataloading.Log import Log
import sys

def DeclareDataless_mining(log_path, filterCandidates=True, candidate_threshold=0.1, constraint_threshold=0.1, withData=False):
    from embeddings.declare.DatalessDeclareEmbedding import DeclareDevMining
    obj = DeclareDevMining()
    print("Data loading")
    log = Log(log_path, withData=withData)
    print("Mining")
    from time import time
    start = time()
    training, _ = obj.run(log, None, candidates=None,
                                                filterCandidates=filterCandidates,
                                                candidate_threshold=candidate_threshold,
                                                constraint_threshold=constraint_threshold)
    end = time()
    time = (end - start)
    print("MaxSat")
    training = training.sparse.to_dense()
    for col in training.columns:
        training.loc[training[col] >= 0, col] = 1
        training.loc[training[col] < 0, col] = 0
    max_sat = training.sum(axis=1) / len(training.columns)
    print(max_sat)

    print("Support :"+str(time))
    supp = training.sum() /len(training.index)
    print(supp)

if __name__ == '__main__':
    args = sys.argv[1:]
    DeclareDataless_mining("/home/giacomo/Scaricati/synthetic_logs(1)/1000_30_10.xes")