from dataloading.Log import Log
import sys

def DeclareDataless_mining(log_path, filterCandidates=True, candidate_threshold=0.1, constraint_threshold=0.1, withData=False):
    from embeddings.declare.DatalessDeclareEmbedding import DeclareDevMining
    obj = DeclareDevMining()
    print("Data loading")
    log = Log(log_path, withData=withData)
    print("Mining")
    training, _ = obj.run(log, None, candidates=None,
                                                filterCandidates=filterCandidates,
                                                candidate_threshold=candidate_threshold,
                                                constraint_threshold=constraint_threshold)
    print("MaxSat")
    training = training.sparse.to_dense()
    for col in training.columns:
        training.loc[training[col] >= 0, col] = 1
        training.loc[training[col] < 0, col] = 0
    sum = training.sum(axis=1)
    print(sum)

if __name__ == '__main__':
    args = sys.argv[1:]
    DeclareDataless_mining("/home/giacomo/PycharmProjects/trace_learning/data/training/bpi20D_decl2_1_true_true.xes")