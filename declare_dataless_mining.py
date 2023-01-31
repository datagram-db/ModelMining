import argparse
import csv
import os.path
import sys
import time

from dataloading.Log import Log
from embeddings.declare.DatalessDeclareEmbedding import DeclareDevMining


def DeclareDataless_mining(log_path, filterCandidates=True, candidate_threshold=0.1, constraint_threshold=0.1,
                           withData=False, output_file="", iteration_num=0):
    with open(output_file, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow(['log_filename', 'mining_algorithm', 'n_traces', 'log_trace_average_length', 'min_support',
                             'iteration_num',
                             'loading_and_indexing_time', 'query_time'])

        obj = DeclareDevMining()

        start = time.time()
        # print("Data loading")
        log = Log(log_path, withData=withData)
        end = time.time()
        loading_time = (end - start) * 1000

        # print("Mining")
        start = time.time()
        training, _ = obj.run(log, None, candidates=None,
                              filterCandidates=filterCandidates,
                              candidate_threshold=candidate_threshold,
                              constraint_threshold=constraint_threshold)

        end = time.time()
        mining_time = (end - start) * 1000

        writer.writerow(
            [log_path, 'ADM_S', len(log.traces), log.max_length, candidate_threshold, iteration_num, loading_time,
             mining_time])
        print("MaxSat")
        training = training.sparse.to_dense()
        for col in training.columns:
            training.loc[training[col] >= 0, col] = 1
            training.loc[training[col] < 0, col] = 0
        sum = training.sum(axis=1)
        print(sum)

        sum1 = training.sum(axis=0)
        sum1 /= len(sum.index)
        # sum1 = sum1[sum1 == 1]
        # print(sum1)
        #
        with open("/home/sam/Documents/Repositories/CodeBases/ModelMining/mined_clauses.csv", 'a',
                   encoding='UTF8') as f1:
             writer1 = csv.writer(f1)
        
             if f1.tell() == 0:
                 writer1.writerow(['algorithm', 'clause', 'support'])
        
             for result in sum1.index:
                 writer1.writerow(['ADM_S', result, sum1[result]])


if __name__ == '__main__':
    args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--support', required=True, action='store',
                        type=float, dest='support', default=False, help="provide minimum support")
    parser.add_argument('-x', '--xes', required=True, action='store',
                        type=str, dest='log_file', default=False, help="provide xes log file")
    parser.add_argument('-o', '--output', required=True, action='store',
                        type=str, dest='output_file', default=False, help="provide directory for csv outputting")
    parser.add_argument('-i', '--iterations', required=True, action='store',
                        type=int, dest='num_iters', default=False,
                        help="provide number of iterations for the algorithm to run")
    args = parser.parse_args()

    for i in range(args.num_iters):
        DeclareDataless_mining(log_path=args.log_file, candidate_threshold=args.support, constraint_threshold=0,
                               output_file=args.output_file, iteration_num=i)
