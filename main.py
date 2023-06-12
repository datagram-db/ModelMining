# This is a sample Python script.
import sys
import datetime

import pandas

from dataloading import Log
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from embeddings.Embeddings import Embedding
import argparse
from scikitutils import trainer

# Press the green button in the gutter to run the script.
from scikitutils.trainer import get_classifier_configuration_from_file

if __name__ == '__main__':
    args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', required=True, action='store',
                        type=str, dest='experimentName', default="experiment_log_name", help="experiment name")
    parser.add_argument('-s', '--support', required=False, action='store',
                        type=float, dest='support', default=0.1, help="provide minimum support")
    parser.add_argument('-l', '--learner', required=False, action='store',
                        type=str, dest='learner', default="learner.yaml", help="provides the learner configuration")
    parser.add_argument('-p', '--trainingPositive', required=False, action='store',
                        type=str, dest='trp', default="data/training/bpi20D_decl2_1_true_true.xes", help="Training Positive: xes file")
    parser.add_argument('-n', '--trainingNegative', required=False, action='store',
                        type=str, dest='trn', default="data/training/bpi20D_decl2_1_true_true.xes", help="Training Negative: xes file")
    parser.add_argument('-P', '--testingPositive', required=False, action='store',
                        type=str, dest='tep', default="data/training/bpi20D_decl2_1_true_true.xes", help="Testing Positive: xes file")
    parser.add_argument('-N', '--testingNegative', required=False, action='store',
                        type=str, dest='ten', default="data/training/bpi20D_decl2_1_true_true.xes", help="Testing Negative: xes file")
    parser.add_argument('-o', '--output', required=True, action='store',
                        type=str, dest='output_file', default="benchmark.csv", help="provide directory for csv outputting")
    # parser.add_argument('-i', '--iterations', required=True, action='store',
    #                     type=int, dest='num_iters', default=False,
    #                     help="provide number of iterations for the algorithm to run")
    args = parser.parse_args()

    # Log.legacy_split_log("/home/giacomo/PycharmProjects/trace_learning/data","bpi20D_decl2_1.xes","/home/giacomo/PycharmProjects/trace_learning/data/training")
    # Log.legacy_split_log("/home/giacomo/PycharmProjects/trace_learning/data","bpi20D_decl2_2.xes","/home/giacomo/PycharmProjects/trace_learning/data/testing")


    loading_start = datetime.datetime.now()
    trP = Log.Log(args.trp, withData=True)
    trN = Log.Log(args.trn, withData=True)
    teP = Log.Log(args.tep, withData=True)
    teN = Log.Log(args.ten, withData=True)
    e = Embedding(trP, trN, teP, teN)
    loading_ends = datetime.datetime.now()
    loading_ms = (loading_ends-loading_start).total_seconds() * 1000

    # tr,te = e.IA_baseline_embedding()
    # tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_IA.csv", index=False)
    # te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_IA.csv", index=False)
    #
    # tr,te = e.DatalessTracewise_embedding()
    # tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_Sam.csv", index=False)
    # te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_Sam.csv", index=False)
    #
    # tr,te = e.Correlation_embedding()
    # tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_ICPM.csv", index=False)
    # te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_ICPM.csv", index=False)

    # tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_dDecl.csv", index=False)
    # te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_dDecl.csv", index=False)

    dec_tree_prec = 0.7
    classifier = get_classifier_configuration_from_file(args.learner)
    embedding1_start = datetime.datetime.now()
    trLE, teLE = e.DeclareDataless_embedding()
    trDE,teDE,_ = e.DeclareWithData_embedding(classifier, True, args.support, 0, dec_tree_prec, False, {"time:timestamp"})
    embedding1_ends = datetime.datetime.now()
    embedding1_ms = (embedding1_ends - embedding1_start).total_seconds() * 1000
    # Avoiding classes to be duplicated!
    trLE.pop("Class")
    teLE.pop("Class")
    tr = pandas.merge(trLE, trDE, left_index=True, right_index=True)
    te = pandas.merge(teLE, teDE, left_index=True, right_index=True)
    ETr = args.trp+"_trainingEmbedding.csv"
    ETe = args.tep+"_testingEmbedding.csv"
    tr.to_csv(ETr, index=False)
    te.to_csv(ETe, index=False)

    #l = Log.Log("/home/giacomo/projects/knobab/data/testing/bpic_2011/data/10/log.xes", withData=True)
    #embedding = l.Correlation_embedding(l.getEventSet(), 1)
    print("OK")
    t = trainer.trainer(None, None, None, False)
    t.init_after(ETr, ",", ETe, ",", classifier)
    t.train_all(loading_ms, embedding1_ms, args.support, args.experimentName, args.output_file)
    # for result in t.results:
    #     print(result.__dict__)
