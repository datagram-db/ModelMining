# This is a sample Python script.
from dataloading import Log
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from embeddings.Embeddings import Embedding

from scikitutils import trainer

# Press the green button in the gutter to run the script.
from scikitutils.trainer import get_classifier_configuration_from_file

if __name__ == '__main__':
    # Log.legacy_split_log("/home/giacomo/PycharmProjects/trace_learning/data","bpi20D_decl2_1.xes","/home/giacomo/PycharmProjects/trace_learning/data/training")
    # Log.legacy_split_log("/home/giacomo/PycharmProjects/trace_learning/data","bpi20D_decl2_2.xes","/home/giacomo/PycharmProjects/trace_learning/data/testing")
    trP = Log.Log("/home/giacomo/PycharmProjects/trace_learning/data/training/bpi20D_decl2_1_true_true.xes", withData=True)
    trN = Log.Log("/home/giacomo/PycharmProjects/trace_learning/data/training/bpi20D_decl2_1_false_false.xes", withData=True)
    teP = Log.Log("/home/giacomo/PycharmProjects/trace_learning/data/testing/bpi20D_decl2_2_true_true.xes",
                  withData=True)
    teN = Log.Log("/home/giacomo/PycharmProjects/trace_learning/data/testing/bpi20D_decl2_2_false_false.xes",
                  withData=True)

    e = Embedding(trP, trN, teP, teN)
    tr,te = e.IA_baseline_embedding()
    tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_IA.csv", index=False)
    te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_IA.csv", index=False)

    tr,te = e.DatalessTracewise_embedding()
    tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_Sam.csv", index=False)
    te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_Sam.csv", index=False)

    tr,te = e.Correlation_embedding()
    tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_ICPM.csv", index=False)
    te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_ICPM.csv", index=False)

    tr,te = e.DeclareDataless_embedding()
    tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_dDecl.csv", index=False)
    te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_dDecl.csv", index=False)

    tr,te = e.DeclareWithData_embedding(get_classifier_configuration_from_file("/home/giacomo/PycharmProjects/trace_learning/learner.yaml"))
    tr.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/training_e_Payload.csv", index=False)
    te.to_csv("/home/giacomo/PycharmProjects/trace_learning/data/testing_e_Payload.csv", index=False)

    #l = Log.Log("/home/giacomo/projects/knobab/data/testing/bpic_2011/data/10/log.xes", withData=True)
    #embedding = l.Correlation_embedding(l.getEventSet(), 1)
    print("OK")
    # t = trainer.trainer("/home/giacomo/Scaricati/titanic/train.csv",",",
    #                     "/home/giacomo/Scaricati/titanic/testing.csv",",",
    #                     [trainer.decision_tree_hyperparameters()])
    # t.train_all()
    # for result in t.results:
    #     print(result.__dict__)
