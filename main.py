# This is a sample Python script.
from dataloading import Log
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scikitutils import trainer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    l = Log.Log("/home/giacomo/projects/knobab/data/testing/bpic_2011/data/10/log.xes", withData=True)
    embedding = l.Correlation_embedding(l.getEventSet(), 1)
    print("OK")
    # t = trainer.trainer("/home/giacomo/Scaricati/titanic/train.csv",",",
    #                     "/home/giacomo/Scaricati/titanic/testing.csv",",",
    #                     [trainer.decision_tree_hyperparameters()])
    # t.train_all()
    # for result in t.results:
    #     print(result.__dict__)
