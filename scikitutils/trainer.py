import os
import sys
import time
from abc import ABC, abstractmethod

import pandas as pd
from sklearn import tree
from .InputData import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import yaml
from yaml.loader import SafeLoader

class from_hyperparameters_instantiate_model(ABC):
    @abstractmethod
    def gen(self):
        pass

class decision_tree_hyperparameters(from_hyperparameters_instantiate_model):
    def __init__(self, criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0):
        self.ccp_alpha = ccp_alpha
        self.class_weight = class_weight
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.max_features = max_features
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.splitter = splitter
        self.criterion = criterion

    def __init__(self, d : dict):
        if (d["type"] == "scikit_decision_tree"):
            self.ccp_alpha = d["ccp_alpha"]
            self.class_weight = d["class_weight"]
            self.min_impurity_decrease = d["min_impurity_decrease"]
            self.max_leaf_nodes = d["max_leaf_nodes"]
            self.random_state = d["random_state"]
            self.max_features = d["max_features"]
            self.min_weight_fraction_leaf = d["min_weight_fraction_leaf"]
            self.min_samples_leaf = d["min_samples_leaf"]
            self.min_samples_split = d["min_samples_split"]
            self.max_depth = d["max_depth"]
            self.splitter = d["splitter"]
            self.criterion = d["criterion"]

    def gen(self):
        return tree.DecisionTreeClassifier(ccp_alpha = self.ccp_alpha, class_weight = self.class_weight, min_impurity_decrease = self.min_impurity_decrease, max_leaf_nodes = self.max_leaf_nodes, random_state = self.random_state, max_features = self.max_features, min_weight_fraction_leaf = self.min_weight_fraction_leaf, min_samples_leaf = self.min_samples_leaf, min_samples_split = self.min_samples_split, max_depth = self.max_depth, splitter = self.splitter, criterion = self.criterion)

class trainer_results:
    def __init__(self, micro_precision,
                 macro_precision,
                 per_class_precision,
                 precision,
                 recall,
                 threshold,
                 training_time,
                 testing_time,
                 loading_data_time,
                 embedding_extract_time,
                 support,
                 experimentName):
        self.experimentName = experimentName
        self.support = support
        self.loading_data_time = loading_data_time
        self.embedding_extract_time = embedding_extract_time
        self.testing_time = testing_time
        self.training_time = training_time
        self.threshold = threshold
        self.recall = recall
        self.precision = precision
        self.per_class_precision = per_class_precision
        self.macro_precision = macro_precision
        self.micro_precision = micro_precision

def trainer_precision(x : trainer_results):
    return sum(x.precision) / len(x.precision)

def trainer_factory_method(d : dict):
    if "type" in d:
        type = d["type"]
        if (type == "scikit_decision_tree"):
            return decision_tree_hyperparameters(d)
        else:
            return None
    else:
        return None

def get_classifier_configuration_from_file(file_conf_path):
    with open(file_conf_path) as f:
        return list(map(lambda x: trainer_factory_method(x), yaml.load(f, Loader=SafeLoader)))
    return list()


def resultsToCSVFile(r : list[trainer_results], filename):
    pd.DataFrame(map(lambda x : x.__dict__, r)).to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))

class trainer:
    def __init__(self, fileTr,
                       deleTr,
                       fileT,
                       deleT,
                       classifiers_conf):
        self.dtLS = classifiers_conf
        self.training = load_embedding(fileTr, deleTr)
        self.testing = load_embedding(fileT, deleT)
        (self.training, self.testing) = one_hot_encoding(self.training, self.testing)
        self.results = list()

    def __init__(self, training, testing, classifiers_conf, doOneHotEncoding=True):
        self.dtLS = classifiers_conf
        self.training = training
        self.testing = testing
        if doOneHotEncoding:
            (self.training, self.testing) = one_hot_encoding(self.training, self.testing)
        self.results = list()

    def getBestClassifier(self, f):
        idx = -1
        maxVal = -sys.maxsize - 1
        for i, res in enumerate(self.results):
            val = f(res)
            if (val > maxVal):
                maxVal = val
                idx = i
        if idx == -1:
            return None
        else:
            return (self.dtLS[idx].learner,maxVal)

    def train_all(self, loading_data_time, embedding_extract_time, support, experimentName, filename = None):
        classifiers = []
        from scikitutils.dt_printer import get_rules
        for conf in self.dtLS:
            tofit = conf.gen()
            t1F = time.time()
            conf.learner = tofit.fit(self.training.X, self.training.Y)
            t2F = time.time()
            y_pred = conf.learner.predict(self.testing.X)
            t3F = time.time()
            learningTime = t2F - t1F
            classificationTime = t3F - t2F
            classifiers.append(get_rules(conf.learner, self.training.X.columns, conf.learner.classes_))
            micro_precision = precision_score(y_pred, self.testing.Y, average='micro')
            macro_precision = precision_score(y_pred, self.testing.Y, average='macro')
            per_class_precision = precision_score(y_pred, self.testing.Y, average=None)
            precision, recall, thresholds = precision_recall_curve(self.testing.Y, y_pred)
            self.results.append(trainer_results(micro_precision,macro_precision,per_class_precision,precision,recall,thresholds,learningTime,classificationTime,loading_data_time,embedding_extract_time,support,experimentName))
        if filename is not None:
            resultsToCSVFile(self.results, filename)
            with open(filename+'_classifiers.txt', 'w') as f:
                f.writelines(classifiers)

def trainFromConfiguration(posnegTr, posnegTe, dele, conf, benchmark_file):
    trainer(posnegTr, dele, posnegTe, dele, get_classifier_configuration_from_file(conf))
    trainer.train_all(benchmark_file)
