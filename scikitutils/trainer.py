import time
from abc import ABC, abstractmethod
from sklearn import tree
from .InputData import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score

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
                 testing_time):
        self.testing_time = testing_time
        self.training_time = training_time
        self.threshold = threshold
        self.recall = recall
        self.precision = precision
        self.per_class_precision = per_class_precision
        self.macro_precision = macro_precision
        self.micro_precision = micro_precision



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

    def train_all(self):
        for conf in self.dtLS:
            tofit = conf.gen()
            t1F = time.time()
            learner = tofit.fit(self.training.X, self.training.Y)
            t2F = time.time()
            y_pred = learner.predict(self.testing.X)
            t3F = time.time()
            learningTime = t2F - t1F
            classificationTime = t3F - t2F
            micro_precision = precision_score(y_pred, self.testing.Y, average='micro')
            macro_precision = precision_score(y_pred, self.testing.Y, average='macro')
            per_class_precision = precision_score(y_pred, self.testing.Y, average=None)
            precision, recall, thresholds = precision_recall_curve(self.testing.Y, y_pred)
            self.results.append(trainer_results(micro_precision,macro_precision,per_class_precision,precision,recall,thresholds,learningTime,classificationTime))