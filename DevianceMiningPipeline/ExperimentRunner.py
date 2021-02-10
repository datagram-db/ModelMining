"""
Full pipeline for deviance mining experiments

Author: Joonas Puura
"""

from random import shuffle
import os

import pandas as pd
import numpy as np

import shutil
import yaml

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2

from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from opyenxes.factory.XFactory import XFactory

from . import baseline_runner, declaredevmining, model
from .deviancecommon import read_XES_log
from .sequence_runner import run_sequences, generateSequences
from .ddm_newmethod_fixed_new import data_declare_main, declare_data_aware_embedding, fisher_calculation

from sklearn.preprocessing import StandardScaler

## Remove features ...
from skfeature.function.similarity_based import fisher_score
from sklearn.decomposition import PCA
from collections import defaultdict

from .payload_extractor import payload_extractor2, payload_embedding
from .utils import *


from .utils.DumpUtils import read_generic_embedding_dump, multidump_compact, read_arff_embedding_dump
from .utils.FileNameUtils import trace_encodings, path_generic_log, extract_file_name_for_dump
from .utils.PandaExpress import ExportDFRowNamesAsSets, ExportDFRowNamesAsLists, dump_extended_dataframes


class ExperimentRunner:
    def __init__(self, experiment_name, output_file, results_folder, inp_path, log_name, output_folder, log_template,
                 dt_max_depth=15, dt_min_leaf=None, selection_method="fisher", selection_counts=None,
                 coverage_threshold=None, sequence_threshold=5, payload=False, payload_settings=None,
                 reencode=False, payload_type=None, payload_dwd_settings=None):

        if not payload_type:
            self.payload_type = "normal"
        else:
            self.payload_type = payload_type

        self.payload_dwd_settings = payload_dwd_settings

        self.payload = payload
        self.payload_settings = payload_settings

        self.counter = 0

        self.reencode = reencode

        self.experiment_name = experiment_name
        self.output_file = output_file
        self.results_folder = results_folder
        self.dt_max_depth = dt_max_depth
        self.dt_min_leaf = dt_min_leaf

        self.inp_path = inp_path
        self.log_name = log_name
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        self.log_template = log_template
        self.log_path = os.path.join(self.output_folder, log_template)
        self.log_path_seq = log_template

        self.train_output_file = "train_" + output_file
        self.test_output_file = "test_" + output_file

        if not coverage_threshold and (selection_method == "coverage" or selection_method == "rf_importance"):
            self.coverage_threshold = 20
        else:
            self.coverage_threshold = coverage_threshold

        self.sequence_threshold = sequence_threshold

        self.method = selection_method
        if not selection_counts and selection_method == "fisher":
            self.selection_counts = [100, 500, 1000]
        else:
            self.selection_counts = selection_counts
        self.encodings = ["tr", "tra", "mr", "mra"]



    def interpret_results(self, results, model_type, sequence_encoding=None):
        if self.method == "fisher":
            assert (False)

            selection_models = defaultdict(dict)
            results_per_selection = defaultdict(list)

            for split in results:
                for train_test in split["result"]:
                    # Train results
                    train_results = train_test["train"]
                    test_results = train_test["test"]
                    rules = rules["test"]
                    selection_count = train_test["selection_count"]

                    res = {
                        "train": train_results,
                        "test": test_results,
                        "rules": rules
                    }

                    results_per_selection[selection_count].append(res)

            for selection_count in self.selection_counts:
                # for encoding per selection
                if model_type == "sequence":
                    test_model_eval = model.ModelEvaluation.ModelEvaluation(
                        "TEST Model {} {} with selection {} with {} features".format(model_type, sequence_encoding,
                                                                                     self.method,
                                                                                     selection_count))
                    train_model_eval = model.ModelEvaluation.ModelEvaluation(
                        "TRAIN Model {} {} with selection {} with {} features".format(model_type, sequence_encoding,
                                                                                      self.method,
                                                                                      selection_count))
                else:
                    test_model_eval = model.ModelEvaluation.ModelEvaluation(
                        "TEST Model {} with selection {} with {} features".format(model_type,
                                                                                  self.method,
                                                                                  selection_count))
                    train_model_eval = model.ModelEvaluation.ModelEvaluation(
                        "TRAIN Model {} with selection {} with {} features".format(model_type,
                                                                                   self.method,
                                                                                   selection_count))
                for result in results_per_selection[selection_count]:
                    train_model_eval.add_results_dict(result["train"])
                    test_model_eval.add_results_dict(result["test"])

                selection_models[selection_count]["train"] = train_model_eval
                selection_models[selection_count]["test"] = test_model_eval

            return selection_models

        else:
            models = defaultdict(dict)
            rules = []

            if model_type == "sequence":
                test_model_eval = model.ModelEvaluation.ModelEvaluation(
                    "TEST Model {} {} with selection {}".format(model_type, sequence_encoding, self.method))
                train_model_eval = model.ModelEvaluation.ModelEvaluation(
                    "TRAIN Model {} {} with selection {}".format(model_type, sequence_encoding, self.method))
            else:
                test_model_eval = model.ModelEvaluation.ModelEvaluation(
                    "TEST Model {} with selection {}".format(model_type, self.method))
                train_model_eval = model.ModelEvaluation.ModelEvaluation(
                    "TRAIN Model {} with selection {} ".format(model_type, self.method))

            for r in results:
                train_model_eval.add_results_dict(r["result"]["train"])
                test_model_eval.add_results_dict(r["result"]["test"])
                rules.append(r["result"]["rules"])

            models["train"] = train_model_eval
            models["test"] = test_model_eval
            models["rules"] = rules

            return models

    # @staticmethod
    # def generate_cross_validation_logs(inp_path,log, log_name, output_folder, max_splits, split_perc): #split_perc = 0.2, max_splits = 5
    #     log_size = len(log)
    #     partition_size = int(split_perc * log_size)
    #     for log_nr in range(max_splits):
    #         new_log = XFactory.create_log(log.get_attributes().clone())
    #         for elem in log.get_extensions():
    #             new_log.get_extensions().add(elem)
    #
    #         #new_log.__classifiers = log.get_classifiers().copy()
    #         new_log.__globalTraceAttributes = log.get_global_trace_attributes().copy()
    #         new_log.__globalEventAttributes = log.get_global_event_attributes().copy()
    #
    #         # Add first part.
    #         for i in range(0, (log_nr * partition_size)):
    #             new_log.append(log[i])
    #
    #         # Add last part.
    #         for i in range((log_nr + 1) * partition_size, log_size):
    #             new_log.append(log[i])
    #
    #         # This is the test partitions, added to end
    #         for i in range(log_nr * partition_size, (log_nr + 1) * partition_size):
    #             if i >= log_size:
    #                 break  # edge case
    #             new_log.append(log[i])
    #
    #         count = 0
    #         for trace in new_log:
    #             if (trace.get_attributes()["Label"].get_value() == "1"):
    #                 count = count +1
    #         assert(count > 0)
    #         count = 0
    #         for trace in new_log:
    #             if (trace.get_attributes()["Label"].get_value() == "0"):
    #                 count = count +1
    #         assert(count > 0)
    #
    #         with open(os.path.join(output_folder,  log_name[:-4] + "_" + str(log_nr + 1) + ".xes"), "w") as file:
    #             XesXmlSerializer().serialize(new_log, file)
    #
    #         with open(os.path.join(inp_path, log_name[:-4] + "_" + str(log_nr + 1) + ".xes"), "w") as file:
    #             XesXmlSerializer().serialize(new_log, file)

    @staticmethod
    def create_folder_structure(directory, payload=False, payload_type=None):
            os.makedirs("./output/", exist_ok=True)
            os.makedirs(directory, exist_ok=True)

            # first level
            for i in range(1, 6):
                current_dir = os.path.join(directory, "split"+ str(i))
                #os.makedirs(current_dir) --> See documentation: this is completely useless, as it will be created with the first leaf creation

                # second level
                os.makedirs(os.path.join(current_dir, "baseline"), exist_ok=True)
                os.makedirs(os.path.join(current_dir, "declare"), exist_ok=True)
                os.makedirs(os.path.join(current_dir, "mr"), exist_ok=True)
                os.makedirs(os.path.join(current_dir, "mra"), exist_ok=True)
                os.makedirs(os.path.join(current_dir, "tr"), exist_ok=True)
                os.makedirs(os.path.join(current_dir, "tra"), exist_ok=True)

                if payload:
                    if payload_type == "normal" or "both":
                        os.makedirs(os.path.join(current_dir, "payload"), exist_ok=True)
                    if payload_type == "dwd" or "both":
                        os.makedirs(os.path.join(current_dir, "dwd"), exist_ok=True)

    @staticmethod
    def cross_validation_pipeline(inp_path, log_name, output_folder, max_splits, training_test_split):#max_splits = 5
        # 1. Load file
        log = read_XES_log(os.path.join(inp_path, log_name))

        # 2. Shuffling & Split into 5 parts for cross validation
        FairLogSplit.generateFairLogSplit(inp_path, log, log_name, output_folder, max_splits, training_test_split)

    @staticmethod
    def read_sequence_log(results_folder, encoding, split_nr, split_perc): #split_perc = 0.8
        split = "split" + str(split_nr)

        ## TODO: perform fair loading: receive the offsets from the input

        file_loc = os.path.join(results_folder, split, encoding)
        #train_path = file_loc + "/" + "globalLog.csv"
        #global_df = pd.read_csv(train_path, sep=";", index_col="Case_ID", na_filter=False)
        #size_df = len(global_df)
        #train_size = int(split_perc * size_df)
        #train_df = global_df.iloc[:train_size, ]
        #test_df = global_df.iloc[train_size:, ]
        train_df, test_df = read_arff_embedding_dump(file_loc, dict())

        return train_df, test_df

    @staticmethod
    def feature_selection(self, train_df, test_df, y_train, params, payload_train_df=None, payload_test_df=None, ):
        if payload_train_df is not None:
            train_df = pd.concat([train_df, payload_train_df], axis=1)
            test_df = pd.concat([test_df, payload_test_df], axis=1)

        ## Enforcing the fact that we want to perform analyses over numerical type
        ## Still, Joonas produces columns that contains strings, forsooth!
        train_df = train_df.select_dtypes(['number'])
        test_df  = test_df.select_dtypes(['number'])
        train_df = train_df.transpose().drop_duplicates().transpose()
        remaining_columns = train_df.columns

        test_df = test_df[remaining_columns]

        # remove no-variance, constants
        train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]
        test_df = test_df[train_df.columns]

        # Turn into np object
        X_train = train_df.values
        X_test = test_df.values
        feature_names = None

        selection_method = self.method

        if selection_method == "PCA":
            # PCA
            standardizer = StandardScaler().fit(X_train)

            # Standardize first
            X_train = standardizer.transform(X_train)
            X_test = standardizer.transform(X_test)

            # Apply PCA
            pca = PCA(n_components=3)

            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

            feature_names = None

        elif selection_method == "chi2":

            sel_count = self.selection_counts

            fitt = SelectKBest(chi2, k=min(X_train.shape[1], sel_count)).fit(X_train, y_train)

            X_train = fitt.transform(X_train)
            X_test = fitt.transform(X_test)

            feature_names = train_df.columns[fitt.get_support()]


        elif selection_method == "fisher":
            sel_count = params["selection_count"]
            scores = fisher_calculation(X_train, y_train)
            #scores = fisher_score.fisher_score(X_train, y_train)
            selected_ranks = fisher_score.feature_ranking(scores)[:sel_count]

            X_train = X_train[:, selected_ranks]
            X_test = X_test[:, selected_ranks]
            for i, rank in enumerate(selected_ranks[:10]):
                print(train_df.columns[rank], scores[i])

            feature_names = train_df.columns[selected_ranks]

        elif selection_method == "coverage":
            # Alternative version
            scores = fisher_calculation(X_train, y_train)
            selected_ranks = fisher_score.feature_ranking(scores)
    
            threshold = self.coverage_threshold
    
            # Start selecting from selected_ranks until every trace is covered N times
            trace_remaining = dict()
            for i, trace_name in enumerate(train_df.index.values):
                trace_remaining[i] = threshold
    
            chosen = 0
            #chosen_ranks = []
            # Go from higher to lower
            for rank in selected_ranks:
                #is_chosen = False
                if len(trace_remaining) == 0:
                    break
                chosen += 1
                # Get column
                marked_for_deletion = set()
                for k in trace_remaining.keys():
                    if train_df.iloc[k, rank] > 0:
                        #if not is_chosen:
                            # Only choose as a feature, if there is at least one trace covered by it.
                            #chosen_ranks.append(rank)
                            #is_chosen = True
    
                        trace_remaining[k] -= 1
                        if trace_remaining[k] <= 0:
                            marked_for_deletion.add(k)
    
                for k in marked_for_deletion:
                    del trace_remaining[k]
    
            X_train = X_train[:, selected_ranks[:chosen]]
            X_test = X_test[:, selected_ranks[:chosen]]
    
            feature_names = train_df.columns[selected_ranks[:chosen]]


        return X_train, X_test, feature_names

    def train_and_evaluate_select(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  payload_train_df=None, payload_test_df=None, params=None,
                                  exp_name=None, split_nr=None, dump_to_folder = None) -> (dict, dict):
        """
        Trains and evaluates model
        :param train_df:
        :param test_df:
        :param params:
        :return:
        """

        self.counter += 1
        # Pwoblem (sic!): sometimes, it happens that the Case_Id column is used for training.
        train_df = train_df.drop(['Case_Id'], axis=1, errors='ignore')
        test_df = test_df.drop(['Case_Id'], axis=1, errors='ignore')

        y_train = train_df.pop('Label').values # pop: removing the column from the table, while preserving the values
        y_test = test_df.pop('Label').values
        X_train, X_test, feature_names = self.feature_selection(train_df, test_df, y_train, params=params,
                                                                payload_train_df=payload_train_df,
                                                                payload_test_df=payload_test_df)
        if not (dump_to_folder is None):
            assert ((isinstance(dump_to_folder, tuple)) and ((len(dump_to_folder)==3)))
            X_traincpy = pd.DataFrame(data=X_train,  columns=feature_names)
            X_traincpy["Label"] = y_train
            X_testcpy = pd.DataFrame(data=X_test,  columns=feature_names)
            X_testcpy["Label"] = y_test
            dump_extended_dataframes(X_traincpy, X_testcpy, dump_to_folder[0], dump_to_folder[1], dump_to_folder[2])

        # Train classifier
        clf = DecisionTreeClassifier(max_depth=self.dt_max_depth)#, min_samples_leaf=self.dt_min_leaf)
        if (np.isnan(y_train).any()) or (np.isnan(X_train).any()) or (np.isnan(y_test).any()) or (np.isnan(X_test).any()):
            nanYtrain = ~np.isnan(y_train)
            X_train = X_train[nanYtrain]
            y_train = y_train[nanYtrain]
            X_train[np.isnan(X_train)] = -1

            nanYtest = ~np.isnan(y_test)
            X_test = X_test[nanYtest]
            y_test = y_test[nanYtest]
            X_test[np.isnan(X_test)] = -1
        clf.fit(X_train, y_train, check_input=False)

        # Evaluate model
        train_results, test_results = ModelUtils.evaluate_dt_model(clf, X_train, y_train, X_test, y_test)
        rules = ModelUtils.export_text2(clf, feature_names.to_list())

        return train_results, test_results, rules

    def train(self, train_df, test_df, payload_train_df=None, payload_test_df=None, split_nr=None, exp_name=None, dump_to_folder = None):
        if self.method == "fisher":
            assert (False) # Dead code
            selection_counts = self.selection_counts
            results = []
            # Trying all selection counts
            for selection_count in selection_counts:
                if payload_train_df is not None:
                    train_results, test_results, rules = self.train_and_evaluate_select(train_df.copy(), test_df.copy(),
                                                                             payload_train_df.copy(), payload_test_df.copy(), params={"selection_count": selection_count})
                else:
                    train_results, test_results, rules = self.train_and_evaluate_select(train_df.copy(), test_df.copy(),
                                                                                 params={"selection_count": selection_count})
                result = {
                    "train": train_results,
                    "test": test_results,
                    "selection_count": selection_count,
                    "rules": rules
                }

                results.append(result)

            return results
        else:
            if payload_train_df is not None:
                train_results, test_results, rules = self.train_and_evaluate_select(train_df.copy(),
                                                                             test_df.copy(),
                                                                             payload_train_df.copy(),
                                                                             payload_test_df.copy(),
                                                                             split_nr=split_nr, exp_name=exp_name, dump_to_folder = dump_to_folder)
            else:
                train_results, test_results, rules = self.train_and_evaluate_select(train_df.copy(),
                                                                            test_df.copy(),
                                                                            split_nr=split_nr, exp_name=exp_name)

            result = {
                "train": train_results,
                "test": test_results,
                    "rules": rules
            }

            return result

    def get_row_names_from_baseline_logs(self, max_range, dataset):
        for split_nr in range(1, max_range+1):
            d = dict()
            train_df, test_df = read_generic_embedding_dump(self.results_folder, split_nr, dataset, d)
            yield ExportDFRowNamesAsLists(train_df, test_df)

    def abstract_train(self, str_key, yamlfile, dataset, max_range):
        results = []
        elements = []
        for split_nr in range(1, max_range+1):
            d = dict()
            train_df, test_df = read_generic_embedding_dump(self.results_folder, split_nr, dataset, d)
            elements.append(d)
            tr_result = self.train(train_df, test_df, split_nr=split_nr, exp_name=dataset)

            result = {
                "result": tr_result,
                "split": split_nr
            }
            results.append(result)

        assert (not (str_key in yamlfile))
        yamlfile[str_key] = elements
        return results

    def baseline_train(self, str_key, yamlfile, max_range):
        return self.abstract_train(str_key, yamlfile, "baseline", max_range)

    def declare_train(self, str_key, yaml_file, max_range):
        return self.abstract_train(str_key, yaml_file, "declare", max_range)

    def sequence_train(self, encoding, yaml_file):
        """
        Trains a sequence model with given encoding
        :param encoding: sequence encoding
        :return:
        """
        results = []
        elements = []
        for split_nr in range(1, 6):
            # Read the log
            train_df, test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)
            elements.append(trace_encodings(self.results_folder, encoding, split_nr))
            tr_result = self.train(train_df, test_df, split_nr=split_nr, exp_name="sequence_{}".format(encoding))

            result = {
                "result": tr_result,
                "split": split_nr,
                "encoding": encoding
            }

            results.append(result)

        yaml_file[encoding] = elements
        return results

    def hybrid_train(self, str_key, yaml_file):
        """
        Hybrid model training
        :return:
        """
        encodings = ["mr", "mra", "tr", "tra"]

        results = []
        elements = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "declare", dict())
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df] + seq_test_list, axis=1)

            multidump_compact(self.results_folder, elements, str_key, merged_train_df, merged_test_df, split_nr)
            tr_result = self.train(merged_train_df, merged_test_df, split_nr=split_nr, exp_name="hybrid")
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[str_key] = elements
        return results

    def payload_train(self, key, yaml_file):
        """
        Trains and tests models just on payload data.
        :return:
        """
        results = []
        elements = []
        for split_nr in range(1, 6):
            baseline_train_df, baseline_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "baseline", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "payload", dict())

            payload_train_df["Label"] = baseline_train_df["Label"]
            payload_test_df["Label"] = baseline_test_df["Label"]

            self.multidump_compact(elements, "payload_for_training", payload_test_df, payload_train_df, split_nr)
            tr_result = self.train(payload_train_df, payload_test_df, split_nr=split_nr, exp_name="payload")

            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file["payload_for_training"] = elements
        return results
    #
    # def multidump_compact(self, elements, forMultiDump, payload_test_df, payload_train_df, split_nr):
    #     tr_f, t_f = dump_extended_dataframes(payload_train_df, payload_test_df, self.results_folder, split_nr,
    #                                          forMultiDump)
    #     d = dict()
    #     d["train"] = os.path.abspath(tr_f)
    #     d["test"] = os.path.abspath(t_f)
    #     elements.append(d)

    def baseline_train_with_data(self, key, yaml_file):
        """

        :return:
        """

        results = []
        elements = []
        for split_nr in range(1, 6):
            train_df, test_df = read_generic_embedding_dump(self.results_folder, split_nr, "baseline", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "payload", dict())
            #merged_train_df = pd.concat([train_df, payload_train_df], axis=1)
            #merged_test_df = pd.concat([test_df, payload_test_df], axis=1)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="baseline_payload", dump_to_folder=(self.results_folder, split_nr, key))

            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results



    def baseline_train_with_dwd(self, key, yaml_file):
        """

        :return:
        """

        results = []
        elements = []
        for split_nr in range(1, 6):
            train_df, test_df = read_generic_embedding_dump(self.results_folder, split_nr, "baseline", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "dwd", dict())

            # merged_train_df = pd.concat([train_df, payload_train_df], axis=1)
            # merged_test_df = pd.concat([test_df, payload_test_df], axis=1)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="baseline_dwd", dump_to_folder=(self.results_folder, split_nr, key))
            extract_file_name_for_dump(self.results_folder, elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results


    def declare_train_with_data(self, key, yaml_file):
        """
        Train and evaluate declare models
        :return:
        """

        results = []
        elements = []
        # Separately for every split. Reduce total number of file parsing.
        for split_nr in range(1, 6):
            train_df, test_df = read_generic_embedding_dump(self.results_folder, split_nr, "declare", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "payload", dict())


            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="declare_data", dump_to_folder=(self.results_folder, split_nr, key))

            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results


    def declare_train_with_dwd(self, key, yaml_file):
        """
        Train and evaluate declare models
        :return:
        """

        results = []
        elements = []
        # Separately for every split. Reduce total number of file parsing.
        for split_nr in range(1, 6):
            train_df, test_df = read_generic_embedding_dump(self.results_folder, split_nr, "declare", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "dwd", dict())

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="declare_dwd", dump_to_folder=(self.results_folder, split_nr, key))

            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results


    def declare_train_with_dwd_data(self, key, yaml_file):
        """
        Train and evaluate declare models
        :return:
        """
        results = []
        elements = []
        # Separately for every split. Reduce total number of file parsing.
        for split_nr in range(1, 6):
            train_df, test_df = read_generic_embedding_dump(self.results_folder, split_nr, "declare", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "payload", dict())
            payload_train_df_2, payload_test_df_2 = read_generic_embedding_dump(self.results_folder, split_nr, "dwd", dict())

            merged_train_df = pd.concat([train_df, payload_train_df_2], axis=1)
            merged_test_df = pd.concat([test_df, payload_test_df_2], axis=1)

            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="declare_dwd_data", dump_to_folder=(self.results_folder, split_nr, key))

            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results


    def sequence_train_with_data(self, encoding, yaml_file):
        """
        Trains a sequence model with given encoding
        :param encoding: sequence encoding
        :return:
        """

        results = []
        elements = []
        key = encoding+"_data"
        for split_nr in range(1, 6):
            # Read the log
            train_df, test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "payload", dict())

            #merged_train_df = pd.concat([train_df, payload_train_df], axis=1)
            #merged_test_df = pd.concat([test_df, payload_test_df], axis=1)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="sequence_data_{}".format(encoding), dump_to_folder=(self.results_folder, split_nr, key))

            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr,
                "encoding": encoding
            }

            results.append(result)

        yaml_file[key] = elements
        return results

    def hybrid_with_data(self, key, yaml_file):
        """
        Hybrid model training with additional data
        :return:
        """

        encodings = ["mr", "mra", "tr", "tra"]
        elements = []
        results = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "declare", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "payload", dict())
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df] + seq_test_list, axis=1)
            # , payload_train_df, payload_test_df
            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df, split_nr=split_nr, exp_name="hybrid_data", dump_to_folder=(self.results_folder, split_nr, key))
            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results


    def hybrid_with_dwd(self, key, yaml_file):
        """
        Hybrid model training with additional data
        :return:
        """

        encodings = ["mr", "mra", "tr", "tra"]

        results = []
        elements = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "declare", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "dwd", dict())
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df] + seq_test_list, axis=1)
            # , payload_train_df, payload_test_df
            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df, split_nr=split_nr, exp_name="hybrid_dwd", dump_to_folder=(self.results_folder, split_nr, key))
            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results

    def hybrid_with_dwd_and_payload(self, key, yaml_file):
        """
        Hybrid model training with additional data
        :return:
        """

        encodings = ["mr", "mra", "tr", "tra"]

        results = []
        elements = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "declare", dict())
            payload_train_df, payload_test_df = read_generic_embedding_dump(self.results_folder, split_nr, "dwd", dict())
            payload_train_df_2, payload_test_df_2 = read_generic_embedding_dump(self.results_folder, split_nr, "payload", dict())
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df, payload_train_df_2] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df, payload_test_df_2] + seq_test_list, axis=1)
            # , payload_train_df, payload_test_df
            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df, split_nr=split_nr, exp_name="hybrid_dwd_payload", dump_to_folder=(self.results_folder, split_nr, key))
            self.extract_file_name_for_dump(elements, key, split_nr)
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        yaml_file[key] = elements
        return results


    def train_and_eval_benchmark(self, max_splits):
        all_results = {}
        yaml_file = {}
        TrL, TeL = self.get_row_names_from_baseline_logs( max_splits, "baseline")

        if True:#not self.payload:
            print("Started working on baseline.")
            baseline_results = self.baseline_train("bs", yaml_file, max_splits)
            all_results["bs"] = self.interpret_results(baseline_results, "baseline")

            print("Started working on declare.")
            declare_results = self.declare_train("dc", yaml_file, max_splits)
            all_results["dc"] = self.interpret_results(declare_results, "declare")

            print("Started working on sequenceMR.")
            sequence_results = self.sequence_train("mr", yaml_file)
            all_results["mr"] = self.interpret_results(sequence_results, "sequence", "mr")

            print("Started working on sequenceTR.")
            sequence_results = self.sequence_train("tr", yaml_file)
            all_results["tr"] = self.interpret_results(sequence_results, "sequence", "tr")

            print("Started working on sequenceTRA.")
            sequence_results = self.sequence_train("tra", yaml_file)
            all_results["tra"] = self.interpret_results(sequence_results, "sequence", "tra")

            print("Started working on sequenceMRA.")
            sequence_results = self.sequence_train("mra", yaml_file)
            all_results["mra"] = self.interpret_results(sequence_results, "sequence", "mra")

            print("Started working on hybrid.")
            hybrid_results = self.hybrid_train("hybrid", yaml_file)
            all_results["hybrid"] = self.interpret_results(hybrid_results, "hybrid")

        if self.payload:
                print("Started working on payload train.")
                payload_results = self.payload_train("bs", yaml_file)
                all_results["payload"] = self.interpret_results(payload_results, "payload")

                print("Started working on baseline with data.")
                baseline_results = self.baseline_train_with_data("bs_data", yaml_file)
                all_results["bs_data"] = self.interpret_results(baseline_results, "baseline")

                print("Started working on declare with data.")
                declare_results = self.declare_train_with_data("dc_data", yaml_file)
                all_results["dc_data"] = self.interpret_results(declare_results, "declare")

                print("Started working on sequenceMR with data.")
                sequence_results = self.sequence_train_with_data("mr", yaml_file)
                all_results["mr_data"] = self.interpret_results(sequence_results, "sequence", "mr")

                print("Started working on sequenceTR with data.")
                sequence_results = self.sequence_train_with_data("tr", yaml_file)
                all_results["tr_data"] = self.interpret_results(sequence_results, "sequence", "tr")

                print("Started working on sequenceTRA with data.")
                sequence_results = self.sequence_train_with_data("tra", yaml_file)
                all_results["tra_data"] = self.interpret_results(sequence_results, "sequence", "tra")

                print("Started working on sequenceMRA with data.")
                sequence_results = self.sequence_train_with_data("mra", yaml_file)
                all_results["mra_data"] = self.interpret_results(sequence_results, "sequence", "mra")

                print("Started working on hybrid with data.")
                payload_results = self.hybrid_with_data("hybrid_data", yaml_file)
                all_results["hybrid_data"] = self.interpret_results(payload_results, "hybrid_data")

                if self.payload_type == "both":
                    print("Started working on declare with dwd.")
                    declare_results = self.declare_train_with_dwd("dc_dwd", yaml_file)
                    all_results["dc_dwd"] = self.interpret_results(declare_results, "declare_dwd")

                    print("Started working on declare with dwd and payload.")
                    declare_results = self.declare_train_with_dwd_data("dc_dwd_payload", yaml_file)
                    all_results["dc_dwd_payload"] = self.interpret_results(declare_results, "declare_payload_dwd")

                    print("Started working on hybrid with dwd.")
                    payload_results = self.hybrid_with_dwd("hybrid_dwd", yaml_file)
                    all_results["hybrid_dwd"] = self.interpret_results(payload_results, "hybrid_dwd")

                    print("Started working on hybrid with dwd and usual payload.")
                    payload_results = self.hybrid_with_dwd_and_payload("hybrid_dwd_payload", yaml_file)
                    all_results["hybrid_dwd_payload"] = self.interpret_results(payload_results, "hybrid_data_dwd")

        weka_yaml_file = os.path.join(self.results_folder, "for_weka_experiments.yaml")
        if not (os.path.exists(weka_yaml_file)):
            print("Writing the yaml file:")
            with open(weka_yaml_file, 'w') as file:
                yaml.dump(yaml_file, file)

        from .GoodPrintResults import printToFile
        line = None
        if (not os.path.exists(os.path.join(self.results_folder, "benchmarks.csv"))):
            line = "dataset,learner,outcome_type,strategy,conftype,confvalue,metrictype,metricvalue\n"
        with open(os.path.join(self.results_folder, "benchmarks.csv"), "a") as csvFile:
            with open(os.path.join(self.results_folder, "rules.txt"), "a") as rulesFile:
                if not (line is None):
                    csvFile.write(line)
                printToFile(all_results, self.experiment_name, "Decision Tree", "max_depth", self.dt_max_depth, csvFile, rulesFile)
                rulesFile.close()
            csvFile.close()

    def prepare_cross_validation(self, max_splits, training_test_split): #max_splits = 5
        self.cross_validation_pipeline(self.inp_path, self.log_name, self.output_folder, max_splits, training_test_split)

    def serialize_complete_dataset(self, isCompleteEmbedding):
        logFilePath = os.path.join(self.inp_path, self.log_name)
        log = read_XES_log(logFilePath)
        d = logFilePath
        if isCompleteEmbedding:
            d = os.path.join("./complete_embeddings/", logFilePath)
        os.makedirs(d, exist_ok=True)
        from DevianceMiningPipeline.baseline_runner import baseline
        from DevianceMiningPipeline.declaredevmining import declare_deviance_mining
        yamlFile = {}
        print("\x1b[6;30;42m Baseline generation:\x1b[0m")
        yamlFile["baseline"] = baseline(d, logFilePath, 1.0)
        print("\x1b[6;30;42m Declare generation:\x1b[0m")
        yamlFile["declare"]  = declare_deviance_mining(d, log, split_size=1.0, reencode=self.reencode)        #run_deviance_new
        print("\x1b[6;30;42m Generate Sequences generation:\x1b[0m")
        yamlFile.update(generateSequences(self.inp_path, logFilePath, d))
        if not (self.payload_settings is None):
            print("\x1b[6;30;42m Payload generation:\x1b[0m")
            yamlFile["payload"] = payload_extractor2(d, logFilePath, self.payload_settings)
        if not (self.payload_dwd_settings is None):
            print("\x1b[6;30;42m DWD Sequences generation:\x1b[0m")
            yamlFile["dwd"] = data_declare_main(d, logFilePath, self.payload_dwd_settings["ignored"], split=1.0)
        with open(os.path.abspath(os.path.join(d, "../"+self.log_name+".yaml")), 'w') as file:
            yaml.dump(yamlFile, file)


    def prepare_data(self, max_splits, training_test_split, doForce = False):
        self.create_folder_structure(self.results_folder, payload=self.payload, payload_type=self.payload_type)
        ignored = []
        if self.payload_dwd_settings is not None:
            ignored = self.payload_dwd_settings["ignored"]

        print("New code by Giacomo Bergami, for evenly splitting and storing the database")
        for i in range(max_splits):
            print("Current run: " +str(i))
            baseline_path = FileNameUtils.baseline_path(i, self.results_folder)
            declare__path = FileNameUtils.declare_path(i, self.results_folder)
            payload__path = FileNameUtils.payload_path(i, self.results_folder)
            declared_path = FileNameUtils.declare_data_aware_path(i, self.results_folder)

            print("\t - reading the log")
            log = read_XES_log(FileNameUtils.getXesName(self.log_path, i))
            TrainingId, TestingId = FairLogSplit.abstractFairSplit(log,
                                                                   TraceUtils.isTraceLabelPositive,
                                                                   TraceUtils.getTraceId,
                                                                   training_test_split)

            print("\t * obtaining the canonical XES representation")
            logTraining, logTesting = \
                LogUtils.xes_to_tracelist_split(log, TrainingId, TestingId)

            print("\t * obtaining the splitted propositional representation")
            propositionalTraining, propositionalTesting = \
                LogUtils.xes_to_propositional_split(log, TrainingId, TestingId)

            print("\t * obtaining the data propositional representation")
            dataPropositionalTraining, dataPropositionalTesting = \
                LogUtils.xes_to_data_propositional_split(log, TrainingId, TestingId, doForce)

            print("\t - writing baseline split")
            STr, STt = baseline_runner.baseline_embedding(baseline_path, propositionalTraining, propositionalTesting)
            assert (STr == TrainingId)
            assert (STt == TestingId)

            print("\t - writing declare split")
            STr, STt = declaredevmining.declare_embedding(declare__path, propositionalTraining, propositionalTesting, reencode=self.reencode)
            assert (STr == TrainingId)
            assert (STt == TestingId)

            if self.payload:
                if self.payload_type == "normal" or self.payload_type == "both":
                    print("\t - writing payload embedding")
                    STr, STt = payload_embedding(payload__path, self.payload_settings, logTraining, logTesting)
                    assert (STr == TrainingId)
                    assert (STt == TestingId)
                if self.payload_type == "dwd" or self.payload_type == "both":
                    print("\t - writing declare with data embedding")
                    STr, STt = declare_data_aware_embedding(ignored, declared_path, dataPropositionalTraining, dataPropositionalTesting)
                    assert (STr == TrainingId)
                    assert (STt == TestingId)

        print("~~ Run sequence miner for all the params")
        run_sequences(self.inp_path, self.log_path_seq, self.results_folder, self.err_logger, max_splits, sequence_threshold=self.sequence_threshold)


    def clean_data(self):
        shutil.rmtree(self.results_folder)


