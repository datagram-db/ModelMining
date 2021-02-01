import jsonpickle

def ifneg_right(lhs, rhs):
    if (lhs <= 0):
        return (rhs)
    else:
        return (lhs)


class ConfigurationFile(object):
    def __init__(self):
        self.auto_ignored = None
        self.payload_settings = None
        self.payload_type = None

    @classmethod
    def explicitInitialization(cls, experiment_name, log_name, output_folder, dt_max_depth, dt_min_leaf, sequence_threshold, payload_type, ignored = None, payload_settings = None):
        cf = cls()
        cf.setExperimentName(experiment_name)
        cf.setLogName(log_name)
        cf.setOutputFolder(output_folder)
        cf.setMaxDepth(dt_max_depth)
        cf.setMinLeaf(dt_min_leaf)
        cf.setSequenceThreshold(sequence_threshold)
        cf.setPayloadType(payload_type)

    # def __init__(self, experiment_name, log_name, output_folder, dt_max_depth, dt_min_leaf, sequence_threshold, payload_type, ignored = None, payload_settings = None):
    #     self.experiment_name = experiment_name
    #     self.log_name = log_name
    #     self.output_folder = output_folder
    #     self.results_folder = experiment_name + "_results"
    #     self.results_file = self.results_folder + ".txt"
    #     self.log_path_seq = log_name[:ifneg_right(log_name.rfind('.'),len(log_name))] + "_{}" + log_name[ifneg_right(log_name.rfind('.'),len(log_name)):]
    #     self.dt_max_depth = dt_max_depth
    #     self.dt_min_leaf = dt_min_leaf
    #     self.auto_ignored = ignored
    #     self.payload_settings = payload_settings
    #     self.sequence_threshold = sequence_threshold
    #     self.payload_type = payload_type

    def setExperimentName(self, experiment_name):
        self.experiment_name = experiment_name
        self.results_folder = experiment_name + "_results"
        self.results_file = self.results_folder + ".txt"

    def setLogName(self, log_name):
        self.log_name = log_name
        self.log_path_seq = log_name[:ifneg_right(log_name.rfind('.'),len(log_name))] + "_{}" + log_name[ifneg_right(log_name.rfind('.'),len(log_name)):]

    def setOutputFolder(self, output_folder):
        self.output_folder = output_folder

    def setMaxDepth(self, dt_max_depth):
        self.dt_max_depth = dt_max_depth

    def setMinLeaf(self, dt_min_leaf):
        self.dt_min_leaf = dt_min_leaf

    def setSequenceThreshold(self, sequence_threshold):
        self.sequence_threshold = sequence_threshold

    def setPayloadType(self, payload_type):
        self.payload_type = payload_type

    def setAutoIgnore(self, ignored):
        if not (ignored is None):
            self.auto_ignored = ignored

    def setPayloadSettings(self,payload_settings):
        if not (payload_settings is None):
            self.payload_settings = payload_settings

    def dump(self, file):
        f = open(file, 'w')
        f.write(jsonpickle.encode(self))
        f.close()

    def complete_embedding_generation(self, INP_PATH, DATA_EXP):
        from DevianceMiningPipeline.ExperimentRunner import ExperimentRunner
        from pathlib import Path
        import os
        ex = ExperimentRunner(experiment_name=self.experiment_name,
                              output_file=self.results_file,
                              results_folder=os.path.join(DATA_EXP, self.results_folder),
                              inp_path=INP_PATH,
                              log_name=self.log_name,
                              output_folder=os.path.join(DATA_EXP, self.output_folder),
                              log_template=self.log_path_seq,
                              dt_max_depth=self.dt_max_depth,
                              dt_min_leaf=self.dt_min_leaf,
                              selection_method="coverage",
                              coverage_threshold=5,
                              sequence_threshold=self.sequence_threshold,
                              payload=not (self.payload_type is None),
                              payload_type=self.payload_type)
        if not self.auto_ignored is None:
            ex.payload_dwd_settings = {"ignored": self.auto_ignored}
        if not self.payload_settings is None:
            ex.payload_settings = self.payload_settings
        ex.serialize_complete_dataset(True)

    def run(self, INP_PATH, DATA_EXP, coverage_thresholds, doNr0 = True, max_splits = 5):
        from pathlib import Path
        import os
        Path(os.path.join(DATA_EXP, self.results_folder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(DATA_EXP, self.output_folder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(DATA_EXP, "error_log")).mkdir(parents=True, exist_ok=True)
        for nr, i in enumerate(coverage_thresholds):
            from DevianceMiningPipeline.ExperimentRunner import ExperimentRunner
            # ex = ExperimentRunner(experiment_name=self.experiment_name,
            #                       output_file=self.results_file,
            #                       results_folder=os.path.join(DATA_EXP, self.results_folder),
            #                       inp_path=INP_PATH,
            #                       log_name=self.log_name,
            #                       output_folder=os.path.join(DATA_EXP, self.output_folder),
            #                       log_template=self.log_path_seq,
            #                       dt_max_depth=self.dt_max_depth,
            #                       dt_min_leaf=self.dt_min_leaf,
            #                       selection_method="coverage",
            #                       coverage_threshold=i,
            #                       sequence_threshold=self.sequence_threshold,
            #                       payload=True,
            #                       payload_type=self.payload_type.value[0])
            ex = ExperimentRunner(experiment_name=self.experiment_name,
                                  output_file=self.results_file,
                                  results_folder=os.path.join(DATA_EXP, self.results_folder),
                                  inp_path=INP_PATH,
                                  log_name=self.log_name,
                                  output_folder=os.path.join(DATA_EXP, self.output_folder),
                                  log_template=self.log_path_seq,
                                  dt_max_depth=self.dt_max_depth,
                                  dt_min_leaf=self.dt_min_leaf,
                                  selection_method="coverage",
                                  coverage_threshold=5,
                                  sequence_threshold=self.sequence_threshold,
                                  payload=not (self.payload_type is None),
                                  payload_type=self.payload_type)
            ex.err_logger = os.path.join(DATA_EXP, "error_log")
            if not self.auto_ignored is None:
                ex.payload_dwd_settings = {"ignored": self.auto_ignored }
            if not self.payload_settings is None:
                ex.payload_settings = self.payload_settings

            with open("train_" + self.results_file, "a+") as f:
                f.write("\n")
            with open("test_" + self.results_file, "a+") as f:
                f.write("\n")
            if (nr == 0) and doNr0:
                ex.prepare_cross_validation(max_splits)  # Splits the log into max_splits different files. FIXME: the split is not accurately selected
                ex.prepare_data()
            ex.train_and_eval_benchmark()