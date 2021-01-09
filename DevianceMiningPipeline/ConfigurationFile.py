import jsonpickle

def ifneg_right(lhs, rhs):
    if (lhs <= 0):
        return (rhs)
    else:
        return (lhs)


class ConfigurationFile(object):
    def __init__(self, experiment_name, log_name, output_folder, dt_max_depth, dt_min_leaf, sequence_threshold, payload_type, ignored = None, payload_settings = None):
        self.experiment_name = experiment_name
        self.log_name = log_name
        self.output_folder = output_folder
        self.results_folder = experiment_name + "_results"
        self.results_file = self.results_folder + ".txt"
        self.log_path_seq = log_name[:ifneg_right(log_name.rfind('.'),len(log_name))] + "_{}" + log_name[ifneg_right(log_name.rfind('.'),len(log_name)):]
        self.dt_max_depth = dt_max_depth
        self.dt_min_leaf = dt_min_leaf
        self.auto_ignored = ignored
        self.payload_settings = payload_settings
        self.sequence_threshold = sequence_threshold
        self.payload_type = payload_type

    def dump(self, file):
        f = open(file, 'w')
        f.write(jsonpickle.encode(self))
        f.close()

    def run(self, INP_PATH, coverage_thresholds):
        import os
        for nr, i in enumerate(coverage_thresholds):
            from DevianceMiningPipeline.ExperimentRunner import ExperimentRunner
            ex = ExperimentRunner(experiment_name=self.experiment_name,
                                  output_file=self.results_file,
                                  results_folder=os.path.join(INP_PATH, self.results_folder),
                                  inp_path=INP_PATH,
                                  log_name=self.log_name,
                                  output_folder=self.output_folder,
                                  log_template=self.log_path_seq,
                                  dt_max_depth=self.payload_settings,
                                  dt_min_leaf=self.dt_min_leaf,
                                  selection_method="coverage",
                                  coverage_threshold=i,
                                  sequence_threshold=self.sequence_threshold,
                                  payload=True,
                                  payload_type=self.payload_type)

            if not self.auto_ignored is None:
                ex.payload_dwd_settings = {"ignored": self.auto_ignored }
            if not self.payload_settings is None:
                ex.payload_settings = self.payload_settings

            with open("train_" + self.results_file, "a+") as f:
                f.write("\n")
            with open("test_" + self.results_file, "a+") as f:
                f.write("\n")
            if nr == 0:
                ex.prepare_cross_validation()
                ex.prepare_data()
            ex.train_and_eval_benchmark()