package it.giacomobergami.unibz;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.Rule;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.*;

public class LoadDatasetsForRipper {
    public static void main(String args[]) throws Exception {
        String lerner = "RipperK";
        String conftype = "iteration";

        if (args.length <6) {
            System.err.println("Error: you should provide at least 5 args");
            System.err.println(" - Input: Training set csv file");
            System.err.println(" - Input: Testing set csv file");
            System.err.println(" - Output: Files where the rules will be appended");
            System.err.println(" - Output: Files where the benchmarks will be printed");
            System.err.println(" - Input: Dataset name associated to the experiment");
            System.err.println(" - Element: Embedding configuration associated to the experiment");
            System.exit(1);
        } else  {
            String train = args[0];
            assert (new File(train).exists());
            String test = args[1];
            assert (new File(test).exists());
            String rule_file = args[2];
            assert (new File(rule_file).getParentFile().exists());
            String csv_file = args[3];
            assert (new File(rule_file).getParentFile().exists());
            String dataset = args[4];
            String elements = args[5];

            System.out.println();

            NumericToNominal nominal = new NumericToNominal();
            nominal.setOptions(new String[]{"-R", "last"});

            try(FileWriter csv_fw = new FileWriter(csv_file, true);
                BufferedWriter csv_bw = new BufferedWriter(csv_fw);
                PrintWriter csvFile = new PrintWriter(csv_bw);
            FileWriter rule_fw = new FileWriter(rule_file, true);
            BufferedWriter rule_bw = new BufferedWriter(rule_fw);
            PrintWriter ruleFile = new PrintWriter(rule_bw))
            {
                for (int i = 0; i<9; i++) {
                    int optimization_i = i * 2;
                    JRip ripperk = new JRip();
                    Instances trainingSet = extracted(train, "Label", nominal);
                    Instances testingSet = extracted(test,  "Label", nominal);

                    Evaluation evaluation = new Evaluation(trainingSet);
                    ripperk.setOptimizations(optimization_i);
                    ripperk.buildClassifier(trainingSet);

                    evaluation.evaluateModel(ripperk, testingSet);
                    double auc = evaluation.areaUnderROC(1);
                    double f1 = evaluation.fMeasure(1);
                    double precision = evaluation.precision(1);
                    double recall = evaluation.recall(1);
                    double acc = evaluation.pctCorrect();

                    for (Rule r : ripperk.getRuleset()) {
                        ruleFile.println(r);
                    }

                    csvFile.println(dataset + "," + lerner + "," + elements + "," + conftype + "," + optimization_i + ",acc,"+(acc));
                    csvFile.println(dataset + "," + lerner + "," + elements + "," + conftype + "," + optimization_i + ",auc," + (auc));
                    csvFile.println(dataset + "," + lerner + "," + elements + "," + conftype + "," + optimization_i + ",f1," + (f1));
                    csvFile.println(dataset + "," + lerner + "," + elements + "," + conftype + "," + optimization_i + ",precision," + (precision));
                    csvFile.println(dataset + "," + lerner + "," + elements + "," + conftype + "," + optimization_i + ",recall," + (recall));

                }
            } catch (IOException e) {
                //exception handling left as an exercise for the reader
            }
        }
    }

    /**
     * Loading a dataset so to be used by JRipper
     *
     * @param filename      File to be loaded
     * @param label         Label associated to the class
     * @param nominal       Converter from numbers to string classes
     * @return              Converted and loaded dataset
     * @throws Exception
     */
    private static Instances extracted(String filename, String label, NumericToNominal nominal) throws Exception {
        CSVLoader training = new CSVLoader();
        training.setSource(new File(filename));
        Instances data = training.getDataSet();
        nominal.setInputFormat(data);
        data = Filter.useFilter(data, nominal);
        data.setClass(data.attribute(label));
        return data;
    }

}
