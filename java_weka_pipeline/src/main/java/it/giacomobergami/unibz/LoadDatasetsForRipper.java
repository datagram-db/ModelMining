package it.giacomobergami.unibz;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.Rule;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.*;
import java.util.ArrayList;

public class LoadDatasetsForRipper {
    public static void test(String args[]) throws Exception {
        String lerner = "RipperK";
        String conftype = "iteration";

        if (args.length <7) {
            System.err.println("Error: you should provide at least 6 args");
            System.err.println(" - Input: Training set csv file");
            System.err.println(" - Input: Testing set csv file");
            System.err.println(" - Output: File where the rules will be appended");
            System.err.println(" - Output: File where the benchmarks will be printed");
            System.err.println(" - Input: Dataset name associated to the experiment");
            System.err.println(" - Input: Embedding configuration associated to the experiment");
            System.err.println(" - Input: Separator for csv file (comma ',', semicolon ';'...)");
            System.exit(1);
        } else  {
            String train_csv = args[0];
            System.out.println(">Training set for csv file: "+train_csv);
            assert (new File(train_csv).exists());
            String test_csv = args[1];
            System.out.println(">Testing set for csv file: "+test_csv);
            assert (test_csv.isEmpty() || (new File(test_csv).exists()));
            String rule_file = args[2];
            System.out.println("<File where the rules will be appended: "+rule_file);
            assert (new File(rule_file).getParentFile().exists());
            String csv_file = args[3];
            System.out.println("<File where the benchmarks will be printed: "+csv_file);
            assert (new File(csv_file).getParentFile().exists());
            String dataset = args[4];
            System.out.println(">Dataset name associated to the experiment: "+dataset);
            String elements = args[5];
            System.out.println(">Embedding configuration associated to the experiment: "+elements);
            String sep = args[6];
            System.out.println(">Cell separator: "+sep);
            System.out.println("===================================================\n\n");



            try(FileWriter csv_fw = new FileWriter(csv_file, true);
                BufferedWriter csv_bw = new BufferedWriter(csv_fw);
                PrintWriter csvFile = new PrintWriter(csv_bw);
            FileWriter rule_fw = new FileWriter(rule_file, true);
            BufferedWriter rule_bw = new BufferedWriter(rule_fw);
            PrintWriter ruleFile = new PrintWriter(rule_bw))
            {
                int max_iteration = 10;
                dumpFile(dataset, train_csv, test_csv, elements, sep, csvFile, ruleFile, max_iteration);
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
    }

    public static boolean dumpFile(String folder_name, String train, String test, String exp_setting_embedding, String sep, PrintWriter csvFile, PrintWriter ruleFile, int max_iteration) throws Exception {
        String lerner = "RipperK";
        String conftype = "iteration";
        NumericToNominal nominal = new NumericToNominal();
        nominal.setOptions(new String[]{"-R", "last"});
        boolean hasError = false;
        for (int i = 0; i< max_iteration; i++) {
            int optimization_i = i * 2;
            JRip ripperk = new JRip();
            Instances trainingSet = extracted(train, "Label", nominal, sep);
            Instances testingSet;
            if (test.isEmpty()) {
                int trainSize = (int) Math.round(trainingSet.size()*0.8);
                int testSize = trainingSet.numInstances() - trainSize;
                Instances tmp = new Instances(trainingSet, 0, trainSize);
                testingSet = new Instances(trainingSet, trainSize, testSize);
                trainingSet = tmp;
            } else {
                testingSet = extracted(test,  "Label", nominal, sep);
            }
            Evaluation evaluation = new Evaluation(trainingSet);
            ripperk.setOptimizations(optimization_i);
            try {
                ripperk.buildClassifier(trainingSet);
            } catch (Exception e) {
                System.err.println("Error while building the classifier!");
                hasError = true;
            }
            double auc = 0.0;
            double f1 = 0.0;
            double precision = 0.0;
            double recall = 0.0;
            double acc = 0.0;
            if (ripperk.getRuleset() == null || ripperk.getRuleset().isEmpty()) {
                System.err.println("ERROR: the classifier produced no rules! Skipping the model testing");
            } else {
                int k = 0;
                Attribute catt = trainingSet.classAttribute();
                ArrayList<Rule> ruleSet = ripperk.getRuleset();
                for (Rule rs : ruleSet) {
                    //double[] simStats = rs.getSimpleStats(k);
                    ArrayList<JRip.Antd> ants = ((JRip.RipperRule) ruleSet.get(k)).getAntds();
                    if ((ants != null) && (!ants.isEmpty())) {
                        ruleFile.println(folder_name + "::" + lerner + "::test::" + exp_setting_embedding + "::" + conftype + "::" + optimization_i+"§\t"+(((JRip.RipperRule) ruleSet.get(k)).toString(catt) + " ("
                                + 0 + "/" + 0 + ")"));
                    }
                    k++;
                }
                try {
                    evaluation.evaluateModel(ripperk, testingSet);
                } catch (Exception e) {
                    System.err.println("Error while evaluating the classifier!");
                    hasError = true;
                }
                //Normalize value to 1
                try {
                    auc = evaluation.areaUnderROC(1);
                } catch (Exception e) {
                    System.err.println("Error while computing auc: setting it to zero");
                    auc = 0.0;
                    hasError = true;
                }
                try {
                    f1 = evaluation.fMeasure(1);
                }catch (Exception e) {
                    System.err.println("Error while computing f1: setting it to zero");
                    f1 = 0.0;
                    hasError = true;
                }
                try {
                    precision = evaluation.precision(1);
                }catch (Exception e) {
                    System.err.println("Error while computing precision: setting it to zero");
                    precision = 0.0;
                    hasError = true;
                }
                try {
                    recall = evaluation.recall(1);
                }catch (Exception e) {
                    System.err.println("Error while computing recall: setting it to zero");
                    recall = 0.0;
                    hasError = true;
                }
                try {
                    acc = evaluation.pctCorrect()/100.0;
                }catch (Exception e) {
                    System.err.println("Error while computing acc: setting it to zero");
                    acc = 0.0;
                    hasError = true;
                }
            }




            csvFile.println(folder_name + "," + lerner + ",test," + exp_setting_embedding + "," + conftype + "," + optimization_i + ",acc,"+(acc));
            csvFile.println(folder_name + "," + lerner + ",test," + exp_setting_embedding + "," + conftype + "," + optimization_i + ",auc," + (auc));
            csvFile.println(folder_name + "," + lerner + ",test," + exp_setting_embedding + "," + conftype + "," + optimization_i + ",f1," + (f1));
            csvFile.println(folder_name + "," + lerner + ",test," + exp_setting_embedding + "," + conftype + "," + optimization_i + ",precision," + (precision));
            csvFile.println(folder_name + "," + lerner + ",test," + exp_setting_embedding + "," + conftype + "," + optimization_i + ",recall," + (recall));

        }
        return hasError;
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
    private static Instances extracted(String filename, String label, NumericToNominal nominal, String separator) throws Exception {

        Instances data = null;
        if (filename.endsWith(".csv")) {
            CSVLoader training = new CSVLoader();
            training.setFieldSeparator(separator);
            training.setSource(new File(filename));
            data = training.getDataSet();
        } else if (filename.endsWith(".arff")) {
            ArffLoader training = new ArffLoader();
            training.setSource(new File(filename));
            data = training.getDataSet();
            label = "label";
        }

        nominal.setInputFormat(data);
        data = Filter.useFilter(data, nominal);
        data.setClass(data.attribute(label));
        return data;
    }

}
