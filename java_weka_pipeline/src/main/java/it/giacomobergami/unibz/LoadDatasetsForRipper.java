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
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.lang.reflect.Field;
import java.util.ArrayList;

public class LoadDatasetsForRipper implements AutoCloseable {

    CSVLoader training;
    ArffLoader training2;

    public LoadDatasetsForRipper() {
        training = new CSVLoader();
        training2 = new ArffLoader();
    }

    @Override
    public void close() throws Exception {
        // Forcing the deletion, forsooth!
        // We need to save disk space as much as we can on Big Data!
        Field m_dataDumper = training.getClass().getDeclaredField("m_dataDumper");
        m_dataDumper.setAccessible(true);
        ((PrintWriter)m_dataDumper.get(training)).close();
        Field m_tempFile = training.getClass().getDeclaredField("m_tempFile");
        m_tempFile.setAccessible(true);
        ((File)m_tempFile.get(training)).delete();
    }

    public boolean dumpFile(String folder_name, String train, String test, String exp_setting_embedding, String sep, PrintWriter csvFile, PrintWriter ruleFile, int max_iteration) throws Exception {
        String lerner = "RipperK";
        String conftype = "iteration";
        NumericToNominal nominal = new NumericToNominal();
        nominal.setOptions(new String[]{"-R", "last"});
        boolean hasError = false;
        for (int i = 0; i< max_iteration; i++) {
            int optimization_i = i * 2;
            System.out.println("\t - optimization: "+optimization_i);
            JRip ripperk = new JRip();
            Instances trainingSet = extracted(train, "Label", nominal, sep);
            Instances testingSet = extracted(test,  "Label", nominal, sep);;
            assert (!test.isEmpty());
            /*if (test.isEmpty()) {
                int trainSize = (int) Math.round(trainingSet.size()*trainSizePerc);
                int testSize = trainingSet.numInstances() - trainSize;
                Instances tmp = new Instances(trainingSet, 0, trainSize);
                testingSet = new Instances(trainingSet, trainSize, testSize);
                trainingSet = tmp;
            } else {
                testingSet = extracted(test,  "Label", nominal, sep);
            }*/

            Evaluation evaluation = new Evaluation(trainingSet);
            ripperk.setOptimizations(optimization_i);
            try {
                ripperk.buildClassifier(trainingSet);
            } catch (Exception e) {
                e.printStackTrace();
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

    static ForceEveryColumnToNumeric f = new ForceEveryColumnToNumeric();

    /**
     * Loading a dataset so to be used by JRipper
     *
     * @param filename      File to be loaded
     * @param label         Label associated to the class
     * @param nominal       Converter from numbers to string classes
     * @return              Converted and loaded dataset
     * @throws Exception
     */
    private Instances extracted(String filename, String label, NumericToNominal nominal, String separator) throws Exception {

        Instances data = null;
        if (filename.endsWith(".csv")) {
            training.setFieldSeparator(separator);
            training.setSource(new File(filename));
            data = training.getDataSet();
        } else if (filename.endsWith(".arff")) {
            training2.setSource(new File(filename));
            data = training2.getDataSet();
            label = "label";
        }

        data.deleteAttributeAt(data.attribute("Case_ID").index());
        data = ForceEveryColumnToNumeric.transform(data);
        nominal.setInputFormat(data);
        data = Filter.useFilter(data, nominal);
        data.setClass(data.attribute(label));
        return data;
    }


}
