package it.giacomobergami.unibz;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.opencsv.CSVReader;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SparseInstance;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.jar.Attributes;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CrawlResultsFolder implements AutoCloseable {

    Map<String, String> map;
    Map<String, List<TrainingTest>> yaml;
    PrintWriter csvFile;
    PrintWriter ruleFile;
    String folderName;

    public CrawlResultsFolder(String path_name) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        mapper.findAndRegisterModules();
        File folder = new File(path_name);
        File yamlFile = Files.list(folder.toPath()).filter(x -> x.toString().endsWith(".yaml")).findAny().get().toFile();


        folderName = folder.getName().replace("_results", "");
        TypeReference<Map<String, List<TrainingTest>>> typeRef = new TypeReference<>() {};
        yaml = mapper.readValue(yamlFile, typeRef);
        //obj.forEach((k,v) -> System.out.println(k+": "+v));

        map = Stream.of(new String[][] {
                {"bs", "IA"},
                {"bs_data", "Data+IA"},
                {"dc", "Declare"},
                {"dc_data", "Data+Declare"},
                {"mr", "IA+MR"},
                {"mr_data", "Data+IA+MR"},
                {"tr", "IA+TR"},
                {"tr_data", "Data+IA+TR"},
                {"mra", "IA+MRA"},
                {"mra_data", "Data+IA+MRA"},
                {"tra", "IA+TRA"},
                {"tra_data", "Data+IA+TRA"},
                {"hybrid", "Hybrid"},
                {"hybrid_data", "Data+Hybrid"},
                {"payload_for_training", "Payload"},
                {"dc_dwd", "Declare Data Aware"},
                {"dc_dwd_payload", "Payload+Declare Data Aware"},
                {"hybrid_dwd", "Hybrid+Declare Data Aware"},
                {"hybrid_dwd_payload", "Hybrid+Payload+Declare Data Aware"}
        }).collect(Collectors.toMap(data ->  data[0], data ->  data[1]));

        File csv_file = Files.list(folder.toPath()).filter(x -> x.toString().endsWith(".csv")).findAny().get().toFile();
        File rule_file = Files.list(folder.toPath()).filter(x -> x.toString().endsWith(".txt")).findAny().get().toFile();

        FileWriter csv_fw = new FileWriter(csv_file, true);
        BufferedWriter csv_bw = new BufferedWriter(csv_fw);
        csvFile = new PrintWriter(csv_bw);

        FileWriter rule_fw = new FileWriter(rule_file, true);
        BufferedWriter rule_bw = new BufferedWriter(rule_fw);
        ruleFile = new PrintWriter(rule_bw);
    }

    void dump() throws Exception {
        for (String keys : map.keySet()) {
            String dump_conf = map.get(keys);
            System.out.println("Running: "+dump_conf);
            if (yaml.containsKey(keys))
            for (TrainingTest tt : yaml.get(keys)) {
                String training_csv = tt.train;
                String testing_csv = tt.test;
                boolean hasError = false;
                String strError = "[generic]";
                try (LoadDatasetsForRipper instance = new LoadDatasetsForRipper()) {
                    hasError = instance.dumpFile(folderName, training_csv, testing_csv, dump_conf, ",", csvFile, ruleFile, 10);
                } catch (Exception e) {
                    e.printStackTrace();
                    strError = ": " + e.toString();
                    hasError = true;
                    // REDO!
                }
                if (hasError) {
                    System.err.println("The dataset raised a Weka error " + strError);
                    System.err.println("This happened while running [map key] " + keys +", dump_conf="+dump_conf+" training/test="+tt);
                    System.err.println("Ignoging it, so to attempt to write the other results");
                    System.err.println("");
                }
                csvFile.flush();
                ruleFile.flush();
            }
        }
    }

    public static void main(String args[]) throws IOException {
        try (CrawlResultsFolder f = new CrawlResultsFolder("/media/giacomo/BigData/output_pipeline/bpi11_mr_tr_results")) {
            f.dump();
        }catch (Exception e) {
e.printStackTrace();
        }
    }

    @Override
    public void close() throws Exception {
        ruleFile.close();
        csvFile.close();
}

    }