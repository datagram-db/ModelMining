package it.giacomobergami.unibz;

import com.opencsv.CSVReader;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

/**
 * Apache implements a more reliable parser, although slower, than Weka's
 */
public class BetterCSVLoaderThanWeka {
    public static Instances loadCSV(String filename_and_datasetname, Set<String> toIgnore) throws FileNotFoundException {
        Instances dataset = null;
        Set<Integer> posToIgnore = new HashSet<>();
        List<String> header = null;
        ArrayList<Attribute> headerToInsert = null;

        try (CSVReader csvReader = new CSVReader(new FileReader(filename_and_datasetname))) {
            String[] values = null;
            int nTotalColumns = 0;
            int distinctId = 0;
            while ((values = csvReader.readNext()) != null) {
                if (values != null)
                    nTotalColumns = values.length;
                else
                    assert (values.length == nTotalColumns);
                if (header == null) {
                    header = Arrays.asList(values);
                    headerToInsert = new ArrayList<>(header.size());
                    for (int i = 0, headerSize = header.size(); i < headerSize; i++) {
                        String columnName = header.get(i);
                        if (!toIgnore.contains(columnName)) {
                            headerToInsert.add(new Attribute(columnName, distinctId++));
                        } else {
                            posToIgnore.add(i);
                        }
                    }
                    dataset = new Instances(filename_and_datasetname, headerToInsert, 0);
                } else {
                    double[] array = new double[distinctId];
                    int currentDistinctId = 0;
                    for (int i = 0; i<nTotalColumns; i++) {
                        if (!posToIgnore.contains(i)) {
                            if (values[i].equalsIgnoreCase("nan")) {
                                array[currentDistinctId++] = 0.0;
                            } else {
                                array[currentDistinctId++] = Double.parseDouble(values[i]);
                            }
                        }
                    }
                    DenseInstance row = new DenseInstance(1.0, array);
                    row.setDataset(dataset);
                    dataset.add(row);
                }
            }
        }catch ( Exception e) {
            e.printStackTrace();
        }
        return dataset;
    }

}
