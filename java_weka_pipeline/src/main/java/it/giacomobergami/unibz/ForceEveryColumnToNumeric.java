package it.giacomobergami.unibz;

import org.w3c.dom.Attr;
import weka.core.*;
import weka.filters.Filter;

import java.util.*;

public class ForceEveryColumnToNumeric  {

    public static Instances transform(Instances table) throws Exception {

        Set<Integer> nonNumericAttributes = new HashSet<>();
        // First: check if there are any non-numerical instances.
        ArrayList<Attribute> attributes = new ArrayList<>(table.numAttributes());
        int numAttributes = table.numAttributes();
        for (int i = 0, n = table.numAttributes(); i<n; i++) {
            Attribute currentAttribute = table.attribute(i);
            if (!currentAttribute.isNumeric()) {
                nonNumericAttributes.add(i);
            }
            attributes.add(new Attribute(currentAttribute.name(), i));
        }

        if (nonNumericAttributes.isEmpty()) {
            return table; // No change is required!
        }

        // Else, load the numerical representation for all the elements, forsooth!
        Instances newDatabase = new Instances(table.relationName(), attributes, table.numInstances());
        for (int i = 0, n = table.numInstances(); i<n; i++) {
            double array[] = new double[numAttributes];
            Instance currInstance = table.instance(i);
            for (int j = 0; j<numAttributes; j++) {
                if (nonNumericAttributes.contains(j)) {
                    String value = currInstance.stringValue(j);
                    if (value.toLowerCase().equals("nan")) {
                        array[j] = 0.0;
                    } else {
                        try {
                            array[j] = Double.valueOf(value);
                        }catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                } else {
                    array[j] = currInstance.value(j);
                }
            }
            newDatabase.add(new DenseInstance(1.0, array));
        }
        return newDatabase;
    }
}
