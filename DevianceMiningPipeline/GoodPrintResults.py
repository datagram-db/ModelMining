
confPrettyPrint = {"bs": "IA",
                    "bs_data": "IA",
                   "dc": "Declare",
                    "dc_data": "Data+Declare",
                   "mr": "IA+MR",
                    "mr_data": "Data+IA+MR",
                   "tr": "IA+TR",
                    "tr_data": "Data+IA+TR",
                   "mra": "IA+MRA",
                    "mra_data": "Data+IA+MRA",
                   "tra": "IA+TRA",
                    "tra_data": "Data+IA+TRA",
                   "hybrid": "Hybrid",
                   "hybrid_data": "Data+Hybrid",
                   "payload": "Payload",
                   "dc_dwd": "Declare Data Aware",
                   "dc_dwd_payload": "Payload+Declare Data Aware",
                   "hybrid_dwd": "Hybrid+Declare Data Aware",
                   "hybrid_dwd_payload": "Hybrid+Payload+Declare Data Aware"
                   }

def printToFile(entry, dataset, lerner, conftype, confvalue, csvFile):
    import _io
    assert (isinstance(csvFile, _io.TextIOWrapper))
    for key in entry:
        key = confPrettyPrint[key]
        elements = ["train", "test"]
        for outcome_type in elements:
            for (acc, auc, f1,precision,recall) in zip(entry[key][outcome_type].accuracies, entry[key][outcome_type].auc, entry[key][outcome_type].f1, entry[key][outcome_type].precision, entry[key][outcome_type].recall):
                csvFile.write(dataset + "," + lerner + "," + elements + "," + conftype + "," + str(confvalue) + ",acc,"+str(acc)+"\n")
                csvFile.write(dataset + "," + lerner + "," + elements + "," + conftype + "," + str(confvalue) + ",auc," + str(auc)+"\n")
                csvFile.write(dataset + "," + lerner + "," + elements + "," + conftype + "," + str(confvalue) + ",f1," + str(f1)+"\n")
                csvFile.write(dataset + "," + lerner + "," + elements + "," + conftype + "," + str(confvalue) + ",precision," + str(precision)+"\n")
                csvFile.write(dataset + "," + lerner + "," + elements + "," + conftype + "," + str(confvalue) + ",recall," + str(recall)+"\n")