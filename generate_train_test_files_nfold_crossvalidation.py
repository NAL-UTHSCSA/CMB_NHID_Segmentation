
import os
from configparser import ConfigParser
import numpy as np
import sys

from sklearn.model_selection import KFold

if (len(sys.argv) != 4):
    print("\nERROR\n  Incorrect usage\n\n  Usage is\n   python3 generate_train_test_files.py <datafile.csv> <POSTFIX_LABEL> <NFOLDS>")
else:
    datafile = sys.argv[1]
    postfix = sys.argv[2]
    nfolds = int(sys.argv[3])

    if (os.path.exists(datafile) == True and os.path.isfile(datafile) == True):

        # First make a list of all the IDs
        idlist = list()
        data = open(datafile, "r")

        for line in data:
            line = line.strip("\n").split(",")

            id = line[0]
            idlist.append(id)

        idlist = np.asarray(idlist)

        kf = KFold(n_splits = nfolds, shuffle = True)
        k = 0
        for train_index, test_index in kf.split(idlist):
            outputpath = os.path.dirname(datafile)

            cfgfile = open(os.path.join(outputpath, "Fold_" + str(k) + "_train_test.ini"), "w")
            configfile = ConfigParser()

            # List all the other IDs as training list
            configfile.add_section("TRAINING_LIST")
            train_ids = idlist[train_index]
            configfile.set("TRAINING_LIST", "train_list", ",".join(train_ids))

            configfile.add_section("TESTING_LIST")
            test_ids = idlist[test_index]
            configfile.set("TESTING_LIST", "test_list", ",".join(test_ids))

            # Set the output subfolder's name to be the current ID
            configfile.add_section("OUTPUT_SUBFOLDER")
            configfile.set("OUTPUT_SUBFOLDER", "output_subfolder", "Fold_" + str(k))

            # Set the current ID in the model name
            configfile.add_section("MODEL_NAME")
            configfile.set("MODEL_NAME", "model_name", "Fold_" + str(k) + "_" + postfix)

            configfile.write(cfgfile)
            cfgfile.close()

            k = k + 1


    else:
        print("\nERROR\n  File is not found\nFile provided=" + datafile)

