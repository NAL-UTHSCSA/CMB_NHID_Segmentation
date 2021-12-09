
# CMB_NHID
These are the scripts for a Multiclass 2D UNet variant for segmenting cerebral microbleeds and non-hemorrhage iron deposits in the basal ganglia. 
The code is based on Python (3.6) , Keras and TensorFlow (1.15)


Needs Python version >= 3.6 

Install the required python packages for your virtual environment using the command
pip install -r requirements.txt




Usage: 

1) Create a comma-separated datafile (.csv) in the following format: 
Subject01,/path/to/subject01/subject01_groundtruth.nii.gz,/path/to/subject01/subject01_swi.nii.gz,/path/to/subject01/subject01_another_modality.nii.gz
Subject02,/path/to/subject02/subject02_groundtruth.nii.gz,/path/to/subject02/subject02_swi.nii.gz,/path/to/subject02/subject02_another_modality.nii.gz
Subject03,/path/to/subject03/subject03_groundtruth.nii.gz,/path/to/subject03/subject03_swi.nii.gz,/path/to/subject03/subject03_another_modality.nii.gz
Subject04,/path/to/subject04/subject04_groundtruth.nii.gz,/path/to/subject04/subject04_swi.nii.gz,/path/to/subject04/subject04_another_modality.nii.gz
Subject05,/path/to/subject05/subject05_groundtruth.nii.gz,/path/to/subject05/subject05_swi.nii.gz,/path/to/subject05/subject05_another_modality.nii.gz
Subject06,/path/to/subject06/subject06_groundtruth.nii.gz,/path/to/subject06/subject06_swi.nii.gz,/path/to/subject06/subject06_another_modality.nii.gz
...
...
...


Each row should contain the information of a single subject. See Example_Datafile.csv

2) Use a text editor to set up the parameters of the configuration file (Example_config_file.ini). In this file, the parameters for the paths, model, training and data augmentations are specified.

3) Generate a series of training/testing configuration files (*TrainTestFile.ini) using the generate_train_test_files_nfold_crossvalidation.py script. Usage is: 
	python3 generate_train_test_files_nfold_crossvalidation.py </path/to/datafile.csv> <POSTFIX_LABEL> <NFOLDS>

Example, 
	python3 generate_train_test_files_nfold_crossvalidation.py Example_Datafile.csv Expt_X 5

4) Generate the data to be used for model training by running the Multiclass_DataGen_2D_Writer_vDec2020.py script. This is a time consuming process (depending on the number of augmentations specified) and NEEDS TO BE DONE ONLY ONCE. Usage is: 
	python3 Multiclass_DataGen_2D_Writer_vDec2020.py </path/to/config_file.ini> </path/to/datafile.csv>


5) Run each individual train/test session using the training/testing configuration files previously generated: 
	python3 Multiclass_DataGen_2D_Train_vDec2020.py </path/to/config_file.ini> </path/to/train_test_file.ini>

The training can be a time consuming process, depending on the specified batch size, GPU memory availability and training data generated. 
For each *TrainTestFile.ini, the code will first train the model using the data of subjects specified in the training list. The model will run for the specified number of epochs. Once all epochs are completed, the trained model will be applied to the specified list of test subjects' data and the predictions will be saved in the specified output_subfolder. 








