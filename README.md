# CMB_NHID
Deep learning based segmentation for cerebral microbleeds (CMB) and non-hemorrhage iron deposits (NHID)

Needs Python version >= 3.6 

Install required python packages using the command
pip install -r requirements.txt


The binary or multiclass programs are set up to run Leave-One-Out cross-validation. 

Within the CMB_2D_Segmentation_V6.py or Multiclass_2D_Segmentation_V1.py files, make the following changes: 
- Assign the "write_path" and "tmpfolder" variables. This is the location where augmented data will be temporarily saved and used by the models. The "write_path" is the variable used in the model. 

- Assign the "output_path" variable. This is where the model will save its outputs. 

- Assign the "datafile" variable. This points to the .csv file containing all the training/testing data files. 


In order to run the multiclass or binary models, run the commands, respectively: 

python3 CMB_2D_Segmentation_V6.py

or 

python3 Multiclass_2D_Segmentation_V1.py
