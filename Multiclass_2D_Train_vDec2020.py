


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops, label
from skimage.exposure import rescale_intensity

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

from tensorflow.python.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

import traceback
import configparser

import Multiclass_DataGen_2D_Reader_vDec2020
import Multiclass_2D_Utils_vDec2020
from Multiclass_2D_Utils_vDec2020 import pad_volume, return_as_list_of_ints, return_as_boolean, return_as_list_of_strings


import sys
import os
import time
import shutil

import numpy as np
import nibabel as nib



filtersize = (3, 3)
dropout = 0.25

def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, filtersize, padding = 'same', kernel_initializer = "he_normal")(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)

    encoder = layers.Conv2D(num_filters, filtersize, padding = 'same', kernel_initializer = "he_normal")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)

    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides = (2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides = (2, 2), padding = 'same', kernel_initializer = "he_normal")(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis = -1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    decoder = layers.Conv2D(num_filters, filtersize, padding = 'same', kernel_initializer = "he_normal")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    decoder = layers.Conv2D(num_filters, filtersize, padding = 'same', kernel_initializer = "he_normal")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    return decoder





def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * K.round(y_pred_f))
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred):
    numLabels = K.int_shape(y_pred)[-1]

    dice = 0
    for index in range(numLabels):
        dice = dice + dice_coef(y_true[:,:,:,index], y_pred[:,:,:, index])
    return dice / (numLabels)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef_multilabel(y_true, y_pred)


#
def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())

    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)

    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    ret = K.switch(K.equal(union, 0), 1.0, intersection / union)

    return ret

def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """

    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]

    total_iou = 0

    # print("Num Labels:", num_labels)
    # iterate over labels to calculate IoU for
    for label in range(1, num_labels):
        # print("Label:", label)
        total_iou = total_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels



def train_model(list_of_training_ids, write_path, root_output_path, output_subfolder, model_name = "DL_model", data_channels = 2, outputclasses = 2, slice_dim = (256, 256, 256), train_batch_size = 16, train_epochs = 50, shuffle = True):

    # Make the root output directory if it does not already exist
    if (os.path.exists(root_output_path) == False):
        os.mkdir(root_output_path)

    # Make the output subfolder where the trained model and training related data will be saved
    output_dir = os.path.join(root_output_path, output_subfolder)
    if (os.path.exists(output_dir) == False):
        os.mkdir(output_dir)

    print("Training model with data from IDs:", list_of_training_ids)
    print("  Saving all training info to: " + output_dir)



    ############################################################
    # Generate training and validation datasets
    ############################################################

    # Split the list into training and validation sets.
    train_ids, validation_ids = train_test_split(list_of_training_ids, test_size = 0.20)

    print("  Reading training dataset...")
    train_dataset = Multiclass_DataGen_2D_Reader_vDec2020.Multiclass_DataGen_2D_Reader_vDec2020(
        list_of_ids = train_ids,
        write_path = write_path,
        batch_size = train_batch_size,
        slice_dim = slice_dim,
        n_channels = data_channels,
        n_classes = outputclasses,
        shuffle = shuffle
    )

    # Multiclass_2D_Utils_vDec2020.visual_check_datagen_reader(train_dataset)

    print("  Reading validation dataset...")
    validation_dataset = Multiclass_DataGen_2D_Reader_vDec2020.Multiclass_DataGen_2D_Reader_vDec2020(
        list_of_ids = validation_ids,
        write_path = write_path,
        batch_size = train_batch_size,
        slice_dim = slice_dim,
        n_channels = data_channels,
        n_classes = outputclasses,
        shuffle = shuffle
    )

    # Multiclass_2D_Utils_vDec2020.visual_check_datagen_reader(validation_dataset)


    # Reset tensorflow graph and session
    tf.reset_default_graph()
    keras.backend.clear_session()

	############################################################
    # Define CNN model
	############################################################
    img_shape = (slice_dim[0], slice_dim[1], data_channels)

    inputs = layers.Input(shape = img_shape) # 256

    encoder0_pool, encoder0 = encoder_block(inputs, 32) 
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) 
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) 
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) 
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) 

    center = conv_block(encoder4_pool, 1024) # center

    decoder4 = decoder_block(center, encoder4, 512) 
    decoder3 = decoder_block(decoder4, encoder3, 256) 
    decoder2 = decoder_block(decoder3, encoder2, 128) 
    decoder1 = decoder_block(decoder2, encoder1, 64) 
    decoder0 = decoder_block(decoder1, encoder0, 32) 

    logits = layers.Conv2D(outputclasses, (1, 1), activation = "relu")(decoder0)

    probability_output = layers.Softmax(name = "class_probability_layer")(logits)

    model = models.Model(inputs = [inputs], outputs = [probability_output])
    model.summary()
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = [mean_iou])
    

	############################################################
	# Model training
	############################################################
    
    train_start_time = time.time()

    
    model_path = os.path.join(output_dir, model_name + "_" + str(outputclasses) + "_Class_2D_DLSeg_Model_" + str(data_channels) + "Ch.h5")
    cp = tf.keras.callbacks.ModelCheckpoint(filepath = model_path, monitor = 'val_mean_iou', mode = 'max', save_best_only = True, verbose = 0)
    csvlogger = tf.keras.callbacks.CSVLogger(filename = os.path.join(output_dir, model_name + "_Training_Metrics.log"), separator = ";", append = True)

    history = model.fit_generator(generator = train_dataset, epochs = train_epochs, validation_data = validation_dataset, callbacks = [cp, csvlogger], verbose = 2)


    dice = history.history['mean_iou']
    val_dice = history.history['val_mean_iou']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print("  Training time required: " + "{0:3.2f}".format(time.time() - train_start_time) + " seconds\n")

    epochs_range = range(train_epochs)

    try: 
        # Plot training/validation loss plots
        plt.figure(figsize = (16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, dice, label = 'mean_iou')
        plt.plot(epochs_range, val_dice, label = 'val_mean_iou')
        plt.legend(loc = 'upper right')
        plt.title('Training and Validation Mean IOU')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label = 'dice_loss')
        plt.plot(epochs_range, val_loss, label = 'val_dice_loss')
        plt.legend(loc = 'upper right')
        plt.title('Training and Validation Mean Dice Loss')

        plt.savefig(os.path.join(output_dir, model_name + "_Training_and_Validation_Loss.png"))
        # plt.show()
        plt.close()

    except: 
        import traceback

        print("\n\nException occurred\n")
        print(sys.exc_info()[0])
        print("Could not produce plots\n\n")

        traceback.print_exc()


    return model_path





############################################################
# Applying the model to test data and saving results
############################################################
def apply_model(model_path, testline, root_output_path, output_subfolder, data_channels = 2, outputclasses = 2, slice_dim = (256, 256, 256), normalize = False, scaling = False):

    testline = testline.replace("\n", "").split(",")

    # Create the output directory for this test, if it does not already exist
    output_dir = os.path.join(root_output_path, output_subfolder)
    if (os.path.exists(output_dir) == False):
        os.mkdir(output_dir)


    # This is where the test-specific outputs will be saved
    subject_output_dir = os.path.join(output_dir, testline[0])
    if (os.path.exists(subject_output_dir) == False):
        os.mkdir(subject_output_dir)

    output_basename = os.path.join(subject_output_dir, testline[0])

    # Load the model that was saved. This should be the best saved model.
    model = models.load_model(model_path, custom_objects = {'mean_iou': mean_iou})


    # Load the volume channesl, and generate output filenames
    channel_niftii = list()
    channel_img = list()
    channel_outputfilenames = list()
    test_mean = list()
    test_std = list()

    for ch in range(data_channels):
        channel_niftii.append(nib.load(testline[ch + 2])) # 0 is the ptid, 1 is the mask/groundtruth
        tmpvol, _ = Multiclass_2D_Utils_vDec2020.pad_volume(channel_niftii[ch].get_data(), newshape = slice_dim, slice_axis = 2)
        channel_img.append(tmpvol)

        channel_outputfilenames.append(output_basename + "_Padded_Channel_" + str(ch) + ".nii.gz")

        # Compute the mean and std dev for the test data
        if (normalize == True):
            test_mean.append(np.mean(channel_img[ch]))
            test_std.append(np.std(channel_img[ch]))

        # Rescale the images' intensity
        if (scaling == True):
            channel_img[ch] = rescale_intensity(channel_img[ch], out_range = (0, 255))

        # # Normalize the image's intensity
        # if (normalize == True):
        #     channel_img[ch] = channel_img[ch] - test_mean[ch]
        #     channel_img[ch] = channel_img[ch] - test_std[ch]

    if (testline[1] is not "") and (os.path.isfile(testline[1]) == True):
        mask_niftii = nib.load(testline[1])
        padded_mask_img, _ = Multiclass_2D_Utils_vDec2020.pad_volume(mask_niftii.get_data())



    # Filenames of the thresholded and unthresholded images
    # Create empty output unthresholded images
    raw_probability_filename = list()

    raw_probability_img = list()


    # Create an array to store the model outputs
    model_output = np.zeros((channel_img[0].shape[0], channel_img[0].shape[1], channel_img[0].shape[2], outputclasses))

    for w in range(outputclasses):
        raw_probability_filename.append(output_basename + "_RawProbability_Class_" + str(w + 1) + ".nii.gz")

        raw_probability_img.append(np.zeros((channel_img[0].shape), dtype = np.float)) # Change to np.float16


    class_prediction = np.zeros(channel_img[0].shape, dtype = np.uint8)

    kth_slice = 0
    while kth_slice < channel_img[0].shape[2]:

        # Create the test patch from all the channels
        test_patch = np.zeros((1, slice_dim[0], slice_dim[1], data_channels))
        for ch in range(data_channels):
            test_patch[0, :, :, ch] = channel_img[ch][:, :, kth_slice]


        # Create empty output patches.
        unthresholded_patch = np.zeros((slice_dim[0], slice_dim[1], outputclasses), dtype = np.float)


        if (np.sum(test_patch[0, :, :, 0]) > 0): # Based on the SWI image, ignore blank slices

            # Apply normalization
            if (normalize == True):
                for ch in range(data_channels):
                    test_patch[0, :, :, ch] = test_patch[0, :, :, ch] - test_mean[ch]
                    test_patch[0, :, :, ch] = test_patch[0, :, :, ch] / test_std[ch]


            # Run the test data through the model
            predicted_patch = model.predict(x = test_patch, batch_size = 1)
            model_output[:, :, kth_slice, :] = np.copy(predicted_patch[0, :, :, :])
            class_prediction[:, :, kth_slice] = np.argmax(predicted_patch[0, :, :, :], axis = -1)

            # Process the output
            for segclass in range(outputclasses):
                unthresholded_patch[:, :, segclass] = np.copy(predicted_patch[0, :, :, segclass])



        for w in range(outputclasses):
            # thresholded_img[w][:, :, kth_slice] = thresholded_patch[:, :, w]
            raw_probability_img[w][:, :, kth_slice] = unthresholded_patch[:, :, w]



        kth_slice = kth_slice + 1


    print("  Writing output files...")

    # Save the class predictions
    class_prediction_niftii = nib.Nifti1Image(class_prediction, channel_niftii[0].affine, channel_niftii[0].header)
    class_prediction_niftii.set_data_dtype(np.uint8)
    nib.save(class_prediction_niftii, output_basename + "_Class_Prediction.nii.gz")


    # Save the thresholded and unthresholded images for each output class
    for w in range(outputclasses):
        unthresholded_predicted_niftii = nib.Nifti1Image(raw_probability_img[w], channel_niftii[0].affine, channel_niftii[0].header)
        unthresholded_predicted_niftii.set_data_dtype(np.float) # Set the datatype of the unthresholded image to float
        nib.save(unthresholded_predicted_niftii, raw_probability_filename[w])


    # Save the padded original volume channels
    for ch in range(data_channels):
        padded_vol_channel_niftii = nib.Nifti1Image(channel_img[ch], channel_niftii[ch].affine, channel_niftii[ch].header)
        nib.save(padded_vol_channel_niftii, channel_outputfilenames[ch])


    # Save the normalized volume channels
    for ch in range(data_channels):
        channel_img[ch] = channel_img[ch] - test_mean[ch]
        channel_img[ch] = channel_img[ch] / test_std[ch]

        normalized_padded_vol_channel_niftii = nib.Nifti1Image(channel_img[ch], channel_niftii[ch].affine, channel_niftii[ch].header)
        normalized_padded_vol_channel_niftii.set_data_dtype(np.float)

        nib.save(normalized_padded_vol_channel_niftii, output_basename + "_Normalized_Padded_Channel_" + str(ch) + ".nii.gz")



    # Save the padded groundtruth mask for later for convenience.
    if (testline[1] is not "") and (os.path.isfile(testline[1]) == True):
        padded_mask_niftii = nib.Nifti1Image(padded_mask_img, mask_niftii.affine, mask_niftii.header)
        nib.save(padded_mask_niftii, output_basename + "_Padded_GroundtruthMask.nii.gz")


    # Save the model_output array to file.
    np.savez_compressed(output_basename + "_RawOutput.npz", rawoutput = model_output)


#









############################################################
### Program starts from here
############################################################
def main():
    if (len(sys.argv) != 3):
        print("\nERROR")
        print("  Incorrect number of parameters.")
        print("  Usage is ")
        print("    <python_executable> Multiclass_DataGen_2D_Train_vDec2020.py <config_file.ini> <train_test_file.ini>\n")

    else:
        print("Beginning training...")

        config_file = sys.argv[1]
        train_test_file = sys.argv[2]
        print("Config file is:\n  " + config_file)
        print("Train/Test file is:\n  " + train_test_file)



        gpu = 0
        try:
            gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        except:
            print("ERROR: Cannot find environment variable CUDA_VISIBLE_DEVICES\nSetting CUDA_VISIBLE_DEVICES to 0")
            gpu = 0


        cfg_train_params = configparser.ConfigParser()
        cfg_train_params.read(config_file)

        write_path = cfg_train_params["PATHS"]["write_path"]
        root_output_path = cfg_train_params["PATHS"]["root_output_path"]
        slice_axis = int(cfg_train_params["MODEL_PARAMETERS"]["slice_axis"])
        data_channels = int(cfg_train_params["MODEL_PARAMETERS"]["data_channels"])
        output_classes = int(cfg_train_params["MODEL_PARAMETERS"]["output_classes"])
        slice_dim = return_as_list_of_ints(cfg_train_params["MODEL_PARAMETERS"]["input_shape"])
        train_batch_size = int(cfg_train_params["MODEL_PARAMETERS"]["train_batch_size"])
        train_epochs = int(cfg_train_params["MODEL_PARAMETERS"]["train_epochs"])
        shuffle = return_as_boolean(cfg_train_params["DATA_AUGMENTATION"]["shuffle"])
        pad = return_as_boolean(cfg_train_params["DATA_AUGMENTATION"]["pad_image"])
        unit_mean_variance_normalize = return_as_boolean(cfg_train_params["DATA_AUGMENTATION"]["unit_mean_variance_normalize"])
        scaling = return_as_boolean(cfg_train_params["DATA_AUGMENTATION"]["scaling"])
        augment_labels = return_as_list_of_ints(cfg_train_params["DATA_AUGMENTATION"]["augment_labels"])
        balance_bkgrnd_slices = return_as_boolean(cfg_train_params["DATA_AUGMENTATION"]["balance_bkgrnd_slices"])
        no_rotations = int(cfg_train_params["DATA_AUGMENTATION"]["no_rotations"])
        no_tralations_axis1 = int(cfg_train_params["DATA_AUGMENTATION"]["no_translations_axis1"])
        no_tralations_axis2 = int(cfg_train_params["DATA_AUGMENTATION"]["no_translations_axis2"])

        print("Model training parameters are:")
        print("  write_path=", write_path, sep = "")
        print("  slice_axis=", slice_axis, sep = "")
        print("  data_channels=", data_channels, sep = "")
        print("  output_classes=", output_classes, sep = "")
        print("  slice_dim=", slice_dim, sep = "")
        print("  train_batch_size=", str(train_batch_size), sep = "")
        print("  train_epochs=", str(train_epochs), sep = "")
        print("  shuffle=", shuffle, sep = "")
        print("  pad_image=", pad, sep = "")
        print("  unit_mean_variance_normalize=", unit_mean_variance_normalize, sep = "")
        print("  scaling=", scaling, sep = "")
        print("  augment_labels=", augment_labels, sep = "")
        print("  balance_bkgrnd_slices=", balance_bkgrnd_slices, sep = "")
        print("  no_rotations=", no_rotations, sep = "")
        print("  no_tralations_axis1=", no_tralations_axis1, sep = "")
        print("  no_tralations_axis2=", no_tralations_axis2, sep = "")
        print("")




        cfg_train_test_list = configparser.ConfigParser()
        cfg_train_test_list.read(train_test_file)

        model_name = cfg_train_test_list["MODEL_NAME"]["model_name"]
        output_subfolder = cfg_train_test_list["OUTPUT_SUBFOLDER"]["output_subfolder"]
        train_list = return_as_list_of_strings(cfg_train_test_list["TRAINING_LIST"]["train_list"])
        test_list = return_as_list_of_strings(cfg_train_test_list["TESTING_LIST"]["test_list"])

        datafile = cfg_train_params["PATHS"]["data_file"]
        print("Data file is: ", datafile, sep = "")

        print("Training/Testing info")
        print("  Model training list (IDs) is:", *train_list)
        print("  Model testing list (IDs) is :", *test_list)
        print("  Output Subfolder: " + output_subfolder, sep = "")


        model_path = None

        # Reset tensorflow graph and session
        tf.reset_default_graph()
        keras.backend.clear_session()
        with tf.device('/gpu:' + str(gpu)):
            model_path = train_model(list_of_training_ids = train_list,
                                         write_path = write_path,
                                         root_output_path = root_output_path,
                                         output_subfolder = output_subfolder,
                                         model_name = model_name,
                                         data_channels = data_channels,
                                         outputclasses = output_classes,
                                         slice_dim = slice_dim,
                                         train_batch_size = train_batch_size,
                                         train_epochs = train_epochs,
                                         shuffle = shuffle
            )

            # Once training is done, read the data file (csv file) and apply the model to the testing data,
            # based on the IDs in test_list
            if (len(test_list) > 0):
                print("\nApplying model to test data")
                

                if (model_path is not None) and (os.path.exists(model_path)) and (os.path.isfile(model_path)):
                    print("  Model is: ", model_path, sep = "")

                    file = open(datafile, "r")

                    for line in file:
                        line = line.strip("\n")
                        splitline = line.split(",")
                        test_subject_id = splitline[0]

                        gt_file = splitline[1]
                        if (gt_file is "") and (os.path.isfile(gt_file) == False):
                            print("  No groundtruth file specified.")

                        if test_subject_id in test_list:
                            print("  Applying model to data with ID " + test_subject_id)

                            apply_model(model_path = model_path,
                                        testline = line,
                                        root_output_path = root_output_path,
                                        output_subfolder = output_subfolder,
                                        data_channels = data_channels,
                                        outputclasses = output_classes,
                                        normalize = unit_mean_variance_normalize,
                                        slice_dim = slice_dim, 
                                        scaling = scaling
                                        ) 


        print("\nTraining completed.\n================================================================")


if (__name__ == "__main__"):
    main()
