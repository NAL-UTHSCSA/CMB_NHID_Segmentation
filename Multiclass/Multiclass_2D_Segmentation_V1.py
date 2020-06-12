


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops, label
from skimage.exposure import rescale_intensity

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

import traceback

import Multiclass_2D_DataGenerator_V1
# from iou import *
# from Metrics import weighted_dice_coefficient_loss, dice_coefficient

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

    # encoder = layers.Dropout(rate = dropout)(encoder)

    encoder = layers.Conv2D(num_filters, filtersize, padding = 'same', kernel_initializer = "he_normal")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)

    # encoder = layers.Dropout(rate = dropout)(encoder)
    # encoder = layers.Conv2D(num_filters, filtersize, padding='same')(encoder)
    # encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation('relu')(encoder)

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

    # decoder = layers.Dropout(rate = dropout)(decoder)

    decoder = layers.Conv2D(num_filters, filtersize, padding = 'same', kernel_initializer = "he_normal")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    # decoder = layers.Dropout(rate = dropout)(decoder)

    decoder = layers.Conv2D(num_filters, filtersize, padding = 'same', kernel_initializer = "he_normal")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    # decoder = layers.Dropout(rate = dropout)(decoder)

    # decoder = layers.Conv2D(num_filters, filtersize, padding='same')(decoder)
    # decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)

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
    # y_true = K.cast(y_true, K.floatx())
    # y_pred = K.cast(y_pred, K.floatx())

    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)

    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection


    # intersection = K.print_tensor(intersection, message = "intersection: ")
    # union = K.print_tensor(union, message = "union: ")

    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    ret = K.switch(K.equal(union, 0), 1.0, intersection / union)
    # ret = K.print_tensor(ret, message = "iou: ")
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

    # initialize a variable to store total IoU in
    # total_iou = K.variable(value = 0, name = "total_iou")

    total_iou = 0

    # print("Num Labels:", num_labels)
    # iterate over labels to calculate IoU for
    for label in range(1, num_labels):
        print("Label:", label)
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels



def run_model(volumelist, write_path, output_path, testline, data_channels = 2, outputclasses = 2, train_batch_size = 16, train_epochs = 50, normalize = True, scaling = True, augment_labels = None, no_rotations = 2, no_translation_axis1 = 2, no_translation_axis2 = 2):

    testline = testline.strip("\n")
    testline = testline.split(",")

    ptid = testline[0]

    output_path = os.path.join(output_path, str(ptid))
    if (os.path.exists(output_path) == False):
        os.mkdir(output_path)

    sys.stdout.write("Training for " + ptid + "\n")

    # Reset tensorflow graph and session
    tf.reset_default_graph()
    keras.backend.clear_session()

    channels = data_channels
    img_shape = (256, 256, channels)
    batch_size = train_batch_size
    epochs = train_epochs

    # Create folders where the data generated is stored.
    if (os.path.exists(write_path) == False):
        os.mkdir(write_path)

    if (os.path.exists(os.path.join(write_path, "train")) == False):
        os.mkdir(os.path.join(write_path, "train"))

    if (os.path.exists(os.path.join(write_path, "validation")) == False):
        os.mkdir(os.path.join(write_path, "validation"))


    
	############################################################
	# Generate training and validation datasets
	############################################################
	
	
	# Split the list into training and validation sets.
    train_filenames, validation_filenames = train_test_split(volumelist, test_size = 0.25)


    trainds = Multiclass_2D_DataGenerator_V1.Multiclass_2D_DataGenerator_V1(
        train_filenames,
        write_path = os.path.join(write_path, "train"),
        axis = 2,
        batch_size = batch_size,
        slice_dim = (256, 256, 256),
        n_channels = channels,
        n_classes = outputclasses,
        shuffle = True,
        pad = True,
        normalize = normalize,
        intensity_scaling = scaling,
        labels_to_augment = augment_labels,
        rotations = no_rotations,
        translations_axis1 = no_translation_axis1,
        translations_axis2 = no_translation_axis2,
        logfile = os.path.join(output_path, ptid + "_Train_DataGen.log"))

    validationds = Multiclass_2D_DataGenerator_V1.Multiclass_2D_DataGenerator_V1(
        validation_filenames,
        write_path = os.path.join(write_path, "validation"),
        axis = 2,
        batch_size = batch_size,
        slice_dim = (256, 256, 256),
        n_channels = channels,
        n_classes = outputclasses,
        shuffle = True,
        pad = True,
        normalize = normalize,
        intensity_scaling = scaling,
        labels_to_augment = augment_labels,
        rotations = no_rotations,
        translations_axis1 = no_translation_axis1,
        translations_axis2 = no_translation_axis2,
        logfile = os.path.join(output_path, ptid + "_Validation_DataGen.log"))


	############################################################
    # Define CNN model
	############################################################
    inputs = layers.Input(shape = img_shape) # 256

    encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8

    center = conv_block(encoder4_pool, 1024) # center

    decoder4 = decoder_block(center, encoder4, 512) # 16
    decoder3 = decoder_block(decoder4, encoder3, 256) # 32
    decoder2 = decoder_block(decoder3, encoder2, 128) # 64
    decoder1 = decoder_block(decoder2, encoder1, 64) # 128
    decoder0 = decoder_block(decoder1, encoder0, 32) # 256

    logits = layers.Conv2D(outputclasses, (1, 1), activation = "relu")(decoder0)
    # output = layers.Softmax()

    # Two output layers: one for the class, another for the probabilities

    probability_output = layers.Softmax(name = "class_probability_layer")(logits)
    # class_output = CustomLayers.Class_Output_Layer(name="class_output_layer")(probability_output)

    model = models.Model(inputs = [inputs], outputs = [probability_output])
    model.summary()
   
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = [mean_iou])
    


    # print(model.output.shape)
	############################################################
	# Model training
	############################################################
    
    start_time = time.time()

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    # cp = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_path, str(ptid) + "_CMB_SWI_QSM_FullSlice_Segmentation.h5"), monitor = 'val_dice_loss', save_best_only = True, verbose = 0)

    model_pathname = os.path.join(output_path, str(ptid) + "_Multiclass_" + str(channels) + "Channel_Segmentation.h5")
    cp = tf.keras.callbacks.ModelCheckpoint(filepath = model_pathname, monitor = 'val_mean_iou', mode = 'max', save_best_only = True, verbose = 0)

    csvlogger = tf.keras.callbacks.CSVLogger(filename = os.path.join(output_path, ptid + "_Training_Metrics.log"), separator = ";", append = True)

    history = model.fit_generator(generator=trainds, epochs = epochs, validation_data=validationds, callbacks = [cp, csvlogger], verbose = 2)


    dice = history.history['mean_iou']
    val_dice = history.history['val_mean_iou']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    sys.stdout.write("    Training time is: " + str(time.time() - start_time) + "\n")

    epochs_range = range(epochs)

    # Plot training/validation loss plots
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dice, label='mean_iou')
    plt.plot(epochs_range, val_dice, label='val_mean_iou')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Mean IOU')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='dice_loss')
    plt.plot(epochs_range, val_loss, label='val_dice_loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Mean Dice Loss')

    plt.savefig(os.path.join(output_path, str(ptid) + "_Training and Validation Loss.png"))
    # plt.show()
    plt.close()



	############################################################
    # Testing and saving results
	############################################################
	
    # Load the model that was saved. This will be the best saved model.
    # model = models.load_model(os.path.join(output_path, str(ptid) + "_CMB_SWI_QSM_FullSlice_Segmentation.h5"), custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})
    model = models.load_model(model_pathname, custom_objects = {'mean_iou': mean_iou})


	

    # Load the volume channesl, and generate output filenames
    channel_niftii = list()
    channel_img = list()
    channel_outputfilenames = list()
    test_mean = list()
    test_std = list()

    for ch in range(channels):
        channel_niftii.append(nib.load(testline[ch + 2])) # 0 is the ptid, 1 is the mask/groundtruth
        channel_img.append(Multiclass_2D_DataGenerator_V1.Multiclass_2D_DataGenerator_V1.pad_volume(channel_niftii[ch].get_data(), newshape = (256, 256, 256), slice_axis = 2))

        channel_outputfilenames.append(str(ptid) + "_Padded_Channel_" + str(ch) + ".nii.gz")

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


    mask_niftii = nib.load(testline[1])
    padded_mask_img = Multiclass_2D_DataGenerator_V1.Multiclass_2D_DataGenerator_V1.pad_volume(mask_niftii.get_data())



    # Filenames of the thresholded and unthresholded images
    # Create empty output unthresholded images
    raw_probability_filename = list()

    raw_probability_img = list()


    # Create an array to store the model outputs
    model_output = np.zeros((channel_img[0].shape[0], channel_img[0].shape[1], channel_img[0].shape[2], outputclasses))

    for w in range(outputclasses):
        raw_probability_filename.append(str(ptid) + "_RawProbability_Class_" + str(w + 1) + ".nii.gz")

        raw_probability_img.append(np.zeros((channel_img[0].shape), dtype = np.float)) # Change to np.float16


    class_prediction = np.zeros(channel_img[0].shape, dtype = np.uint8)

    kth_slice = 0
    while kth_slice < channel_img[0].shape[2]:

        # Create the test patch from all the channels
        test_patch = np.zeros((1, 256, 256, channels))
        for ch in range(channels):
            test_patch[0, :, :, ch] = channel_img[ch][:, :, kth_slice]


        # Create empty output patches.
        unthresholded_patch = np.zeros((256, 256, outputclasses), dtype = np.float)


        if (np.sum(test_patch[0, :, :, 0]) > 0): # Based on the SWI image, ignore blank slices

            # Apply normalization
            if (normalize == True):
                for ch in range(channels):
                    test_patch[0, :, :, ch] = test_patch[0, :, :, ch] - test_mean[ch]
                    test_patch[0, :, :, ch] = test_patch[0, :, :, ch] / test_std[ch]


            # Run the test data through the model
            predicted_patch = model.predict(x = test_patch, batch_size = 1)

            model_output[:, :, kth_slice, :] = np.copy(predicted_patch[0, :, :, :])


            class_prediction[:, :, kth_slice] = np.argmax(predicted_patch[0, :, :, :], axis = -1)

            # Process the output
            for segclass in range(outputclasses):
                unthresholded_patch[:, :, segclass] = np.copy(predicted_patch[0, :, :, segclass])

                # thresholded_patch[:, :, segclass] = np.copy(predicted_patch[0, :, :, segclass])
                # thresholded_patch[predicted_patch[0, :, :, segclass] >= 0.5, segclass] = segclass + 1
                # thresholded_patch[predicted_patch[0, :, :, segclass] < 0.5, segclass] = 0



            # # For generating and saving figures of the slices
            # plt.figure(figsize=(20, 10))
            #
            # plt.subplot(2, 2, 1)
            # plt.imshow(test_patch[0, :, :, 0], cmap="gray")
            # plt.title("SWI")
            #
            # plt.subplot(2, 2, 2)
            # plt.imshow(test_patch[0, :, :, 1], cmap="gray")
            # plt.title("QSM")
            #
            # # plt.subplot(2, 3, 3)
            # # plt.imshow(test_patch[0, :, :, 2], cmap="gray")
            # # plt.title("T1")
            # #
            # # plt.subplot(2, 3, 4)
            # # plt.imshow(test_patch[0, :, :, 3], cmap="gray")
            # # plt.title("T2")
            #
            # plt.subplot(2, 2, 3)
            # plt.imshow(kth_mask_whole_slice, cmap="gray")
            # plt.title("Actual Mask")
            #
            # plt.subplot(2, 2, 4)
            # plt.imshow(thresholded_patch, cmap="gray")
            # plt.title("Predicted Mask, th=0.5")
            #
            # plt.savefig(os.path.join(output_path, str(ptid) + "_Slice_" + str(kth_slice) + "_axial.jpg"))
            # plt.close()

        for w in range(outputclasses):
            # thresholded_img[w][:, :, kth_slice] = thresholded_patch[:, :, w]
            raw_probability_img[w][:, :, kth_slice] = unthresholded_patch[:, :, w]



        kth_slice = kth_slice + 1


    # Save the class predictions
    class_prediction_niftii = nib.Nifti1Image(class_prediction, mask_niftii.affine, mask_niftii.header)
    nib.save(class_prediction_niftii, os.path.join(output_path, ptid + "_Class_Prediction.nii.gz"))


    # Save the thresholded and unthresholded images for each output class
    for w in range(outputclasses):
        # thresholded_predicted_niftii = nib.Nifti1Image(thresholded_img[w], channel_niftii[0].affine, channel_niftii[0].header)
        # nib.save(thresholded_predicted_niftii, os.path.join(output_path, thresholded_predicted_filename[w]))

        unthresholded_predicted_niftii = nib.Nifti1Image(raw_probability_img[w], channel_niftii[0].affine, channel_niftii[0].header)
        unthresholded_predicted_niftii.set_data_dtype(np.float) # Set the datatype of the unthresholded image to float
        nib.save(unthresholded_predicted_niftii, os.path.join(output_path, raw_probability_filename[w]))


    # Save the padded original volume channels
    for ch in range(channels):
        padded_vol_channel_niftii = nib.Nifti1Image(channel_img[ch], channel_niftii[ch].affine, channel_niftii[ch].header)
        nib.save(padded_vol_channel_niftii, os.path.join(output_path, channel_outputfilenames[ch]))


    # Save the normalized volume channels
    for ch in range(channels):
        channel_img[ch] = channel_img[ch] - test_mean[ch]
        channel_img[ch] = channel_img[ch] / test_std[ch]

        normalized_padded_vol_channel_niftii = nib.Nifti1Image(channel_img[ch], channel_niftii[ch].affine, channel_niftii[ch].header)
        normalized_padded_vol_channel_niftii.set_data_dtype(np.float)

        nib.save(normalized_padded_vol_channel_niftii, os.path.join(output_path, str(ptid) + "_Normalized_Padded_Channel_" + str(ch) + ".nii.gz"))



    # Save the padded groundtruth mask for later for convenience.
    padded_mask_niftii = nib.Nifti1Image(padded_mask_img, mask_niftii.affine, mask_niftii.header)
    nib.save(padded_mask_niftii, os.path.join(output_path, str(ptid) + "_Padded_GroundtruthMask.nii.gz"))


    # Save the model_output array to file.
    np.savez_compressed(os.path.join(output_path, str(ptid) + "_RawOutput.npz"), rawoutput = model_output)


    # cmb_labels = label(thresholded_img, background = 0)
    # cmb_region_props = regionprops(cmb_labels)
    #
    # sys.stdout.write("\n  Number of CMBs detected: " + str(len(cmb_region_props)) + "\n")
    # for rs in range(len(cmb_region_props)):
    #     sys.stdout.write("\n        " + cmb_region_props[rs].)
    sys.stdout.write("Training & Testing Done\n=========================================================\n\n")










############################################################
### Program starts from here
############################################################


# The write_path is the variable used later on. 
write_path = "temporary_folder"
tmpfolder = "/path/to/"


try:
    write_path = os.path.join(tmpfolder, write_path)

except KeyError:
    write_path = os.path.join(tmpfolder, write_path)
    sys.stderr.write("\n\nCannot find ${SBIA_TMPDIR}\nUsing " + write_path + "\n\n")


try: 
    output_path = "/path/to/output/directory"
    datafile = open("/path/to/dataset_multiclass_SWI_QSM_T2.csv", "r")

    masterlist = list()
    for lines in datafile:
        masterlist.append(lines)


    # Check and make sure that output_path exists and is empty
    if (os.path.exists(output_path) == False):
        os.mkdir(output_path)



    # For multiple job submissions on cluster, with input from terminal
    # start = int(sys.argv[1])
    # stop = int(sys.argv[2])
    # write_path = write_path + "_" + str(start) + "-" + str(stop)
    # k = start
    # while (k <= stop):

    # For single jobs on desktop/laptops
    k = 0
    while (k < len(masterlist)):

        # testline is the line in masterlist that contains all the info for testing. The remaining lines are for training
        testline = masterlist[k]
        testline = testline.strip("\n")


        # Make a copy of the masterlist and then delete the line that is used for testing
        trainlist = masterlist.copy()
        del trainlist[k]


        # Check and make sure that write_path exists and is empty before training and testing for the next subject
        if (os.path.exists(write_path) == False):
            os.mkdir(write_path)
        else:
            if (len(os.listdir(write_path)) > 0):
                shutil.rmtree(write_path)
                os.mkdir(write_path)


        # Reset tensorflow graph and session
        tf.reset_default_graph()
        keras.backend.clear_session()

        # Select GPU and run program
        with tf.device('/gpu:0'):
            run_model(trainlist, write_path, output_path, testline,
                      data_channels = 3,
                      outputclasses = 3,
                      train_batch_size = 8,
                      train_epochs = 30,
                      normalize = True,
                      scaling = False,
                      augment_labels = [1, 2], 
					  no_rotations = 16, 
					  no_translation_axis1 = 10, 
					  no_translation_axis2 = 10
					  )


        # Next testline
        k = k + 1

    print("\n\nException occurred\n\n")



except:
    print("\n\nException occurred\n")
    print(sys.exc_info())
    print("\n\n")
    traceback.print_exc()

finally:
	# At the end, delete all the temporary generated data
    shutil.rmtree(write_path)


