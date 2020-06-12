


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops, label

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

import DataGenerator_V7
# from Metrics import weighted_dice_coefficient_loss, dice_coefficient

import sys
import os
import time
import shutil

import numpy as np
import nibabel as nib





filtersize = (3, 3)


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, filtersize, padding='same', kernel_initializer = "he_normal")(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)

    encoder = layers.Conv2D(num_filters, filtersize, padding='same', kernel_initializer = "he_normal")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)

    # encoder = layers.Conv2D(num_filters, filtersize, padding='same')(encoder)
    # encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation('relu')(encoder)

    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding = 'same', kernel_initializer = "he_normal")(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    decoder = layers.Conv2D(num_filters, filtersize, padding='same', kernel_initializer = "he_normal")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    decoder = layers.Conv2D(num_filters, filtersize, padding='same', kernel_initializer = "he_normal")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    # decoder = layers.Conv2D(num_filters, filtersize, padding='same')(decoder)
    # decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)

    return decoder





def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss



def iou(y_true, y_pred):
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
    # y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    # y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def bce_iou_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + iou_loss(y_true, y_pred)
    return loss

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)




def run_model(volumelist, write_path, output_path, testline, data_channels = 2, train_batch_size = 16, train_epochs = 30, normalize = True, no_rotations = 2, no_translation_axis1 = 2, no_translation_axis2 = 2):

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
    # keras.backend.set_floatx("float32")

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


    trainds = DataGenerator_V7.DataGenerator_V7(
        train_filenames,
        write_path = os.path.join(write_path, "train"),
        axis = 2,
        batch_size = batch_size,
        slice_dim = (256, 256, 256),
        n_channels = channels,
        n_classes = 2,
        shuffle = True,
        pad = True,
        normalize = normalize,
        augment_cmbs = True,
        augment_regular = True,
        rotations = no_rotations,
        translations_axis1 = no_translation_axis1,
        translations_axis2 = no_translation_axis2,
        logfile = os.path.join(output_path, ptid + "_Train_DataGen.log"))

    validationds = DataGenerator_V7.DataGenerator_V7(
        validation_filenames,
        write_path = os.path.join(write_path, "validation"),
        axis = 2,
        batch_size = batch_size,
        slice_dim = (256, 256, 256),
        n_channels = channels,
        n_classes = 2,
        shuffle = True,
        pad = True,
        normalize = normalize,
        augment_cmbs = True,
        augment_regular = True,
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

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)


    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = 'adam', loss = bce_iou_loss, metrics = [iou_loss])
    # model.compile(optimizer = 'adam', loss = weighted_dice_coefficient_loss, metrics = [dice_coefficient])

    # model.summary()


	############################################################
	# Model training
	############################################################

    start_time = time.time()



    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path, str(ptid) + "_CMB_SWI_QSM_FullSlice_Segmentation.h5"), monitor='val_iou_loss', save_best_only=True, verbose=0)
    csvlogger = tf.keras.callbacks.CSVLogger(filename = os.path.join(output_path, ptid + "_Training_Metrics.log"), separator = ";", append = True)

    history = model.fit_generator(generator=trainds, epochs = epochs, validation_data=validationds, callbacks = [cp, csvlogger], verbose = 2)



    dice = history.history['iou_loss']
    val_dice = history.history['val_iou_loss']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    sys.stdout.write("    Training time is: " + str(time.time() - start_time) + "\n")

    epochs_range = range(epochs)

    # Plot training/validation loss plots
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dice, label='Training IOU Loss')
    plt.plot(epochs_range, val_dice, label='Validation IOU Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation IOU Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(os.path.join(output_path, str(ptid) + "_Training and Validation Loss.png"))
    # plt.show()



	############################################################
    # Testing and saving results
	############################################################
	
	
    # Load the model that was saved. This will be the best saved model.
    model = models.load_model(os.path.join(output_path, str(ptid) + "_CMB_SWI_QSM_FullSlice_Segmentation.h5"), custom_objects={'bce_iou_loss': bce_iou_loss, 'iou_loss': iou_loss})

   
    thresholded_predicted_filename = str(ptid) + "_Thresholded_0.5_FullSlice_Predicted.nii.gz"
    unthresholded_predicted_filename = str(ptid) + "_Unthresholded_FullSlice_Predicted.nii.gz"


    channel_niftii = list()
    channel_img = list()
    channel_outputfilenames = list()
    test_mean = list()
    test_std = list()

    for ch in range(channels):
        channel_niftii.append(nib.load(testline[ch + 2])) # 0 is the ptid, 1 is the mask/groundtruth
        channel_img.append(DataGenerator_V7.DataGenerator_V7.pad_volume(channel_niftii[ch].get_data(), newshape = (256, 256, 256), slice_axis = 2))

        channel_outputfilenames.append(str(ptid) + "_Padded_Channel_" + str(ch) + ".nii.gz")

        # Compute the mean and std dev for the test data
        if (normalize == True):
            test_mean.append(np.mean(channel_img[ch]))
            test_std.append(np.std(channel_img[ch]))





    # mask_niftii = nib.load(testline[1])
    # mask_img = DataGenerator_V7.DataGenerator_V7.pad_volume(mask_niftii.get_fdata().astype(mask_niftii.get_data_dtype()))


    # Create empty output images
    thresholded_img = np.zeros((channel_img[0].shape), dtype = np.int)
    unthresholded_img = np.zeros((channel_img[0].shape), dtype = np.float32)


    kth_slice = 0
    while kth_slice < channel_img[0].shape[2]:

        # Create the test patch from all the channels
        test_patch = np.zeros((1, 256, 256, channels))
        for ch in range(channels):
            test_patch[0, :, :, ch] = channel_img[ch][:, :, kth_slice]


        # Create empty output patches.
        thresholded_patch = np.zeros((256, 256))
        unthresholded_patch = np.zeros((256, 256))


        if (np.sum(test_patch[0, :, :, 0]) > 0): # Based on the SWI image, ignore blank slices

            # Apply normalization
            if (normalize == True):
                for ch in range(channels):
                    test_patch[0, :, :, ch] = test_patch[0, :, :, ch] - test_mean[ch]
                    test_patch[0, :, :, ch] = test_patch[0, :, :, ch] / test_std[ch]


            predicted_patch = model.predict(x = test_patch, batch_size = 1)

            unthresholded_patch = np.copy(predicted_patch[0, :, :, 0])

            thresholded_patch = predicted_patch[0, :, :, 0]
            thresholded_patch[predicted_patch[0, :, :, 0] >= 0.5] = 1
            thresholded_patch[predicted_patch[0, :, :, 0] < 0.5] = 0


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


        thresholded_img[:, :, kth_slice] = thresholded_patch
        unthresholded_img[:, :, kth_slice] = unthresholded_patch

        kth_slice = kth_slice + 1


    thresholded_predicted_niftii = nib.Nifti1Image(thresholded_img, channel_niftii[0].affine, channel_niftii[0].header)
    nib.save(thresholded_predicted_niftii, os.path.join(output_path, thresholded_predicted_filename))

    unthresholded_predicted_niftii = nib.Nifti1Image(unthresholded_img, channel_niftii[0].affine, channel_niftii[0].header)
    unthresholded_predicted_niftii.set_data_dtype(np.float) # Set the datatype of the unthresholded image to float
    nib.save(unthresholded_predicted_niftii, os.path.join(output_path, unthresholded_predicted_filename))


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


    cmb_labels = label(thresholded_img, background = 0)
    cmb_region_props = regionprops(cmb_labels)

    sys.stdout.write("\n    Number of CMBs detected: " + str(len(cmb_region_props)) + "\n")
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
    output_path = "/path/to/output/dataset_binaryclass_SWI_QSM_T2.csv"
    datafile = open("/path/to/dataset.csv", "r")
   

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
            run_model(trainlist, write_path, output_path, testline, data_channels = 3, train_batch_size = 8, train_epochs = 30, normalize = True, no_rotations = 6, no_translation_axis1 = 8, no_translation_axis2 = 8)


        # Next testline
        k = k + 1


except:
    print("\n\nException occurred\n")
    print(sys.exc_info()[0])
    print("\n\n")

finally:
	# At the end, delete all the temporary generated data
    shutil.rmtree(write_path)

