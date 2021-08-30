


import os
import random
import sys
import time
import configparser

import nibabel as nib
import numpy as np
import scipy.misc
import scipy.ndimage


from skimage.exposure import rescale_intensity
from Multiclass_2D_Utils_vDec2020 import pad_volume, return_as_list_of_ints, return_as_boolean

class Multiclass_DataGen_2D_Writer_vDec2020():

    def __init__(self, dataline,
                 write_path = "/tmp/CNN_Model",
                 root_output_path = "/tmp/CNN_Model/Output",
                 axis = 2,
                 batch_size = 32,
                 slice_dim = (256, 256, 256),
                 n_channels = 4,
                 n_classes = 3,
                 shuffle = True,
                 pad = True,
                 normalize = True,
                 intensity_scaling = True,
                 labels_to_augment = None,
                 balance_bkgrnd_slices = True,
                 rotations = 6,
                 translations_axis1 = 6,
                 translations_axis2 = 6
                 ):

        self._slice_dim = slice_dim
        self._batch_size = batch_size

        self._n_channels = n_channels
        self._n_classes = n_classes
        self._shuffle = shuffle

        self._slice_axis = axis

        self._normalize_data = normalize
        self._intensity_scale = intensity_scaling

        self._balance_bkgrnd_slices = balance_bkgrnd_slices
        self._labels_to_augment = None
        if (labels_to_augment is not None) and (type(labels_to_augment) is list or type(labels_to_augment) is tuple):
            self._labels_to_augment = labels_to_augment.copy()


        self._rotations = rotations
        self._translations_axis1 = translations_axis1
        self._translations_axis2 = translations_axis2



        # The input variable dataline contains the volume filenames in the order:
        # ptid,mask_file,vol1,vol2,vol3,...
        # separated by comma

        # So split the line by comma
        iline = (dataline.strip("\n")).split(",")
        subject_id = iline[0]

        # Check to make sure that the number of channels in dataline is the same as specified.
        assert (len(iline) == self._n_channels + 2), "ERROR:\n  Specified number of channels are not the same as the volumes specified in the input file."


        # Create a subject-specific output directory
        if (os.path.exists(root_output_path) != True):
            os.mkdir(root_output_path)


        # Create the parent write_path if it does not already exist
        write_path = os.path.join(write_path, subject_id)
        if (os.path.exists(write_path) == False):
            os.makedirs(write_path)




        self._write_path_list = list(os.path.join(write_path, "vol_" + str(i)) for i in range(self._n_channels + 1))

        # Create the subdirectories (self._write_path_list) in the parent write_path
        for i in range(len(self._write_path_list)):
            if (os.path.exists(self._write_path_list[i]) == False):
                os.mkdir(self._write_path_list[i])


        # For each subject (line in csv),
        # put all the volumes into a list, where the first element in the list is the segmentation/groundtruth file,
        # followed by the other volumes
        ivols = list()
        for j in range(1, len(iline)):
            ivols.append(iline[j])

            if (j == 1):
                print("  (Groundtruth) vol_0=" + iline[j])
            else:
                print("                vol_" + str(j - 1) + "=" + iline[j])




        # Write out the slices of all the volumes for each subject
        print("  Working on subject " + subject_id + "")
        self.__write_to_file2__(id = subject_id, volumes_list = ivols, write_path_list = self._write_path_list, padding = pad)



        # Make a list of lists for all the volumes that were sliced and written to write_path's subdirectories
        self._list_vol_slices = list(None for i in range(self._n_channels + 1)) # Empty list for all the volumes
        k = 0
        for paths in self._write_path_list:
            self._list_vol_slices[k] = os.listdir(paths) # list of files in each write_path_list

            # Add the full path for each file in the lists
            for a in range(len(self._list_vol_slices[k])):
                self._list_vol_slices[k][a] = os.path.join(paths, self._list_vol_slices[k][a])

            # Make sure that all lists are sorted.
            self._list_vol_slices[k].sort()

            k = k + 1

        # Total number of SWI slices is considered as the reference
        self._total_slices = len(self._list_vol_slices[1])

        # Sanity check, make sure that the total slices in each write_path subdirectory is the same
        for i in range(len(self._list_vol_slices)):
            if (len(self._list_vol_slices[i]) != self._total_slices):
                sys.stderr.write("\n\nERROR\nTotal number of slices not same for volumes\n\n")
                break









    def __save_img__(self, filename_without_extension, X):
        '''

        :param filename_without_extension: The filename without the extension. Extension added in this function
        :param X: Array to be saved
        :return: None
        '''

        # # Save using scipy
        # scipy.misc.imsave(filename_without_extension + ".tiff", X)

        # Save using Numpy
        np.savez_compressed(filename_without_extension + ".npz", a = X)





    def __slice_has_labels__(self, slice, labels):
        slice_flattened = slice.astype(np.int).flatten().tolist()
        res = any(elem in labels for elem in slice_flattened)
        return res






    def __write_to_file2__(self, id, volumes_list, write_path_list, padding = False):
        start_time = time.time()


        base_name = list("" for i in range(len(volumes_list)))
        loaded_vols = list(None for i in range(len(volumes_list)))

        subject_mean = list(0 for i in range(len(volumes_list)))
        subject_std = list(0 for i in range(len(volumes_list)))

        # Load all of a single subject's volumes. The first volume is CMB_Segmentation, the second is SWI, the thrid is QSM, etc
        for i in range(len(volumes_list)):
            loaded_vols[i] = nib.load(volumes_list[i]).get_data() # Load the niftii objects into loaded_vols[i]
            base_name[i] = id + "_vol" + str(i)



        # Now, do some of the remaining preprocessing
        for i in range(len(loaded_vols)):

            # Do padding if necessary
            if (padding == True):
                loaded_vols[i], _notused = pad_volume(loaded_vols[i], newshape = self._slice_dim)  # Replace the each loaded_vols[i] with the padded image array



            # Rescale image intensity, ignoring the first volume which is the mask
            if (self._intensity_scale == True):
                if (i != 0):  # loaded_vols[0] is the mask image.
                    loaded_vols[i] = rescale_intensity(loaded_vols[i], out_range = (0, 255))

            # Calculate the subject-specific mean and std dev first,
            # Here, we are computing the mean and std dev of the mask (groundtruth file)
            # but later on, when augmenting and writing, slices from the mask are NOT normalized.
            if (self._normalize_data == True):
                subject_mean[i] = np.mean(loaded_vols[i])
                subject_std[i] = np.std(loaded_vols[i])

                print("  vol_" + str(i) + " mean = " + "{:+.4f}".format(subject_mean[i]) + ", std dev = " + "{:+.4f}".format(subject_std[i]))





        # For counting the number of blank/non-blank slices
        no_of_nonblank_slices = 0

        # For counting the number of CMB slices
        no_of_cmb_slices = 0

        # For counting the number of augmented CMB slices
        augmented_cmbs = 0

        # For counting the number of augmented regular slices
        augmented_regular = 0


        # For all coronal (axis = 0) slices
        if (self._slice_axis == 0):

            for i in range(loaded_vols[1].shape[self._slice_axis]):
                swi_slice = loaded_vols[1][i, :, :]


                if (np.sum(swi_slice) > 0):
                    no_of_nonblank_slices = no_of_nonblank_slices + 1

                    for j in range(len(volumes_list)):
                        self.__save_img__(os.path.join(write_path_list[j], base_name[j] + "_Axis" + str(self._slice_axis) + "_" + str(i)), loaded_vols[j][i, :, :])


        # For all sagittal (axis = 1) slices
        elif (self._slice_axis == 1):

            for i in range(loaded_vols[1].shape[self._slice_axis]):
                swi_slice = loaded_vols[1][:, i, :]

                if (np.sum(swi_slice) > 0):
                    no_of_nonblank_slices = no_of_nonblank_slices + 1

                    for j in range(len(volumes_list)):
                        self.__save_img__(os.path.join(write_path_list[j], base_name[j] + "_Axis" + str(self._slice_axis) + "_" + str(i)), loaded_vols[j][:, i, :])


        # For all axial (axis = 2) slices
        elif (self._slice_axis == 2):

            for i in range(loaded_vols[1].shape[self._slice_axis]):
                swi_slice = loaded_vols[1][:, :, i] # Take the ith swi slice and check for blank
                cmb_slice = loaded_vols[0][:, :, i] # Take the ith CMB slice and check if it contains a segmentation


                if (np.sum(swi_slice) > 0):
                    no_of_nonblank_slices = no_of_nonblank_slices + 1

                    for j in range(len(volumes_list)):
                        slice_to_save = loaded_vols[j][:, :, i]

                        # Normalize the slice from each volume with their respective mean and std dev
                        # and make sure that the mask image (j == 0) is not normalized
                        if ((self._normalize_data == True) and (j != 0)):
                            slice_to_save = slice_to_save - subject_mean[j]
                            slice_to_save = slice_to_save / subject_std[j]

                        self.__save_img__(os.path.join(write_path_list[j], base_name[j] + "_Axis" + str(self._slice_axis) + "_" + str(i)), slice_to_save)

                    if (np.sum(cmb_slice) > 0):
                        no_of_cmb_slices = no_of_cmb_slices + 1



            if (self._labels_to_augment is not None and len(self._labels_to_augment) > 0):
                print("  Agumenting slices with labels of interest...")

                for ith_slice in range(loaded_vols[1].shape[self._slice_axis]):

                    cmb_slice = loaded_vols[0][:, :, ith_slice] # Take the ith CMB slice and check for presence of CMBs

                    if (self.__slice_has_labels__(cmb_slice, self._labels_to_augment) == True):

                        cmb_slices_list = list()
                        cmb_basenames_list = list()

                        for j1 in range(len(loaded_vols)):
                            cmb_slice_to_augment_save = loaded_vols[j1][:, :, ith_slice]

                            # Normalize the cmb slice from each volume with their respective mean and std dev
                            # and make sure that the mask image (j1 == 0) is not normalized
                            if ((self._normalize_data == True) and (j1 != 0)):
                                cmb_slice_to_augment_save = cmb_slice_to_augment_save - subject_mean[j1]
                                cmb_slice_to_augment_save = cmb_slice_to_augment_save / subject_std[j1]

                            cmb_slices_list.append(cmb_slice_to_augment_save)
                            cmb_basenames_list.append(os.path.join(write_path_list[j1], base_name[j1] + "_Axis" + str(self._slice_axis) + "_" + str(ith_slice) + "_AugCMB"))

                        augs = self.__do_augmentation__(cmb_slices_list, ith_slice, cmb_basenames_list)
                        augmented_cmbs = augmented_cmbs + augs



            if (self._balance_bkgrnd_slices == True):
                print("  Balancing background slices...")

                # Augment normal/regular slices also to balance with labelled slices
                while ((augmented_regular + no_of_nonblank_slices - no_of_cmb_slices) < augmented_cmbs):
                    # Augmentation of regular slices, based on non-empty SWI slice
                    # regular_slice = loaded_vols[1][:, :, ith_slice]


                    random_i = random.randint(0, loaded_vols[1].shape[self._slice_axis] - 1) # Select a random slice
                    regular_slice = loaded_vols[1][:, :, random_i]

                    # Check to make sure that the slice is non-blank and contains no CMBs
                    cmb_slice1 = loaded_vols[0][:, :, ith_slice]  # Take the ith CMB slice and check for presence of CMBs
                    if (np.sum(regular_slice) > 0) and (self.__slice_has_labels__(cmb_slice1, self._labels_to_augment) == False):
                        regular_slices_list = list()
                        regular_basenames_list = list()

                        for j2 in range(len(loaded_vols)):
                            regular_slice_to_augment_save = loaded_vols[j2][:, :, random_i]

                            # Normalize the regular slice from each volume with their respective mean and std dev
                            # and make sure that the mask image (j2 == 0) is not normalized
                            if ((self._normalize_data == True) and (j2 != 0)):
                                regular_slice_to_augment_save = regular_slice_to_augment_save - subject_mean[j2]
                                regular_slice_to_augment_save = regular_slice_to_augment_save / subject_std[j2]

                            regular_slices_list.append(regular_slice_to_augment_save)
                            regular_basenames_list.append(os.path.join(write_path_list[j2], base_name[j2] + "_Axis" + str(self._slice_axis) + "_" + str(random_i) + "_AugReg"))

                        raugs = self.__do_augmentation__(regular_slices_list, random_i, regular_basenames_list)
                        augmented_regular = augmented_regular + raugs



        else:
            sys.stderr.write("  Invalid axis provided when writing 2D images for file series: " + volumes_list[1])




        print("\n  Blank/discarded slices=" + str(loaded_vols[1].shape[self._slice_axis] - no_of_nonblank_slices), end = "")
        print(", Background slices=" + str(no_of_nonblank_slices - no_of_cmb_slices), end = "")
        print(", Slices with labels of interest=" + str(no_of_cmb_slices), end = "")
        print(", Augmented labels of interest slices=" + str(augmented_cmbs), end = "")
        print(", Augmented background slices=" + str(augmented_regular), end = "")
        print(", Time taken=" + "{:.2f}".format(time.time() - start_time) + " seconds" + "\n\n")



    def __do_augmentation__(self, slice_list, sliceNo, basenames_list):

        num_augmentations = 0
        flippedup_slice_list = list(None for q in range(len(slice_list)))
        flippedlr_slice_list = list(None for q in range(len(slice_list)))

        # Flips
        # Flip the slices, and then keep for later augmentations on flipped slices
        num_augmentations = num_augmentations + 1
        for i in range(len(slice_list)):
            flippedup_slice_list[i] = self.__augment_flip_up_down(slice_list[i], aug_write_path = os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)))

        num_augmentations = num_augmentations + 1
        for i in range(len(slice_list)):
            flippedlr_slice_list[i] = self.__augment_flip_left_right(slice_list[i], aug_write_path = os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)))



        # Rotations
        degrees = random.sample(range(1, 179), self._rotations)  # Generate some positive integers

        num_augmentations = num_augmentations + 2 * len(degrees)

        for d in degrees:
            for i in range (len(slice_list)):
                # Do rotations for positive degree, and negative degree
                if (i == 0):
                    self.__augment_rotation__(slice = slice_list[i], degree = d, aug_write_path = os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)
                    self.__augment_rotation__(slice = slice_list[i], degree = -d, aug_write_path = os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)
                else:
                    self.__augment_rotation__(slice = slice_list[i], degree = d, aug_write_path = os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)
                    self.__augment_rotation__(slice = slice_list[i], degree = -d, aug_write_path = os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)


        # Do the same rotations on the up-down and left-right flipped slices
        num_augmentations = num_augmentations + 2 * len(degrees) * 2

        for d in degrees:
            for i in range(len(flippedup_slice_list)):
                # Do rotations for positive degree, and negative degree
                if (i == 0):
                    self.__augment_rotation__(slice = flippedup_slice_list[i], degree = d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)
                    self.__augment_rotation__(slice = flippedup_slice_list[i], degree = -d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)

                    self.__augment_rotation__(slice = flippedlr_slice_list[i], degree = d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)
                    self.__augment_rotation__(slice = flippedlr_slice_list[i], degree = -d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)

                else:
                    self.__augment_rotation__(slice = flippedup_slice_list[i], degree = d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)
                    self.__augment_rotation__(slice = flippedup_slice_list[i], degree = -d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)

                    self.__augment_rotation__(slice = flippedlr_slice_list[i], degree = d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)
                    self.__augment_rotation__(slice = flippedlr_slice_list[i], degree = -d, aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)




        # Translations
        translation_axis1 = random.sample(range(-45, 45), self._translations_axis1)  # Generate some random numbers for translations
        translation_axis2 = random.sample(range(-45, 45), self._translations_axis2)

        num_augmentations = num_augmentations + (len(translation_axis2) * len(translation_axis1))

        for tr1 in range(len(translation_axis1)):
            for tr2 in range(len(translation_axis2)):
                for i in range(len(slice_list)):
                    if (i == 0):
                        self.__augment_translation__(slice = slice_list[i], translate_axis = (translation_axis1[tr1], translation_axis2[tr2]), aug_write_path=os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)
                    else:
                        self.__augment_translation__(slice = slice_list[i], translate_axis = (translation_axis1[tr1], translation_axis2[tr2]), aug_write_path = os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)


        # Do the same translations for up-down and left-right flipped slices
        num_augmentations = num_augmentations + (len(translation_axis2) * len(translation_axis1)) * 2

        for tr1 in range(len(translation_axis1)):
            for tr2 in range(len(translation_axis2)):
                for i in range(len(flippedup_slice_list)):
                    if (i == 0):
                        self.__augment_translation__(slice = flippedup_slice_list[i], translate_axis = (translation_axis1[tr1], translation_axis2[tr2]), aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)
                        self.__augment_translation__(slice = flippedup_slice_list[i], translate_axis = (translation_axis1[tr1], translation_axis2[tr2]), aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = True)
                    else:
                        self.__augment_translation__(slice = flippedup_slice_list[i], translate_axis = (translation_axis1[tr1], translation_axis2[tr2]), aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)
                        self.__augment_translation__(slice = flippedup_slice_list[i], translate_axis = (translation_axis1[tr1], translation_axis2[tr2]), aug_write_path = os.path.join(basenames_list[i] + "_Axis_Flipped" + str(self._slice_axis) + "_" + str(sliceNo)), is_gt = False)

        return num_augmentations




    def __augment_translation__(self, slice, translate_axis = (0, 0), aug_write_path = "/tmp/aug", is_gt = False):
        # Do the translation
        if (is_gt == True):
            translated_slice = scipy.ndimage.shift(slice, shift = translate_axis, mode = "nearest", order = 0)
        else:
            translated_slice = scipy.ndimage.shift(slice, shift = translate_axis, mode = "nearest")#, order = 0)

        # Write to file
        self.__save_img__(aug_write_path + "_Tr" + str(translate_axis[0]) + "_" + str(translate_axis[1]), translated_slice)



    def __augment_rotation__(self, slice, degree = 0, aug_write_path = "/tmp/aug", is_gt = False):
        # Do the rotation
        if (is_gt == True):
            rotated_slice = scipy.ndimage.rotate(slice, degree, reshape = False, mode = "nearest", order = 0)

        else:
            rotated_slice = scipy.ndimage.rotate(slice, degree, reshape = False, mode = "nearest") #, order = 0)

        # Write to file
        self.__save_img__(aug_write_path + "_Rot" + str(degree), rotated_slice)



    def __augment_flip_up_down(self, slice, aug_write_path = "/tmp/aug"):
        # Do the flip
        flipped_ud = np.flipud(slice)

        # Write to file
        self.__save_img__(aug_write_path + "_FlipUD", flipped_ud)

        # Return the flipped slice
        return flipped_ud



    def __augment_flip_left_right(self, slice, aug_write_path = "/tmp/aug"):
        # Do the flip
        flipped_lr = np.fliplr(slice)

        # Write to file
        self.__save_img__(aug_write_path + "_FlipLR", flipped_lr)

        # Return the flipped slice
        return flipped_lr






def main():

    if (len(sys.argv) != 3):
        print("\nERROR")
        print("  Incorrect number of parameters.")
        print("  Usage is ")
        print("    <python_executable> Multiclass_DataGen_2D_Writer_vDec2020.py </path/to/config_file.ini> </path/to/datafile.csv>\n")

    else:
        print("Working")

        config_file = sys.argv[1]
        datafile = sys.argv[2]

        print("Config file is:\n  " + config_file)
        print("Data file is:\n  " + datafile)


        cfg = configparser.ConfigParser()
        cfg.read(config_file)
        sections = cfg.sections()

        write_path = cfg["PATHS"]["write_path"]
        root_output_path = cfg["PATHS"]["root_output_path"]
        slice_axis = int(cfg["MODEL_PARAMETERS"]["slice_axis"])
        data_channels = int(cfg["MODEL_PARAMETERS"]["data_channels"])
        output_classes = int(cfg["MODEL_PARAMETERS"]["output_classes"])
        slice_dim = return_as_list_of_ints(cfg["MODEL_PARAMETERS"]["input_shape"])
        train_batch_size = int(cfg["MODEL_PARAMETERS"]["train_batch_size"])
        shuffle = return_as_boolean(cfg["DATA_AUGMENTATION"]["shuffle"])
        pad = return_as_boolean(cfg["DATA_AUGMENTATION"]["pad_image"])
        unit_mean_variance_normalize = return_as_boolean(cfg["DATA_AUGMENTATION"]["unit_mean_variance_normalize"])
        scaling = return_as_boolean(cfg["DATA_AUGMENTATION"]["scaling"])
        augment_labels = return_as_list_of_ints(cfg["DATA_AUGMENTATION"]["augment_labels"])
        balance_bkgrnd_slices = return_as_boolean(cfg["DATA_AUGMENTATION"]["balance_bkgrnd_slices"])
        no_rotations = int(cfg["DATA_AUGMENTATION"]["no_rotations"])
        no_tralations_axis1 = int(cfg["DATA_AUGMENTATION"]["no_translations_axis1"])
        no_tralations_axis2 = int(cfg["DATA_AUGMENTATION"]["no_translations_axis2"])

        print("Data generation parameters are: ")
        print("  write_path=", write_path)
        print("  slice_axis=", slice_axis)
        print("  data_channels=", data_channels)
        print("  output_classes=", output_classes)
        print("  slice_dim=", slice_dim)
        print("  train_batch_size=", str(train_batch_size))
        print("  shuffle=", shuffle)
        print("  pad_image=", pad)
        print("  unit_mean_variance_normalize=", unit_mean_variance_normalize)
        print("  scaling=", scaling)
        print("  augment_labels=", augment_labels)
        print("  balance_bkgrnd_slices=", balance_bkgrnd_slices)
        print("  no_rotations=", no_rotations)
        print("  no_tralations_axis1=", no_tralations_axis1)
        print("  no_tralations_axis2=", no_tralations_axis2)
        print("")

        df = open(datafile)

        for dataline in df:
            dataline = dataline.strip("\n").strip("\r")

            mdg = Multiclass_DataGen_2D_Writer_vDec2020(dataline,
                                                        write_path = write_path,
                                                        root_output_path = root_output_path,
                                                        axis = slice_axis,
                                                        batch_size = train_batch_size,
                                                        slice_dim = slice_dim,
                                                        n_channels = data_channels,
                                                        n_classes = output_classes,
                                                        shuffle = shuffle,
                                                        pad = pad,
                                                        normalize = unit_mean_variance_normalize,
                                                        intensity_scaling = scaling,
                                                        labels_to_augment = augment_labels,
                                                        balance_bkgrnd_slices = balance_bkgrnd_slices,
                                                        rotations = no_rotations,
                                                        translations_axis1 = no_tralations_axis1,
                                                        translations_axis2 = no_tralations_axis2
                                                        )

        print("\nData writing completed.\n================================================================")


if (__name__ == "__main__"):
    main()
