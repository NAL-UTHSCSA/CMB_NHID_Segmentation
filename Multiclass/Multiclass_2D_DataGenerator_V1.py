
import os
import random
import sys
import time

import nibabel as nib
import numpy as np
import scipy.misc
import scipy.ndimage
from tensorflow.python import keras
from skimage.exposure import rescale_intensity


class Multiclass_2D_DataGenerator_V1(keras.utils.Sequence):

    def __init__(self, vol_filelist,
                 write_path = "/tmp/CNN_Model",
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
                 rotations = 6,
                 translations_axis1 = 6,
                 translations_axis2 = 6,
                 logfile = "/tmp/CNN_Model/DataGen.log"):
        '''
        This is the initializer. This takes as input vol_filelist, which contains a list of volume paths for mask and MRI.
        The files in vol_filelist are arranged in comma-separated-values (CSV) format in the following manner:
            patient_id,maskpath,swi_path,qsm_path,...
        @:param vol_filelist

        '''

        self._slice_dim = slice_dim
        self._batch_size = batch_size

        self._n_channels = n_channels
        self._n_classes = n_classes
        self._shuffle = shuffle

        self._slice_axis = axis

        self._normalize_data = normalize
        self._intensity_scale = intensity_scaling

        self._labels_to_augment = None
        if (labels_to_augment is not None) and (type(labels_to_augment) is list or type(labels_to_augment) is tuple):
            self._labels_to_augment = labels_to_augment.copy()


        self._rotations = rotations
        self._translations_axis1 = translations_axis1
        self._translations_axis2 = translations_axis2

        # Setup the log file
        self._logfile = logfile

        # Create the parent write_path if it does not already exist
        if (os.path.exists(write_path) == False):
            os.mkdir(write_path)
            # self.logger("  Creating new folder in " + write_path + "\n")

        self._logger = open(self._logfile, "a")

        self._write_path_list = list(os.path.join(write_path, "vol_" + str(i)) for i in range(self._n_channels + 1))

        # Create the subdirectories (self._write_path_list) in the parent write_path
        for i in range(len(self._write_path_list)):
            if (os.path.exists(self._write_path_list[i]) == False):
                os.mkdir(self._write_path_list[i])



        # Each line in the vol_filelist contains the volume filenames in the order:
        # ptid,mask_file,swi_file,qsm_file,t1_file,t2_file, ...
        # separated by comma
        for lines in vol_filelist:
            iline = (lines.strip("\n")).split(",")

            # For each subject (line in csv),
            # put all the volumes into a list, where the first element in the list is the mask,
            # followed by swi, qsm, ...
            ivols = list()
            for j in range(1, len(iline)):
                ivols.append(iline[j])

            # Write out the slices of all the volumes for each subject
            self.logger("  Working on subject " + iline[0] + "")
            self.__write_to_file2__(ivols, write_path_list = self._write_path_list, padding = pad)



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

        self.logger("  Total files generated=" + str(self._total_slices) + "\n\n")

        self.on_epoch_end()

        self._logger.close()




    def logger(self, strdata):
        print(strdata, end = "")
        self._logger.write(strdata)




    @staticmethod
    def pad_volume(vol, newshape=(256, 256, 256), slice_axis = 2):
        '''
        This function pads the input volume according to the provided newshape. This function will not pad the axis
        along which slicing will take place.
        For example, if the slicing axis is 2, then this function will pad only axis 0 and 1.

        :param vol: volume whose shape is to be checked/padded
        :param newshape: (int tuple) The new shape of the volume
        :return: a volume whose shape conforms to a specified size
        '''

        volshape = vol.shape

        # Pad the volume using numpy.pad() and constant padding of 0
        # if volshape[0] < newshape[0] and slice_axis != 0:
        #     padding = (int)(np.floor((newshape[0] - volshape[0]) / 2))
        #     vol = np.pad(vol, ((padding, padding), (0, 0), (0, 0)), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        #
        # if volshape[1] < newshape[1] and slice_axis != 1:
        #     padding = (int)(np.floor((newshape[1] - volshape[1]) / 2))
        #     vol = np.pad(vol, ((0, 0), (padding, padding), (0, 0)), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        #
        # if volshape[2] < newshape[2] and slice_axis != 2:
        #     padding = (int)(np.floor((newshape[2] - volshape[2]) / 2))
        #     vol = np.pad(vol, ((0, 0), (0, 0), (padding, padding)), "constant", constant_values=((0, 0), (0, 0), (0, 0)))

        # Pad the volume using numpy.pad() with padding of whatever the edge value is
        if volshape[0] < newshape[0] and slice_axis != 0:
            padding = (int)(np.floor((newshape[0] - volshape[0]) / 2))
            vol = np.pad(vol, ((padding, padding), (0, 0), (0, 0)), "edge")

        if volshape[1] < newshape[1] and slice_axis != 1:
            padding = (int)(np.floor((newshape[1] - volshape[1]) / 2))
            vol = np.pad(vol, ((0, 0), (padding, padding), (0, 0)), "edge")

        if volshape[2] < newshape[2] and slice_axis != 2:
            padding = (int)(np.floor((newshape[2] - volshape[2]) / 2))
            vol = np.pad(vol, ((0, 0), (0, 0), (padding, padding)), "edge")

        return vol


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



    def __load_img__(self, filename):

        # Load using scipy
        # X = scipy.misc.imread(filename)
        # return X


        # Load using Numpy
        X = np.load(filename)
        return X['a']


    def __slice_has_labels__(self, slice, labels):
        slice_flattened = slice.astype(np.int).flatten().tolist()
        res = any(elem in labels for elem in slice_flattened)
        return res


    def __write_to_file2__(self, volumes_list, write_path_list, padding = False):
        start_time = time.time()


        base_name = list("" for i in range(len(volumes_list)))
        loaded_vols = list(None for i in range(len(volumes_list)))

        subject_mean = list(0 for i in range(len(volumes_list)))
        subject_std = list(0 for i in range(len(volumes_list)))

        # Load all of a single subject's volumes. The first volume is CMB_Segmentation, the second is SWI, the thrid is QSM, etc
        for i in range(len(volumes_list)):
            loaded_vols[i] = nib.load(volumes_list[i]) # Load the niftii objects into loaded_vols[i]

            base_name[i] = os.path.basename(loaded_vols[i].file_map['image'].filename)
            base_name[i] = base_name[i].replace(".nii", "")
            base_name[i] = base_name[i].replace(".gz", "")

            # Do padding if necessary
            if (padding == True):
                # loaded_vols[i] = DataGenerator_V7.pad_volume(loaded_vols[i].get_fdata().astype(loaded_vols[i].get_data_dtype()), newshape = self._slice_dim) # Replace the each loaded_vols[i] with the padded image array
                loaded_vols[i] = Multiclass_2D_DataGenerator_V1.pad_volume(loaded_vols[i].get_data(), newshape = self._slice_dim)  # Replace the each loaded_vols[i] with the padded image array

                # if (i == 0):
                #     loaded_vols[i] = Multiclass_2D_DataGenerator_V1.pad_volume(loaded_vols[i].get_data().astype(np.uint8), newshape=self._slice_dim)  # Replace the each loaded_vols[i] with the padded image array
                # else:
                #     loaded_vols[i] = Multiclass_2D_DataGenerator_V1.pad_volume(loaded_vols[i].get_data(), newshape = self._slice_dim)  # Replace the each loaded_vols[i] with the padded image array
            else:
                # loaded_vols[i] = loaded_vols[i].get_fdata().astype(loaded_vols[i].get_data_dtype())  # Replace the each loaded_vols[i] with the unpadded image array
                # loaded_vols[i] = loaded_vols[i].get_fdata()
                loaded_vols[i] = loaded_vols[i].get_data()



            # Rescale image intensity, ignoring the first volume which is the mask
            if (self._intensity_scale == True):
                if (i != 0):  # loaded_vols[0] is the mask image.
                    loaded_vols[i] = rescale_intensity(loaded_vols[i], out_range = (0, 255))

            # Calculate the subject-specific mean and std dev first
            if (self._normalize_data == True):
                subject_mean[i] = np.mean(loaded_vols[i])
                subject_std[i] = np.std(loaded_vols[i])

                self.logger("\n    vol_" + str(i) + " mean = " + "{:+.4f}".format(subject_mean[i]) + ", std dev = " + "{:+.4f}".format(subject_std[i]))





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
            sys.stderr.write("    Invalid axis provided when writing 2D images for file series: " + volumes_list[1])




        self.logger("\n    Empty/discarded slices=" + str(loaded_vols[1].shape[self._slice_axis] - no_of_nonblank_slices))
        self.logger(", Non-CMB slices=" + str(no_of_nonblank_slices - no_of_cmb_slices))
        self.logger(", CMB slices=" + str(no_of_cmb_slices))
        self.logger(", Augmented CMB slices=" + str(augmented_cmbs))
        self.logger(", Augmented Regular slices=" + str(augmented_regular))
        self.logger(", Time taken=" + "{:.2f}".format(time.time() - start_time) + " seconds" + "\n\n")



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



    def on_epoch_end(self):
        self._indexes = np.arange(self._total_slices)

        if self._shuffle == True:
            np.random.shuffle(self._indexes)



    def __len__(self):
        return int(np.floor(self._total_slices / self._batch_size))



    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self._indexes[index * self._batch_size:(index + 1) * self._batch_size]

        # Find list of vols and corresponding masks
        temp_list = list(list() for i in range(self._n_channels + 1))

        for i in range(self._n_channels + 1):
            temp_list[i] = [self._list_vol_slices[i][k] for k in indexes]

        #
        # list_swis_temp = [self._list_swis[k] for k in indexes]
        # list_masks_temp = [self._list_masks[k] for k in indexes]
        # list_qsms_temp = [self._list_qsms[k] for k in indexes]
        # list_t1s_temp = [self._list_t1s[k] for k in indexes]
        # list_t2s_temp = [self._list_t2s[k] for k in indexes]

        # for j in range(32):
        #     for i in range(len(temp_list)):
        #         print(temp_list[i][j], sep=", ")
        #     print("\n")

        # Generate data
        X, y = self.__data_generation(temp_list)

        return X, y



    def __data_generation(self, temp_list):

        X = np.zeros((self._batch_size, self._slice_dim[0], self._slice_dim[1], self._n_channels))
        y = np.zeros((self._batch_size, self._slice_dim[0], self._slice_dim[1], self._n_classes))

        # Generate the data
        for i in range(self._batch_size):

            #
            # swi_img = self.__load_img__(temp_list[1][i])
            # qsm_img = self.__load_img__(temp_list[2][i])

            # y[i, :, :, 0] = msk_img
            # X[i, :, :, 0] = swi_img
            # X[i, :, :, 1] = qsm_img

            # y[i, :, :, 0] = self.__load_img__(temp_list[0][i])

            msk_img = self.__load_img__(temp_list[0][i])

            # Binarize, excluding the background
            for j in range(1, self._n_classes):
                jclass_img = np.copy(msk_img)

                jclass_img[msk_img != j] = 0   # Set all other labels to 0
                jclass_img = jclass_img / j    # Binarize the image

                # temp_jclass.append(jclass)
                y[i, :, :, j] = jclass_img

            # # When also segmenting background as a class
            zero_class = np.copy(msk_img)
            zero_class = zero_class + 1
            zero_class[zero_class > 1] = 0
            y[i, :, :, 0] = zero_class


            for j in range(self._n_channels):
                X[i, :, :, j] = self.__load_img__(temp_list[j + 1][i])


            #
            # plt.figure(figsize = (8, 8))
            # plt.subplot(2, 3, 1)
            # plt.imshow(X[i, :, :, 0], cmap = "gray")
            # plt.title("SWI")
            #
            # plt.subplot(2, 3, 2)
            # plt.imshow(X[i, :, :, 1], cmap = "gray")
            # plt.title("QSM")
            #
            # plt.subplot(2, 3, 3)
            # plt.imshow(X[i, :, :, 2], cmap = "gray")
            # plt.title("T2")
            #
            # plt.subplot(2, 3, 4)
            # plt.imshow(msk_img, cmap = "gray")
            # plt.title("Groundtruth")
            #
            # plt.subplot(2, 3, 5)
            # plt.imshow(y[i, :, :, 0], cmap = "gray")
            # plt.title("CMB")
            #
            # plt.subplot(2, 3, 6)
            # plt.imshow(y[i, :, :, 1], cmap = "gray")
            # plt.title("Iron Deposits")
            #
            # plt.show()
            # plt.close()

        return X, y



# #
# # For testing
# from matplotlib import pyplot as plt
#
# datafile = open("/media/rashidt/Data/MESA_CMB/Scripts/Multiclass_2D_Segmentation_V1/AAA3.csv", "r")
#
# vols = list()
# for lines in datafile:
#     iline = lines.strip("\n")
#     vols.append(iline)
# bs = 128
# sd = (256, 256, 256)
#
# import shutil
# write_path = "/tmp/CNN_Model"
# if (os.path.exists(write_path)):
#     shutil.rmtree(write_path)
#     os.mkdir(write_path)
#
# dg = Multiclass_2D_DataGenerator_V1(
#         vols,
#         write_path = write_path,
#         axis = 2,
#         batch_size = bs,
#         slice_dim = (256, 256, 256),
#         n_channels = 3,
#         n_classes = 4,
#         shuffle = True,
#         pad = True,
#         normalize = True,
#         intensity_scaling = False,
#         labels_to_augment = [2, 3],
#         rotations = 3,
#         translations_axis1 = 3,
#         translations_axis2 = 3,
#         logfile = os.path.join(write_path, "Train_DataGen.log"))
# #
# for item in range(dg.__len__()):
#     x, y = dg.__getitem__(item)
#     print("Batch:", item)
#     #
#     for i in range(x.shape[0]):
#         plt.figure(figsize = (16, 8))
#
#         plt.subplot(2, 4, 1)
#         plt.imshow(x[i, :, :, 0], cmap = "gray")
#         plt.title("SWI")
#
#         plt.subplot(2, 4, 2)
#         plt.imshow(x[i, :, :, 1], cmap = "gray")
#         plt.title("QSM")
#
#         plt.subplot(2, 4, 3)
#         plt.imshow(x[i, :, :, 2], cmap = "gray")
#         plt.title("T2")
#
#         plt.subplot(2, 4, 5)
#         plt.imshow(y[i, :, :, 0], cmap = "gray")
#         plt.title("Background")
#
#         plt.subplot(2, 4, 6)
#         plt.imshow(y[i, :, :, 1], cmap = "gray")
#         plt.title("NonROI Brain")
#
#         plt.subplot(2, 4, 7)
#         plt.imshow(y[i, :, :, 2], cmap = "gray")
#         plt.title("CMB")
#
#         plt.subplot(2, 4, 8)
#         plt.imshow(y[i, :, :, 3], cmap = "gray")
#         plt.title("Iron Deposits")
#
#
#         # plt.subplot(2, 3, 4)
#         # plt.imshow(x1[i, :, :, 0], cmap="gray")
#         # plt.title("SWI Z-Scaled")
#         #
#         # plt.subplot(2, 3, 5)
#         # plt.imshow(x1[i, :, :, 1], cmap="gray")
#         # plt.title("QSM Z-Scaled")
#         #
#         # plt.subplot(2, 3, 6)
#         # plt.imshow(y[i, :, :, 0], cmap="gray")
#         # plt.title("Mask")
#
#         plt.show()
#         plt.close()
#     # #
