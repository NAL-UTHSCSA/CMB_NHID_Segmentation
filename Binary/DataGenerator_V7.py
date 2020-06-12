
import os
import numpy as np
import nibabel as nib
import sys
import random
import gc
import time

import scipy.misc
import scipy.ndimage
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


from tensorflow.python import keras


class Z_Scored_Scaler():
    '''
    This is a Z-score scaler for a whole group.
    First initialize this object, then add volumes with the add_volume() method, and then
    call calculate_z_score_params() function to compute the mean and std of the whole group
    '''


    def __init__(self,
                 scaler_name = "default",
                 slice_dim = (256, 256, 256),
                 slice_axis = 2):
        self._mean = 0.0
        self._std = 0.0
        self._name = scaler_name

        self._slice_dim = slice_dim
        self._slice_axis = slice_axis

        self._volume_list = list()



    def add_volume(self, vol):
        self._volume_list.append(vol.strip("\n"))


    def get_number_of_volumes(self):
        return len(self._volume_list)



    def calculate_z_score_params(self):
        sys.stdout.write("  Computing Z-scores for " + str(len(self._volume_list)) + " " + self._name + " files\n")

        total_slices = 0

        # First load all volumes into list
        for i1 in range(len(self._volume_list)):
            self._volume_list[i1] = nib.load(self._volume_list[i1])
            self._volume_list[i1] = DataGenerator_V7.pad_volume(self._volume_list[i1].get_fdata().astype(self._volume_list[i1].get_data_dtype()), newshape = (256, 256, 256), slice_axis = self._slice_axis)

            total_slices = total_slices + self._volume_list[i1].shape[self._slice_axis]

        # Create a temporary array to hold all the voxels of all images
        voxels = np.zeros((self._slice_dim[0], self._slice_dim[1], total_slices))

        slice = 0
        # Load each slice from all volumes into voxels array
        for i2 in range(len(self._volume_list)):
            for j in range(self._volume_list[i2].shape[self._slice_axis]):
                voxels[:, :, slice] = self._volume_list[i2][:, :, j]
                slice = slice + 1

        # Compute the mean and standard deviation
        self._mean = voxels.mean()
        self._std = voxels.std()


        # Since the voxels array is very large and memory intensive, clear the memory allocation
        voxels = None
        gc.collect()

        sys.stdout.write("    Mean = " + str(self._mean) + ", Std Dev = " + str(self._std) + "\n\n")




class DataGenerator_V7(keras.utils.Sequence):

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
                 augment_cmbs = True,
                 augment_regular = True,
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

        self._augment_cmbs = augment_cmbs
        self._augment_regular = augment_regular

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
                loaded_vols[i] = DataGenerator_V7.pad_volume(loaded_vols[i].get_data(), newshape = self._slice_dim) # Replace the each loaded_vols[i] with the padded image array
            else:
                loaded_vols[i] = loaded_vols[i].get_data() #.astype(loaded_vols[i].get_data_dtype())  # Replace the each loaded_vols[i] with the unpadded image array


            if (self._normalize_data == True):
                # Calculate the subject-specific mean and std dev first
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




            for ith_slice in range(loaded_vols[1].shape[self._slice_axis]):
                # Augmentation based on CMB_Seg data
                cmb_slice = loaded_vols[0][:, :, ith_slice] # Take the ith CMB slice and check for presence of CMBs

                if (np.sum(cmb_slice) > 0):

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



            while ((augmented_regular + no_of_nonblank_slices - no_of_cmb_slices) < augmented_cmbs):
                # Augmentation of regular slices, based on non-empty SWI slice
                # regular_slice = loaded_vols[1][:, :, ith_slice]

                random_i = random.randint(0, loaded_vols[1].shape[self._slice_axis] - 1) # Select a random slice
                regular_slice = loaded_vols[1][:, :, random_i]

                # Check to make sure that the slice is non-blank and contains no CMBs
                if (np.sum(regular_slice) > 0) and (np.sum(loaded_vols[0][:, :, random_i]) == 0):
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

        # Rotations
        degrees = random.sample(range(1, 60), self._rotations)  # Generate some positive integers

        num_augmentations = num_augmentations + 2 * len(degrees)

        for d in degrees:
            for i in range (len(slice_list)):
                # Do rotations for positive degree, and negative degree
                self.__augment_rotation__(slice = slice_list[i], degree = d, aug_write_path=os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), sliceNo = i)
                self.__augment_rotation__(slice = slice_list[i], degree = -d, aug_write_path=os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), sliceNo = i)



        # self.logger("\n    Rotations used for " + os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)) + ": ")
        # for d in degrees:
        #     self.logger(str(d) + ",")
        #
        # self.logger("\n    Translations in axis1 for " + os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)) + ": ")
        # for tr1 in range(len(translation_axis1)):
        #     self.logger(str(tr1) + ",")
        #
        # self.logger("\n    Translations in axis2 for " + os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)) + ": ")
        # for tr2 in range(len(translation_axis2)):
        #     self.logger(str(tr2) + ",")
        # self.logger("\n")



        # Translations
        translation_axis1 = random.sample(range(-45, 45), self._translations_axis1)  # Generate some random numbers for translations
        translation_axis2 = random.sample(range(-45, 45), self._translations_axis2)

        num_augmentations = num_augmentations + (len(translation_axis2) * len(translation_axis1))

        for tr1 in range(len(translation_axis1)):
            for tr2 in range(len(translation_axis2)):
                for i in range(len(slice_list)):
                    self.__augment_translation__(slice = slice_list[i], translate_axis = (translation_axis1[tr1], translation_axis2[tr2]), aug_write_path=os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)), sliceNo = i)


        # Flips
        num_augmentations = num_augmentations + 1

        for i in range(len(slice_list)):
            self.__augment_flip_up_down(slice_list[i], aug_write_path=os.path.join(basenames_list[i] + "_Axis" + str(self._slice_axis) + "_" + str(sliceNo)))


        return num_augmentations



    def __augment_translation__(self, slice, translate_axis = (0, 0), aug_write_path = "/tmp/aug", sliceNo = 1):
        # Do the translation
        if (sliceNo == 0): # If the slice (i.e. the 0th slice) is the groundtruth slice, use 0 order, i.e. nearest neighbor interpolation
            ord = 0
        else:
            ord = 3

        translated_slice = scipy.ndimage.shift(slice, shift = translate_axis, mode = "nearest", order = ord)

        # Write to file
        self.__save_img__(aug_write_path + "_Tr" + str(translate_axis[0]) + "_" + str(translate_axis[1]), translated_slice)



    def __augment_rotation__(self, slice, degree = 0, aug_write_path = "/tmp/aug", sliceNo = 1):
        # Do the rotation
        if (sliceNo == 0): # If the slice (i.e. the 0th slice) is the groundtruth slice, use 0 order, i.e. nearest neighbor interpolation
            ord = 0
        else:
            ord = 3

        rotated_slice = scipy.ndimage.rotate(slice, degree, reshape = False, mode = "nearest", order = 3)

        # Write to file
        self.__save_img__(aug_write_path + "_Rot" + str(degree), rotated_slice)



    def __augment_flip_up_down(self, slice, aug_write_path = "/tmp/aug"):
        # Do the flip
        flipped_ud = np.flipud(slice)

        # Write to file
        self.__save_img__(aug_write_path + "_FlipUD", flipped_ud)



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
        y = np.zeros((self._batch_size, self._slice_dim[0], self._slice_dim[1], 1))

        # Generate the data
        for i in range(len(temp_list[0])):
            # msk_img = self.__load_img__(temp_list[0][i])
            #
            # swi_img = self.__load_img__(temp_list[1][i])
            # qsm_img = self.__load_img__(temp_list[2][i])

            # y[i, :, :, 0] = msk_img
            # X[i, :, :, 0] = swi_img
            # X[i, :, :, 1] = qsm_img

            y[i, :, :, 0] = self.__load_img__(temp_list[0][i])

            for j in range(self._n_channels):
                X[i, :, :, j] = self.__load_img__(temp_list[j + 1][i])
           
        return X, y



#
# # # For testing
# datafile = open(".\\AAA.csv", "r")
#
# vols = list()
# for lines in datafile:
#     iline = lines.strip("\n")
#     vols.append(iline)
#
# bs = 32
# sd = (256, 256, 256)
#
# import shutil
# write_path = "E:\\tmp"
# if (os.path.exists(write_path)):
#     shutil.rmtree(write_path)
#     os.mkdir(write_path)
# #
# dg = DataGenerator_V7(vols,
#                  write_path = write_path,
#                  axis = 2,
#                  batch_size = bs,
#                  slice_dim = sd,
#                  n_channels = 3,
#                  n_classes = 3,
#                  shuffle = True,
#                  pad = True,
#                  normalize = True,
#                  augment_cmbs = True,
#                  augment_regular = True,
#                  rotations = 2,
#                  translations_axis1 = 2,
#                  translations_axis2 = 2,
#                       logfile = "E:\\tmp\\log.txt")
#
# x, y = dg.__getitem__(0)
#
#
# for i in range(x.shape[0]):
#     plt.figure(figsize=(16, 8))
#
#     plt.subplot(1, 4, 1)
#     plt.imshow(x[i, :, :, 0], cmap = "gray")
#     plt.title("SWI")
#
#     plt.subplot(1, 4, 2)
#     plt.imshow(x[i, :, :, 1], cmap = "gray")
#     plt.title("QSM")
#
#     plt.subplot(1, 4, 3)
#     plt.imshow(x[i, :, :, 2], cmap = "gray")
#     plt.title("T2 Unscaled")
#
#     plt.subplot(1, 4, 4)
#     plt.imshow(y[i, :, :, 0], cmap = "gray")
#     plt.title("GroundTruth")
#
#     # plt.subplot(2, 3, 4)
#     # plt.imshow(x1[i, :, :, 0], cmap="gray")
#     # plt.title("SWI Z-Scaled")
#     #
#     # plt.subplot(2, 3, 5)
#     # plt.imshow(x1[i, :, :, 1], cmap="gray")
#     # plt.title("QSM Z-Scaled")
#     #
#     # plt.subplot(2, 3, 6)
#     # plt.imshow(y[i, :, :, 0], cmap="gray")
#     # plt.title("Mask")
#
#     plt.show()
#     plt.close()
#


# niftii = nib.load("/home/rashidt/Data/DeepLearning_CMB_Data/MESA/3014460/3014460_QSM_SWI_20180807093751_11_e1.nii.gz")
# img = niftii.get_fdata()
#
#
# slice = img[:, :, 41]
#
#
# flipped_lr = np.fliplr(slice)
# flipped_ud = np.flipud(slice)
#
# # rotated_20 = scipy.ndimage.rotate(slice, 20, reshape = False, mode = "nearest")
# # rotated_m20 = scipy.ndimage.rotate(slice, -20, reshape = False, mode = "nearest")
#
# #
# # shifted10 = scipy.ndimage.shift(slice, (10, 10), mode = "nearest")
# # shifted20 = scipy.ndimage.shift(slice, (20, 20), mode = "nearest")
#
#
# plt.figure(figsize=(16, 8))
#
# plt.subplot(1, 3, 1)
# plt.imshow(slice, cmap = "gray")
# plt.title("Original")
#
#
# plt.subplot(1, 3, 2)
# plt.imshow(flipped_lr, cmap = "gray")
# plt.title("Flipped LR")
#
# plt.subplot(1, 3, 3)
# plt.imshow(flipped_ud, cmap = "gray")
# plt.title("Flipped UD")
#
#
#
# plt.show()
# plt.close()
