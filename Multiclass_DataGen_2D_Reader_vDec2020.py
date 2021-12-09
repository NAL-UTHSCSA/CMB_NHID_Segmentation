
import os
import random
import sys
import time

import nibabel as nib
import numpy as np
import scipy.misc
import scipy.ndimage

import tensorflow as tf

from tensorflow.python import keras
from skimage.exposure import rescale_intensity


class Multiclass_DataGen_2D_Reader_vDec2020(keras.utils.Sequence):

    def __init__(self,
                 list_of_ids,
                 write_path = "/tmp/CNN_Model",
                 batch_size = 32,
                 slice_dim = (256, 256, 256),
                 n_channels = 4,
                 n_classes = 3,
                 shuffle = True,
                 ):

        self._slice_dim = slice_dim
        self._batch_size = batch_size

        self._n_channels = n_channels
        self._n_classes = n_classes
        self._shuffle = shuffle



        self._write_path_list = list(os.path.join(write_path, subject_id) for subject_id in list_of_ids)

        # Make a list of lists for all the volumes that were sliced and written to write_path's subdirectories
        self._list_vol_slices = list(list() for i in range(self._n_channels + 1)) # Empty list for all the volumes


        # For each channel, search the channel folder in the subject directory. During data generation,
        # all data is saved in subject specific folders, each subject specific folder having vol_0, vol_1,
        # vol_2, ..., vol_n subfolders
        for k in range(self._n_channels + 1):

            for subject_write_path in self._write_path_list:
                tmppath = os.path.join(subject_write_path, "vol_" + str(k))

                tmplist = list()
                for tmproot, tmpdirs, tmpfiles in os.walk(tmppath, topdown = False):
                    # Locate all the slice files. Each slice files are saved in Numpy .npz format

                    for fname in tmpfiles:
                        if (".npz" in fname):
                            self._list_vol_slices[k].append(os.path.join(tmproot, fname)) # Add the full path of the slice file

                # Make sure that all lists are sorted.
                self._list_vol_slices[k].sort()



        # Total number of vol_1 slices is considered as the reference
        self._total_slices = len(self._list_vol_slices[1])

        # Sanity check, make sure that the total slices each vol_i subdirectory is the same
        for i in range(len(self._list_vol_slices)):
            if (len(self._list_vol_slices[i]) != self._total_slices):
                print("\n\nERROR\nTotal number of slices not same for volumes\n\n")
                break

        print("  Total files per volume generated=" + str(self._total_slices) + "\n\n")


        self.on_epoch_end()



    def __load_img__(self, filename):

        # Load using scipy
        # X = scipy.misc.imread(filename)
        # return X


        # Load using Numpy
        X = np.load(filename)
        return X['a']



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


        # Generate data
        X, y = self.__data_generation(temp_list)

        return X, y



    def __data_generation(self, temp_list):

        X = np.zeros((self._batch_size, self._slice_dim[0], self._slice_dim[1], self._n_channels))
        y = np.zeros((self._batch_size, self._slice_dim[0], self._slice_dim[1], self._n_classes))

        # Generate the data
        for i in range(self._batch_size):

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

        return X, y
