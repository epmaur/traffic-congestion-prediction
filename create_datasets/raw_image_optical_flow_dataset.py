from scipy.misc import imread
from scipy import average
import numpy as np
import glob
import h5py
import os
import cv2

"""
Using the dense_opticalflow_frames.py the needed optical flow frames are cut out of the video.
Each example is in an individual folder containing 12 optical flow frames and one raw image file.
Example folders can be seen under 'example_images/optical_flow_raw_image'.
Each folder name contains "yes" or "no" representing if the intersection was congested or not.
Based on "yes" or "no" the correct label is attached (0 represents "no", 1 represents "yes".
The folders are divided into train and test folders manually.

NB! Folder example images exists only in Github repository due to file size capacity in ained.ttu.ee! 
https://github.com/epmaur/traffic-congestion-prediction/tree/master/example_images
"""


def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)
    else:
        return arr


def append_to_dataset(matrix, label, type, index):
    hdf5_path = 'datasets/optical_flow_raw_image_13_layers.hdf5'  # address to where you want to save the hdf5 file

    hdf5_file = h5py.File(hdf5_path, mode='a')
    print((hdf5_file[type + "_matrix"]).shape)
    print(matrix.shape)
    hdf5_file[type + "_matrix"][index, ...] = matrix[None]
    hdf5_file[type + "_labels"][index, ...] = label

    hdf5_file.close()


hdf5_path = 'datasets/optical_flow_raw_image_13_layers.hdf5'  # address of the hdf5 file

hdf5_file = h5py.File(hdf5_path, mode='w')

# Manually set the shape of matrix (nr of examples, img_width, img_height, nr of layers in matrix)
hdf5_file.create_dataset("train_matrix", (208, 64, 64, 13), np.int8, maxshape=(None, None, None, None))
hdf5_file.create_dataset("test_matrix", (78, 64, 64, 13), np.int8, maxshape=(None, None, None, None))
hdf5_file.create_dataset("train_labels", (208, ), np.int8, maxshape=(None, ))
hdf5_file.create_dataset("test_labels", (78, ), np.int8, maxshape=(None, ))


hdf5_file.close()

imageDirectory = '../example_images/optical_flow_raw_image'


root = imageDirectory + "/train"
index = 0
for path, subdirs, files in os.walk(root):
    for subdir in subdirs:
        label = 0 if 'no' in subdir else 1
        images_path = imageDirectory + '/train/' + subdir + '/*.png'
        addrs = glob.glob(images_path)
        a = np.zeros((64, 64, len(addrs) + 1))

        for i in range(len(addrs)):
            img = to_grayscale(imread(addrs[i]).astype(float))
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            a[:, :, i] = img

        jpg_path = imageDirectory + '/train/' + subdir + '/*.jpg'
        jpg_addrs = glob.glob(jpg_path)
        img = imread(jpg_addrs[0]).astype(float)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        a[:, :, 1] = to_grayscale(img)
        append_to_dataset(a, label, 'train', index)
        index += 1


index = 0
root = imageDirectory + "/test"
for path, subdirs, files in os.walk(root):
    for subdir in subdirs:
        label = 0 if 'no' in subdir else 1
        images_path = imageDirectory + '/test/' + subdir + '/*.png'
        addrs = glob.glob(images_path)
        a = np.zeros((64, 64, len(addrs) + 1))

        for i in range(len(addrs)):
            img = to_grayscale(imread(addrs[i]).astype(float))
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            a[:, :, i] = img

        jpg_path = imageDirectory + '/test/' + subdir + '/*.jpg'
        jpg_addrs = glob.glob(jpg_path)
        img = imread(jpg_addrs[0]).astype(float)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        a[:, :, 1] = to_grayscale(img)
        append_to_dataset(a, label, 'test', index)
        index += 1
