import glob
import h5py
import numpy as np
import cv2

"""
Using the raw_image_frames_example.py the needed frames are cut out of the video.
and are in a folder (called 'raw_image' in this example).
The files are divided into train and test folder manually.
Each file name contains "yes" or "no" representing if the intersection was congested or not.
Based on "yes" or "no" the correct label is attached (0 represents "no", 1 represents "yes".
"""

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'datasets/raw_image_dataset.hdf5'  # address to where you want to save the hdf5 file

# Path to folder where saved images are
images_train_path = '../example_images/raw_image/train/*.jpg'
images_test_path = '../example_images/raw_image/test/*.jpg'


# Read addresses and labels from the 'train' folder
addrs_train = glob.glob(images_train_path)
addrs_test = glob.glob(images_test_path)
labels_train = [0 if 'no' in addr_train else 1 for addr_train in addrs_train]  # 0 = no, 1 = yes
labels_test = [0 if 'no' in addr_test else 1 for addr_test in addrs_test]  # 0 = no, 1 = yes


train_shape = (len(addrs_train), 64, 64)
test_shape = (len(addrs_test), 64, 64)

hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(addrs_train),), np.int8)
hdf5_file["train_labels"][...] = labels_train
hdf5_file.create_dataset("test_labels", (len(addrs_test),), np.int8)
hdf5_file["test_labels"][...] = labels_test


mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(addrs_train)):
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(addrs_train)))

    addr = addrs_train[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    # save the image
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(labels_train))


# loop over test addresses
for i in range(len(addrs_test)):
    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(addrs_test)))
    addr = addrs_test[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    # save the image
    hdf5_file["test_img"][i, ...] = img[None]

# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()