#!/usr/bin/python3
# # Preprocessing the Proba-V Dataset
# ![proba_v_dataset](media/proba_v_dataset.jpg "Logo Title Text 1")
# 
# The following notebook provides a very flexible pipeline for processing the Proba-V Dataset. We have already split the original dataset in train validation and test. The test set is the original one of the ESA Proba-V challenge without ground-truths. The validation set is composed of all the scenes we used to evaluate our network and all significant solutions presented in literature at the time of writing.
# 
# **NB**: with the setting "train_full=True" our validation split will be ignored, and you will have a pre-processed dataset with all scenes available in the dataset. It is useful if you want to compete in the [PROBA-V Super Resolution post mortem Challenge](https://kelvins.esa.int/proba-v-super-resolution-post-mortem/home/)
# 
# **The notebook is divided in**:
# - 1.0 [Dataset Loading](#section_ID)
# - 2.0 [Dataset pre-processing](#preprocessing)
#     - 2.1 Register dataset
#     - 2.2 Select the best T LR images
#     - 2.3 Pre-augment dataset (temporal permutation)
# - 3.0 [Visualize the Pre-Processed Datataset](#visualize)
# - 4.0 [Save dataset](#save)

#Akshay system: run this in conda base


# import utils and basic libraries
# from preprocessing import load_dataset, register_dataset
from utils.preprocessing import load_dataset, register_dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib


#-------------
# Settings
#-------------
L = 10                               # number of temporal dimensions loaded
T = 9                                # number of temporal dimensions to be used
# dataset_dir = os.getcwd()          # input dir (train val and test splitted)
# dataset_output_dir = 'Holopix50k_burst/grayscale2/output'       # output dir
dataset_dir = "../training_datasets/Holopix50k_burst/grayscale2"          # input dir (train val and test splitted)
dataset_output_dir = '../training_datasets/Holopix50k_burst/grayscale2/npy_files_4000'       # output dir
train_full = False                   # train without a validation

pathlib.Path(dataset_output_dir).mkdir(parents=True,exist_ok=True)



# # validation loading
# X_val, X_val_masks, y_val, y_val_masks = load_dataset(base_dir=dataset_dir, part="val", L=L, T=T)
# print(f"Val scenes: {len(X_val)} | Val RED y shape: {y_val.shape}")

# # validation registration
# X_val, X_val_masks = register_dataset(X_val, X_val_masks)

# # if train_full:
# #     X_train = np.concatenate((X_train, X_val))    
# #     y_train = np.concatenate((y_train, y_val))
#     # y_train_masks = np.concatenate((y_train_masks, y_RED_val_masks))

# # # save validation
# if not train_full:
#     np.save(os.path.join(dataset_output_dir, 'X_val.npy'), X_val)
#     np.save(os.path.join(dataset_output_dir, 'y_val.npy'), y_val)
#     np.save(os.path.join(dataset_output_dir, 'y_val_masks.npy'), y_val_masks)

# print("saved val npy files")

# del X_val
# del X_val_masks
# del y_val
# del y_val_masks




# test loading
X_test, X_test_masks, y_test, y_test_masks = load_dataset(base_dir=dataset_dir,part="test", L=L, T=T)
print(f"Test scenes: {len(X_test)} | Test y shape: {y_test.shape}")

# test registration
X_test, X_test_masks = register_dataset(X_test, X_test_masks)

# # save test
np.save(os.path.join(dataset_output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(dataset_output_dir, 'y_test.npy'), y_test)
np.save(os.path.join(dataset_output_dir, 'y_test_masks.npy'), y_test_masks)

print("saved test npy files")

del X_test
del X_test_masks
del y_test
del y_test_masks




# # train loading
# X_train, X_train_masks, y_train, y_train_masks = load_dataset(base_dir=dataset_dir, part="train", L=L, T=T)
# print(f"Train scenes: {len(X_train)} | Train y shape: {y_train.shape}")
# print(f"Train single scene shape: {X_train[0].shape}")

# # # 3.0 Visualize the Pre-Processed Datataset
# index = 1
# fig = plt.figure(figsize=(8, 8))
# columns = 5
# rows = 2
# for i in range(T):
#     fig.add_subplot(rows, columns, i+1)
#     plt.imshow(X_train[index][...,i], cmap = 'gray')
# plt.savefig(os.path.join(dataset_output_dir,'sample_preprocessing_before_registration.png'))

# # train registration
# X_train, X_train_masks = register_dataset(X_train, X_train_masks)


# # # 3.0 Visualize the Pre-Processed Datataset
# fig = plt.figure(figsize=(8, 8))
# columns = 5
# rows = 2
# for i in range(T):
#     fig.add_subplot(rows, columns, i+1)
#     plt.imshow(X_train[index][...,i], cmap = 'gray')
# plt.savefig(os.path.join(dataset_output_dir,'sample_preprocessing.png'))

# # save training
# np.save(os.path.join(dataset_output_dir, 'X_train.npy'), X_train)
# np.save(os.path.join(dataset_output_dir, 'y_train.npy'), y_train)
# np.save(os.path.join(dataset_output_dir, 'y_train_masks.npy'), y_train_masks)

# print("saved train npy files")

# del X_train
# del X_train_masks
# del y_train
# del y_train_masks
