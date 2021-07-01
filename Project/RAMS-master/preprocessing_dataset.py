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


# import utils and basic libraries
from preprocessing import load_dataset,select_T_images,register_dataset,augment_dataset
#from utils.preprocessing import load_dataset,select_T_images,register_dataset,augment_dataset
import numpy as np
import os
import matplotlib.pyplot as plt



#-------------
# Settings
#-------------
T = 9                                # number of temporal dimension
dataset_dir = 'probav_data'          # input dir (train val and test splitted)
dataset_output_dir = 'dataset'       # output dir
train_full = False                   # train without a validation



# train loading
# X_train, X_train_masks, y_train, y_train_masks = load_dataset(base_dir=dataset_dir, part="train")
X_train, y_train = load_dataset(base_dir=dataset_dir, part="train")
print(f"Train scenes: {len(X_train)} | Train RED y shape: {y_train.shape}")

# validation loading
# X_val, X_RED_val_masks, y_val, y_RED_val_masks = load_dataset(base_dir=dataset_dir, part="val")
X_val, y_val = load_dataset(base_dir=dataset_dir, part="val")
print(f"Val scenes: {len(X_val)} | Val RED y shape: {y_val.shape}")

# test loading
# X_test, X_RED_test_masks = load_dataset(base_dir=dataset_dir,part="test")
X_test = load_dataset(base_dir=dataset_dir,part="test")
print(f"Test scenes: {len(X_test)}")



# # 2.0 Dataset Pre-Processing

#TODO: permute and select 9 images





if train_full:
    X_train = np.concatenate((X_train, X_val))    
    y_train = np.concatenate((y_train, y_val))
    # y_train_masks = np.concatenate((y_train_masks, y_RED_val_masks))


# # 3.0 Visualize the Pre-Processed Datataset

#-------------
# Settings
#-------------
index = 30

fig, ax = plt.subplots(2, T, figsize=(20,5))
ax[0,i].imshow(X_train[index][...,i], cmap = 'gray')
ax[0,i].axis('off')
ax[1,i].imshow(X_train_masks[index][...,i], cmap = 'gray')
ax[1,i].axis('off')



if not os.path.isdir(dataset_output_dir):
    os.mkdir(dataset_output_dir)

# save training
np.save(os.path.join(dataset_output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(dataset_output_dir, 'y_train.npy'), y_train)
# np.save(os.path.join(dataset_output_dir, 'y_train_masks.npy'), y_train_masks)

# save validation
if not train_full:
    np.save(os.path.join(dataset_output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(dataset_output_dir, 'y_val.npy'), y_val)
    # np.save(os.path.join(dataset_output_dir, 'y_RED_val_masks.npy'), y_RED_val_masks)

# save test
np.save(os.path.join(dataset_output_dir, 'X_test.npy'), X_test)
