#!/usr/bin/env python
# coding: utf-8

# # Train RAMS Deep Neural Network on Proba-V Dataset
# ![proba_v_dataset](media/rams_architecture.png "Logo Title Text 1")
# 
# The following notebook provides a script to train the residual attention network for multi-image super-resolution (RAMS). It makes use of the pre-processed dataset (train and validation) saved in the 'dataset' folder and using the main settings it selects a band to train with. 
# 
# **NB**: We strongly discouraged to run this notebook without an available GPU on the host machine. The original training (ckpt folder) has been performed on a 2080 Ti GPU card with 11GB of memory in approximately 24 hours.
# 
# **The notebook is divided in**:
# - 1.0 [Dataset Loading](#loading)
# - 2.0 [Dataset Pre-Processing](#preprocessing)
#     - 2.1 Make patches
#     - 2.2 Clarity patches check
#     - 2.3 Pre-augment dataset (temporal permutation)
# - 3.0 [Build the network](#network)
# - 4.0 [Train the network](#train)

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


# import utils and basic libraries
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.preprocessing import gen_sub, bicubic
from utils.loss import l1_loss, psnr, ssim
from utils.network import RAMS
from utils.training import Trainer
from skimage import io
from zipfile import ZipFile


# In[ ]:


# gpu settings (we strongly discouraged to run this notebook without an available GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# In[ ]:


#-------------
# General Settings
#-------------
PATH_DATASET = 'dataset' # pre-processed dataset path
name_net = 'RAMS' # name of the network
LR_SIZE = 32 # pathces dimension
SCALE = 3 # upscale of the proba-v dataset is 3
HR_SIZE = LR_SIZE * SCALE # upscale of the dataset is 3
OVERLAP = 32 # overlap between pathces
CLEAN_PATH_PX = 0.85 # percentage of clean pixels to accept a patch
band = 'NIR' # choose the band for the training
checkpoint_dir = f'ckpt/{band}_{name_net}_retrain' # weights path
log_dir = 'logs' # tensorboard logs path
submission_dir = 'submission' # submission dir


# In[ ]:


#-------------
# Network Settings
#-------------
FILTERS = 32 # features map in the network
KERNEL_SIZE = 3 # convolutional kernel size dimension (either 3D and 2D)
CHANNELS = 9 # number of temporal steps
R = 8 # attention compression
N = 12 # number of residual feature attention blocks
lr = 1e-4 # learning rate (Nadam optimizer)
BATCH_SIZE = 32 # batch size
EPOCHS_N = 100 # number of epochs


# In[ ]:


# create logs folder
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


# <a id="loading"></a>
# # 1.0 Dataset loading

# In[ ]:


# load training dataset
X_train = np.load(os.path.join(PATH_DATASET, f'X_{band}_train.npy'))
y_train = np.load(os.path.join(PATH_DATASET, f'y_{band}_train.npy'))
y_train_mask = np.load(os.path.join(PATH_DATASET, f'y_{band}_train_masks.npy'))


# In[ ]:


# load validation dataset
X_val = np.load(os.path.join(PATH_DATASET, f'X_{band}_val.npy'))
y_val = np.load(os.path.join(PATH_DATASET, f'y_{band}_val.npy'))
y_val_mask = np.load(os.path.join(PATH_DATASET, f'y_{band}_val_masks.npy'))


# In[ ]:


# print loaded dataset info
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('y_train_mask: ', y_train_mask.shape)


print('X_val: ', X_val.shape)
print('y_val: ', y_val.shape)
print('y_val_mask: ', y_val_mask.shape)


# <a id="preprocessing"></a>
# # 2.0 Dataset Pre-Processing

# ## 2.1 Make patches

# In[ ]:


# create patches for LR images
d = LR_SIZE  # 32x32 patches
s = OVERLAP  # overlapping patches
# Ex: n = (128-d)/s+1 = 7 -> 49 sub images from each image
print(X_train.shape)
print(X_train[...,0].shape)
X_train_patches = gen_sub(X_train[...,0],d,s)
#X_train_patches = gen_sub(X_train,d,s)
X_val_patches = gen_sub(X_val[...,0],d,s)
#X_val_patches = gen_sub(X_val,d,s)

# In[ ]:


# create patches for HR images and masks
d = HR_SIZE  # 96x96 patches
s = OVERLAP * SCALE  # overlapping patches
# Ex: n = (384-d)/s+1 = 7 -> 49 sub images from each image

y_train_patches = gen_sub(y_train,d,s)
y_train_mask_patches = gen_sub(y_train_mask,d,s)


y_val_patches = gen_sub(y_val,d,s)
y_val_mask_patches = gen_sub(y_val_mask,d,s)


# In[ ]:


# print first patch and check if LR is in accordance with HR
fig, ax = plt.subplots(1,2, figsize=(10,10))
ax[0].imshow(X_train_patches[0,:,:,0], cmap = 'gray')
ax[1].imshow(y_train_patches[0,:,:,0], cmap = 'gray')


# In[ ]:


# free up memory
del X_train, y_train, y_train_mask

del X_val, y_val, y_val_mask


# ## 2.2 Clarity patches check

# In[ ]:


# find patches indices with a lower percentage of clean pixels in train array
patches_to_remove_train = [i for i,m in enumerate(y_train_mask_patches) if np.count_nonzero(m)/(HR_SIZE*HR_SIZE) < CLEAN_PATH_PX]


# In[ ]:


# find patches indices with a lower percentage of clean pixels in validation array
patches_to_remove_val = [i for i,m in enumerate(y_val_mask_patches) if np.count_nonzero(m)/(HR_SIZE*HR_SIZE) < CLEAN_PATH_PX]


# In[ ]:


# print number of patches to be removed
print(len(patches_to_remove_train))
print(len(patches_to_remove_val))


# In[ ]:


# remove patches not clean
X_train_patches = np.delete(X_train_patches,patches_to_remove_train,axis=0)
y_train_patches =  np.delete(y_train_patches,patches_to_remove_train,axis=0)
y_train_mask_patches =  np.delete(y_train_mask_patches,patches_to_remove_train,axis=0)

X_val_patches = np.delete(X_val_patches,patches_to_remove_val,axis=0)
y_val_patches =  np.delete(y_val_patches,patches_to_remove_val,axis=0)
y_val_mask_patches =  np.delete(y_val_mask_patches,patches_to_remove_val,axis=0)


# <a id="network"></a>
# # 3.0 Build the network

# In[ ]:


# build rams network
rams_network = RAMS(scale=SCALE, filters=FILTERS, 
                 kernel_size=KERNEL_SIZE, channels=CHANNELS, r=R, N=N)


# In[ ]:


# print architecture structure
rams_network.summary(line_length=120)


# <a id="train"></a>
# # 4.0 Train the network

# In[ ]:


trainer_rams = Trainer(rams_network, band, HR_SIZE, name_net,
                      loss=l1_loss,
                      metric=psnr,
                      optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
                      checkpoint_dir=os.path.join(checkpoint_dir),
                      log_dir=log_dir)


# In[ ]:


trainer_rams.fit(X_train_patches,
                [y_train_patches.astype('float32'), y_train_mask_patches], initial_epoch = 0,
                batch_size=BATCH_SIZE, evaluate_every=400, data_aug = True, epochs=EPOCHS_N,
                validation_data=(X_val_patches, [y_val_patches.astype('float32'), y_val_mask_patches])) 

