import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.holopix_dataset import HolopixDataset
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader
from os.path import join
from pathlib import Path


inputPath = '/media/akshay/akshay_HDD/saarland/sem2/HLCV/hlcv2021/Project/training_datasets/Holopix50k/'
outputPath = '/media/akshay/akshay_HDD/saarland/sem2/HLCV/hlcv2021/Project/training_datasets/Holopix50k_burst/grayscale2/'
lrOutputPath = join(outputPath, 'burst')
hrOutputPath = join(outputPath, 'gt')
Path(lrOutputPath).mkdir(parents=True, exist_ok=True)
Path(hrOutputPath).mkdir(parents=True, exist_ok=True)

grayscale = True
augmentImagePair = True

# print("HERE: 0")
#read the images
holopix_dataset = HolopixDataset(root=inputPath, split='test', image_pair=augmentImagePair)

# print("HERE: 1")
#process the image
#burst size corresponds to one input and not a pair
synthetic_dataset = SyntheticBurst(holopix_dataset, burst_size=5, crop_sz=384, image_pair=augmentImagePair)

# print("HERE: 2")
#creating DataLoader object gives you option to shuffle etc
data_loader = DataLoader(synthetic_dataset, batch_size=4)

# print("HERE: 3")
#write the image
for i, processedInstance in enumerate(synthetic_dataset):

    # print("HERE: 3.1")
    burst = processedInstance[0]
    gt = processedInstance[1]
    # print(burst.permute(0, 2, 3, 1).shape)
    # print(gt.permute(1, 2, 0).shape)

    burst_array = (burst.permute(0, 2, 3, 1).clamp(0.0, 1.0) *255).cpu().numpy().astype(np.uint8)
    gt = (gt.permute(1, 2, 0).clamp(0.0, 1.0) *255).cpu().numpy().astype(np.uint8)
    # print(burst_array.shape)

    if grayscale:
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(hrOutputPath + '/image' + str(i) + '.jpg', gt)

    for j in range(burst_array.shape[0]):
        if grayscale:
            res = cv2.cvtColor(burst_array[j],cv2.COLOR_BGR2GRAY)
        else:
            res = burst_array[j]
        cv2.imwrite(lrOutputPath + '/image' + str(i) + '_instance' + str(j) + '.jpg', res)