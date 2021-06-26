import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.holopix_dataset import HolopixDataset
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader


#read the images
holopix_dataset = HolopixDataset(root='./toy_dataset/', split='train')

#process the image
synthetic_dataset = SyntheticBurst(holopix_dataset, burst_size=5, crop_sz=256)

#creating DataLoader object gives you option to shuffle etc
data_loader = DataLoader(synthetic_dataset, batch_size=4)

#write the image
for i, processedInstance in enumerate(synthetic_dataset):

    burst = processedInstance[0]
    gt = processedInstance[1]
    # print(burst.permute(0, 2, 3, 1).shape)
    # print(gt.permute(1, 2, 0).shape)

    burst_array = (burst.permute(0, 2, 3, 1).clamp(0.0, 1.0) *255).cpu().numpy().astype(np.uint8)
    gt = (gt.permute(1, 2, 0).clamp(0.0, 1.0) *255).cpu().numpy().astype(np.uint8)
    # print(burst_array.shape)


    cv2.imwrite('./toy_dataset/gt/image' + str(i) + '.jpg', gt)
    for j in range(burst_array.shape[0]):
        cv2.imwrite('./toy_dataset/burst/image' + str(i) + '_instance' + str(j) + '.jpg', burst_array[j])