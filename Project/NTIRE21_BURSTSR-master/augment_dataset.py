import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.holopix_dataset import HolopixDataset
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader

grayscale = False

# print("HERE: 0")
#read the images
holopix_dataset = HolopixDataset(root='./toy_dataset/', split='train')

# print("HERE: 1")
#process the image
synthetic_dataset = SyntheticBurst(holopix_dataset, burst_size=5, crop_sz=256)

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
    cv2.imwrite('./toy_dataset/gt/image' + str(i) + '.jpg', gt)

    for j in range(burst_array.shape[0]):
        if grayscale:
            res = cv2.cvtColor(burst_array[j],cv2.COLOR_BGR2GRAY)
        else:
            res = burst_array[j]
        cv2.imwrite('./toy_dataset/burst/image' + str(i) + '_instance' + str(j) + '.jpg', res)