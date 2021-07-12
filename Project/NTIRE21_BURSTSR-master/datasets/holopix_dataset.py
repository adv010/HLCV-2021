import torch
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

class HolopixDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', image_pair=True, sceneSampling=None):
        super().__init__()

        if split in ['train', 'val', 'test']:
            if image_pair:
                self.img_pth = [join(root, split, 'left'), join(root, split, 'right')]                
            else:
                self.img_pth = join(root, split, 'left')
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_pair = image_pair
        self.sceneSampling = sceneSampling
        self.image_list = self._get_image_list()

    def _get_image_list(self):
        if self.image_pair:
            image_list1 = [f for f in sorted(listdir(self.img_pth[0])) if isfile(join(self.img_pth[0], f))]
            image_list2 = [f for f in sorted(listdir(self.img_pth[1])) if isfile(join(self.img_pth[1], f))]

            if self.sceneSampling is not None:
                image_list1 = image_list1[:self.sceneSampling]
                image_list2 = image_list2[:self.sceneSampling]

            image_list = [image_list1, image_list2]
        else:
            image_list = [f for f in listdir(self.img_pth) if isfile(join(self.img_pth, f))]

            if self.sceneSampling is not None:
                image_list = image_list[:self.sceneSampling]

        return image_list

    def _get_image(self, im_id):
        if self.image_pair:
            path1 = join(self.img_pth[0], self.image_list[0][im_id])
            img1 = cv2.imread(path1)
            path2 = join(self.img_pth[1], self.image_list[1][im_id])
            img2 = cv2.imread(path2)
            # print(path1)
            # print(path2)
            img_list = [img1, img2]
            img = np.stack(img_list, axis=0)
        else:
            path = join(self.img_pth, self.image_list[im_id])
            img = cv2.imread(path)
        return img

    def get_image(self, im_id):
        frame = self._get_image(im_id)
        return frame

    def __len__(self):
        if self.image_pair:
            return len(self.image_list[0])
        else:
            return len(self.image_list)

    def __getitem__(self, index):
        frame = self._get_image(index)
        return frame
