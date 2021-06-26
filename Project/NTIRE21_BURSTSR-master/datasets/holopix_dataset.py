import torch
# import os
import cv2
from os import listdir
from os.path import isfile, join


class HolopixDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train'):
        super().__init__()

        if split in ['train', 'val', 'test']:
            self.img_pth = join(root, split)
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_list = self._get_image_list()

    def _get_image_list(self):
        image_list = [f for f in listdir(self.img_pth) if isfile(join(self.img_pth, f))]
        return image_list

    def _get_image(self, im_id):
        path = join(self.img_pth, self.image_list[im_id])
        img = cv2.imread(path)
        return img

    def get_image(self, im_id):
        frame = self._get_image(im_id)

        return frame

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        frame = self._get_image(index)

        return frame
