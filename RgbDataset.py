import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

from DataAugument.condition_data_augmentation import Augmenter


import shutil

AUGMENTATION_CONFIG = {
    'random_flip': True,
    'random_rotate': True,
    'random_noise': False,
    'random_cut': False
}

class RgbDataset(Dataset):
    """HSI dataset."""

    def __init__(self, root_dir, input_size, transform = False):
        super(RgbDataset, self).__init__()
        self.root_dir = root_dir
        self.lable_names = os.listdir(root_dir)
        self.lable_idx = {}
        self.length = 0
        self.images_name = []
        self.input_size = (input_size, input_size)
        self.transform = transform  # Whether to enable data enhancement
        count = 0
        for label in self.lable_names:
            datapath = root_dir + "/" + label
            lst = list(filter(lambda s: s[-3:] == "jpg", os.listdir(datapath)))
            for i in lst:
                path = label + "/" + i
                self.images_name.append(path)
            self.length = self.length + len(lst)
            self.lable_idx[label] = count
            count = count + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_name = self.images_name[index]
        image = cv2.imread(self.root_dir + '/' + image_name)
        assert image is not None
        _image = cv2.resize(image, dsize = self.input_size, interpolation=cv2.INTER_CUBIC)
        _image = np.array(_image)
        _image = _image.astype('float32')
        _image = _image/255
        _image = _image.transpose(2, 0, 1)
        _image = torch.from_numpy(_image)
        lable_name = image_name.split('/')[0]
        lable_index = self.lable_idx[lable_name]

        if self.transform is True:
            dataAugument = Augmenter(augmentation_config=AUGMENTATION_CONFIG)
            _image, lable_index = dataAugument([_image,lable_index])

        return [_image, lable_index]


if __name__ == '__main__':
    pass


