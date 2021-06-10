from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import linecache as lc
# from skimage import io
import torch.utils.data as data
import torch


class BinaryDataset(data.Dataset):
    def __init__(self, class_name, dataset_name, positive_num=6000, negative_num=5000):
        imgs = []
        num1 = 0
        num0 = 0
        for sample in dataset_name:
            if num0 == -1 and num1 == -1:
                break
            if sample[1] == class_name:
                if num1 >= positive_num or num1 == -1:
                    num1 = -1
                    continue
                labeli = 1
                num1 = num1 + 1
                imgs.append((sample[0], labeli))
            else:
                if num0 >= negative_num or num0 == -1:
                    num0 = -1
                    continue
                labeli = 0
                num0 = num0 + 1
                imgs.append((sample[0], labeli))
        self.imgs = imgs

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return len(self.imgs)


def extract_dataset(class_num, dataset_name, posi_num, nega_num, batch_size):
    data_loader = []
    kwargs1 = {'num_workers': 0, 'pin_memory': True} 
    for i in range(class_num):
        ds_i = BinaryDataset(class_name=i, dataset_name=dataset_name, positive_num=posi_num, negative_num=nega_num)
        data_loader_i = data.DataLoader(dataset=ds_i,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    **kwargs1)
        data_loader.append(data_loader_i)
    return data_loader

class Delete_0_Dataset(data.Dataset):
    def __init__(self, dataset_name):
        imgs = []
        # num = 0
        for sample in dataset_name:
            if sample[1] != 0:
                # num = num + 1
                imgs.append(sample)
        self.imgs = imgs

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return len(self.imgs)

class Filter_0_Dataset(data.Dataset):
    def __init__(self, dataset_name, num_samples):
        imgs = []
        num = 0
        for sample in dataset_name:
            if sample[1] == 0 and num < num_samples:
                num = num + 1
                imgs.append(sample)
        self.imgs = imgs

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return len(self.imgs)
