
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t

    

class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 低光、高光、结构图文件路径列表
        self.low_folder = os.path.join(data_dir, 'low')
        self.high_folder = os.path.join(data_dir, 'high')
        self.low_struct_folder = os.path.join(data_dir, 'low_s')

        self.low_files = sorted([x for x in os.listdir(self.low_folder) if is_image_file(x)])
        self.high_files = sorted([x for x in os.listdir(self.high_folder) if is_image_file(x)])
        self.low_struct_files = sorted([x for x in os.listdir(self.low_struct_folder) if is_image_file(x)])

        # 文件名必须一一对应
        assert self.low_files == self.high_files == self.low_struct_files, \
            "low, high, low_s 文件名必须一一对应"

    def __getitem__(self, index):
        # 文件路径
        low_path = os.path.join(self.low_folder, self.low_files[index])
        high_path = os.path.join(self.high_folder, self.high_files[index])
        low_struct_path = os.path.join(self.low_struct_folder, self.low_struct_files[index])

        # 加载图片
        im_low = load_img(low_path)
        im_high = load_img(high_path)
        im_low_struct = load_img(low_struct_path)

        # 同步随机种子
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)
            random.seed(seed)
            torch.manual_seed(seed)
            im_high = self.transform(im_high)
            random.seed(seed)
            torch.manual_seed(seed)
            im_low_struct = self.transform(im_low_struct)

        return im_low, im_high, im_low_struct

    def __len__(self):
        return len(self.low_files)

    

class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        # 低光、正常光、结构图文件路径列表
        self.low_folder = os.path.join(data_dir, 'Low')
        self.normal_folder = os.path.join(data_dir, 'Normal')
        self.low_struct_folder = os.path.join(data_dir, 'Low_s')

        self.low_files = sorted([x for x in os.listdir(self.low_folder) if is_image_file(x)])
        self.normal_files = sorted([x for x in os.listdir(self.normal_folder) if is_image_file(x)])
        self.low_struct_files = sorted([x for x in os.listdir(self.low_struct_folder) if is_image_file(x)])

        # 文件名必须一一对应
        assert self.low_files == self.normal_files == self.low_struct_files, \
            "Low, Normal, Low_s 文件名必须一一对应"

    def __getitem__(self, index):
        # 文件路径
        low_path = os.path.join(self.low_folder, self.low_files[index])
        normal_path = os.path.join(self.normal_folder, self.normal_files[index])
        low_struct_path = os.path.join(self.low_struct_folder, self.low_struct_files[index])

        # 加载图片
        im_low = load_img(low_path)
        im_normal = load_img(normal_path)
        im_low_struct = load_img(low_struct_path)

        # 同步随机种子
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)
            random.seed(seed)
            torch.manual_seed(seed)
            im_normal = self.transform(im_normal)
            random.seed(seed)
            torch.manual_seed(seed)
            im_low_struct = self.transform(im_low_struct)

        return im_low, im_normal, im_low_struct

    def __len__(self):
        return len(self.low_files)



class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 低光、正常光、结构图文件路径列表
        self.low_folder = os.path.join(data_dir, 'Low')
        self.normal_folder = os.path.join(data_dir, 'Normal')
        self.low_struct_folder = os.path.join(data_dir, 'Low_s')

        self.low_files = sorted([x for x in os.listdir(self.low_folder) if is_image_file(x)])
        self.normal_files = sorted([x for x in os.listdir(self.normal_folder) if is_image_file(x)])
        self.low_struct_files = sorted([x for x in os.listdir(self.low_struct_folder) if is_image_file(x)])

        # 文件名必须一一对应
        assert self.low_files == self.normal_files == self.low_struct_files, \
            "Low, Normal, Low_s 文件名必须一一对应"

    def __getitem__(self, index):
        # 文件路径
        low_path = os.path.join(self.low_folder, self.low_files[index])
        normal_path = os.path.join(self.normal_folder, self.normal_files[index])
        low_struct_path = os.path.join(self.low_struct_folder, self.low_struct_files[index])

        # 加载图片
        im_low = load_img(low_path)
        im_normal = load_img(normal_path)
        im_low_struct = load_img(low_struct_path)

        # 同步随机种子
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)
            random.seed(seed)
            torch.manual_seed(seed)
            im_normal = self.transform(im_normal)
            random.seed(seed)
            torch.manual_seed(seed)
            im_low_struct = self.transform(im_low_struct)

        return im_low, im_normal, im_low_struct

    def __len__(self):
        return len(self.low_files)
    

