
import os
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import *
import torch.nn.functional as F

class SICEDatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
            factor = 8
            h, w = input.shape[1], input.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect').squeeze(0)
        return input, file, h, w

    def __len__(self):
        return len(self.data_filenames)
    


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, struct_dir=None, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

        self.struct_dir = struct_dir
        if struct_dir is not None:
            struct_filenames = [join(struct_dir, x) for x in os.listdir(struct_dir) if is_image_file(x)]
            struct_filenames.sort()
            # 保证文件名对应
            self.struct_filenames = struct_filenames
            # print("68-",data_filenames,struct_filenames)
            assert [os.path.basename(f) for f in data_filenames] == [os.path.basename(f) for f in struct_filenames], \
                "结构图和原图文件名必须一一对应"
        else:
            self.struct_filenames = None

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.struct_filenames is not None:
            struct_img = load_img(self.struct_filenames[index])
            if self.transform:
                input = self.transform(input)
                struct_img = self.transform(struct_img)
            return input, struct_img, file
        else:
            if self.transform:
                input = self.transform(input)
            return input, file

    def __len__(self):
        return len(self.data_filenames)
