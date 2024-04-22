import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import _pickle as cPickle


def padding(spec, ref_len):
    frame_len, filters = spec.shape
    padd_len = ref_len - frame_len
    if padd_len > 0:
        return torch.cat((spec, torch.zeros(padd_len, filters, dtype=spec.dtype)), 0)
    else:
        t = spec[0:220, :]
        return spec[0:220, :]


import torch


def repeat_padding(spec, ref_len):
    frame_len, filters = spec.shape
    if frame_len < ref_len:
        # Calculate how many times we need to repeat the spec
        num_repeats = ref_len // frame_len
        additional_frames = ref_len % frame_len

        # Repeat the whole spec for num_repeats times
        spec_repeated = spec.repeat(num_repeats, 1)

        # If additional frames are needed, take a slice from the spec
        if additional_frames > 0:
            spec_repeated = torch.cat((spec_repeated, spec[:additional_frames]), 0)

        return spec_repeated
    else:
        # If spec is longer than the reference length, truncate it
        return spec[:ref_len, :]


class AudioDataset(Dataset):  # 创建一个叫做DogVsCatDataset的Dataset，继承自父类torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, transform=None):

        self.root_dir = root_dir
        self.fl_path = glob.glob(os.path.join(root_dir, '*.pckl'))

    def __len__(self):
        return len(self.fl_path)

    def __getitem__(self, idx):

        with open(self.fl_path[idx], 'rb') as feature_handle:
            file = cPickle.load(feature_handle)
        if isinstance(file[0], torch.Tensor):
            lfcc = file[0]
        else:
            lfcc = torch.from_numpy(file[0])
        # lfcc= padding(lfcc.squeeze(),220)
        lfcc = repeat_padding(lfcc.squeeze(), 750)
        # lfcc=lfcc.t()
        label = torch.from_numpy(np.array(file[1][2] - 1))  # 获得类型
        name = os.path.basename(self.fl_path[idx]).replace('.pckl', '')

        return lfcc, name, label
