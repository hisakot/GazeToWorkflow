import csv
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config

class Dataset(Dataset):
    def __init__(self, data_dir, img_height, img_width, transform=None,
                 is_train_True, inf_rotate=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.inf_rotate = inf_rotate
        self.H = img_height
        self.W = img_width
        self.img_path = []
        cfg = config.Config()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.is_train:
            # something
        if self.inf_rotate:
            # something

        return inputs, targets
