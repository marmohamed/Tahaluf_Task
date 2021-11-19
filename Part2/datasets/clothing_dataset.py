from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import StratifiedShuffleSplit

class ClothingDataset(Dataset):

    def __init__(self, csv_file, root_dir, split=None, transform=None, valid_size=0.2):
        self.df = pd.read_csv(csv_file)
        self.df = self.clean_dataset(self.df)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.class_names = sorted(list(self.df.label.unique()))
        self.n_classes = len(self.class_names)
        self.df.label = self.df.label.map(lambda i: self.class_names.index(i))

        if split is not None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=0)
            for train_index, test_index in sss.split(self.df['image'], self.df['label']):
                if split.lower() == 'valid':
                    self.df = self.df.iloc[test_index]
                else:
                    self.df = self.df.iloc[train_index]

        if split is not None and split.lower() == 'train':
            self.label_dist = self.df['label'].value_counts().sort_index().tolist()
        else:
          self.label_dist = None

        self.image_ids = self.df['image'].unique().tolist()

    def clean_dataset(self, df):
        df = df[(df.label != 'Other') & (df.label != 'Not sure')]
        return df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_id + ".jpg")
        image = io.imread(img_path)[:, :, :3]
        label = self.df[self.df['image'] == image_id]["label"].values
        label = torch.tensor(label)
        # sample = {'image': image, 'label': label}
        image = image / 255.
        image = image.astype(np.float32)
        if self.transform:
            sample = self.transform(image=image)
            image = sample['image']
            # label = sample['label']
        # print(img_path, image.shape)
        return image, label