import os
import torch
import pandas as pd
from skimage.io import imread
import numpy as np

import torchvision.transforms.functional
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.transforms as T

data_folder = '../IPEO_Planet_project'
if not os.path.exists(data_folder):
    data_folder = input("Enter the data folder path: ")
    assert os.path.exists(data_folder), "I did not find the folder at, " + str(data_folder)


class AdjustSaturation(object):
    """Adjust the saturation of a tensor image.
    Args:
        saturation factor (float): if 0 -> black and white, if 1 -> same as the input
    """

    def __init__(self, saturation_factor):
        assert isinstance(saturation_factor, (int, float))
        self.saturation_factor = saturation_factor

    def __call__(self, img):
        new_tensor = transforms.functional.adjust_saturation(img, self.saturation_factor)

        return new_tensor


class NormalizeTensor(object):
    """
    Normalize image [0-1] (or 0-255) to zero-mean unit standard deviation
    """

    def __init__(self, mean=(0.504, 0.504, 0.503), std=(0.019, 0.018, 0.018)):
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float))
        self.mean = mean
        self.std = std

    def __call__(self, img):
        new_tensor = img.Normalize(self.mean, self.std)
        return new_tensor


default_transform = T.Compose([
    T.Resize((224, 224)),
    NormalizeTensor])


class MultiLayerCNN(nn.Module):
    def __init__(self):
        super(MultiLayerCNN, self).__init__()
        self.dropped = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv2d(3, 60,
                               kernel_size=11, stride=3)  # old kernel_size=5 and output channels was 10
        self.pool_max = nn.MaxPool2d(4, stride=2)  # old without stride
        self.conv2 = nn.Conv2d(60, 90, kernel_size=5)  # old input output was 10,20
        self.pool_avg = nn.AvgPool2d(4, stride=2) # old without stride
        self.conv3 = nn.Conv2d(90, 260, kernel_size=5) # old input output was 20,60
        self.pool_max2 = nn.MaxPool2d(3, stride=2)  # old without stride
        self.fc1 = nn.Linear(260 * 6 * 6, 100)  # single dense layer for the network, old: nn.Linear(60 * 10 * 10, 100)
        self.fc2 = nn.Linear(100, 17)  # single dense layer for the network
        self.batchNorm = nn.BatchNorm2d(3)
        self.loss = nn.BCELoss()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.batchNorm(x)
        #print(f"step 1 : {x.shape}")
        x = self.pool_max(nn.functional.relu(self.conv1(x)))
        #print(f"step 2 : {x.shape}")
        x = nn.functional.relu(self.conv2(x))
        #print(f"step 3 : {x.shape}")
        x = self.pool_avg(x)
        #print(f"step 4 : {x.shape}")
        x = self.conv3(x)
        x = self.pool_max2(x)
        #print(f"step 5 : {x.shape}")
        x = x.view(-1, 260 * 6 * 6)
        #print(f"step 6 : {x.shape}")
        x = self.fc1(x)
        x = self.dropped(x)
        #print(f"step 7 : {x.shape}")
        x = self.fc2(x)

        return x


class AmazonSpaces(Dataset):
    """Amazon aerial image dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)

        self.tags = self.labels['tags'].str.split(expand=True).stack().unique()
        for tag in self.tags:
            a = [(tag in i.split()) for i in self.labels['tags']]
            self.labels[tag] = np.zeros(len(a), dtype=int) + a

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels['image_name'].iloc[idx])
        img_name = f'{img_name}.jpg'
        image = imread(img_name)

        if self.transform:
            image = self.transform(image)

        labels = self.labels.loc[idx, self.tags].to_numpy(dtype=np.float64)

        # other output
        sample = {'image': image, 'labels': labels}

        # print(type(image), type(labels))
        return sample
