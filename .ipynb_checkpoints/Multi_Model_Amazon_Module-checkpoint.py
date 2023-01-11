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


class GroundCNN(nn.Module):
    def __init__(self):
        super(GroundCNN, self).__init__()
        self.dropped = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv2d(3, 10,
                               kernel_size=5)  # Input is a 3 plane 256x256 tensor (RBB) -> output 10 planes of 254x254
        self.pool_max = nn.MaxPool2d(3)  # output of dim-2 x dim-2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # input 10 planes 127x127, output 20 planes of 125x125
        self.pool_avg = nn.AvgPool2d(3)
        self.conv3 = nn.Conv2d(20, 60, kernel_size=5)
        self.fc1 = nn.Linear(60 * 7 * 7, 1000)  # single dense layer for the network
        self.fc2 = nn.Linear(1000,14)
        self.batchNorm = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.batchNorm(x)
        # print(f"step 1 : {x.shape}")
        x = self.pool_max(nn.functional.relu(self.conv1(x)))
        x = self.dropped(x)
        # print(f"step 2 : {x.shape}")
        x = nn.functional.relu(self.conv2(x))
        x = self.pool_avg(x)
        x = self.dropped(x)
        # print(f"step 3 : {x.shape}")
        x = nn.functional.relu(self.conv3(x))
        x = self.pool_max(x)
        x = self.dropped(x)
        # print(f"step 4 : {x.shape}")
        x = x.view(-1, 60 * 7 * 7)
        # print(f"step 5 : {x.shape}")
        x = self.fc1(x)
        x = self.dropped(x)
        # print(f"step 6 : {x.shape}")
        x = self.fc2(x)
        return x


class CloudCNN(nn.Module):
    def __init__(self):
        super(CloudCNN, self).__init__()
        self.batchNorm = nn.BatchNorm2d(3)
        self.dropped = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv2d(3, 10,
                               kernel_size=10)  # Input is a 3 plane 256x256 tensor (RBB) ->
        # output 10 planes of 218x218
        self.pool_max = nn.MaxPool2d(4)  # output of dim-2 x dim-2
        self.conv2 = nn.Conv2d(10, 30, kernel_size=5)  # input 10 planes 127x127, output 20 planes of 125x125
        self.pool_avg = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(30*14*14, 100)  # first dense layer for the network
        self.fc2 = nn.Linear(100, 3)  # second dense layer
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.batchNorm(x)
        # print(f"step 0 : {x.shape}")
        x = self.pool_max(nn.functional.relu(self.conv1(x)))
        x = self.dropped(x)
        # print(f"step 1 : {x.shape}")
        x = self.pool_avg(nn.functional.relu(self.conv2(x)))
        x = self.dropped(x)
        # print(f"step 2 : {x.shape}")
        x = x.view(-1, 30*14*14)
        # print(f"step 3 : {x.shape}")
        x = self.fc1(x)
        # print(f"step 4 : {x.shape}")
        x = self.fc2(x)
        # print(f"step 5 : {x.shape}")
        return self.smax(x)


class AmazonSpaceDual(Dataset):
    """Amazon aerial image dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on each batch.
        """
        # A Panda's table for tags associated with images.
        # 2 columns: "image_name" and "tags"
        self.labels = pd.read_csv(csv_file)

        # Dictionaries listing all the tags
        self.tags_cloud = {'clear': 0, 'partly_cloudy': 1, 'cloudy': 2}
        self.tags_ground = {'haze': 0, 'primary': 1, 'agriculture': 2, 'water': 3, 'habitation': 4, 'road': 5, 'cultivation': 6, 'slash_burn': 7, 'conventional_mine': 8, 'bare_ground': 9, 'artisinal_mine': 10, 'blooming': 11, 'selective_logging': 12, 'blow_down': 13}

        # Creation of binary tag presence for optimisation of model
        for tag in self.tags_ground.keys():
            a = [(tag in i.split()) for i in self.labels['tags']]
            self.labels[tag] = np.zeros(len(a), dtype=int) + a
        for tag in self.tags_cloud.keys():
            a = [(tag in i.split()) for i in self.labels['tags']]
            self.labels[tag] = np.zeros(len(a), dtype=int) + a

        # Other information
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reading the image
        img_name = os.path.join(self.root_dir,
                                self.labels['image_name'].iloc[idx])
        img_name = f'{img_name}.jpg'
        image = imread(img_name)

        # Applying given transforms
        if self.transform:
            image = self.transform(image)


        # Getting the target values
        cloud_target = self.labels.loc[idx, self.tags_cloud.keys()].to_numpy(dtype=np.float32)
        ground_target = self.labels.loc[idx, self.tags_ground.keys()].to_numpy(dtype=np.float32)

        # other output
        sample = {'image': image, 'cloud_target': cloud_target, 'ground_target': ground_target}

        return sample
