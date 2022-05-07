import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging


class SiamAirNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.firstlayer = nn.Sequential(
            # how many channels do we have??
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=3,stride=1,padding=1),
            # (320 - 3 + 2*1)/1 + 1 = 320
            nn.ReLU(inplace=True),
            # (320 - 2)/2 + 1 = 160
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=1,stride=1,padding=0),
            # (160 - 1 )/1 + 1
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            # 80
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.regression = nn.Sequential(
            # we have to double here because the images will be concatenated
            nn.Linear(20*20*20*2, 1024),
            # nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 12)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.firstlayer(x)
        output = self.transition(output)
        output = self.transition(output)
        output = self.transition(output)
        output = self.flatten(output)
        return output

    def forward(self, input_t, input_ref):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input_t = self.forward_once(input_t)
        # print(input_t.shape)
        input_ref = self.forward_once(input_ref)
        # print(input_ref.shape)
        input_conc = torch.cat((input_t,input_ref),dim=1)
        # print(input_conc.shape)
        res = self.regression(input_conc)
        return res

