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


#############
# another architecture
#########################

class Dense_Block2D(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(Dense_Block2D, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growthRate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat((x, out), 1)


class Dense_Block3D(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(Dense_Block3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=growthRate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat((x, out), 1)


class Transition_Layer2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        bn = self.relu(self.bn(x))
        out = self.conv(bn)
        out = self.max_pool(out)
        return out


class Transition_Layer3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer3d, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        bn = self.bn(x)
        out = self.relu(bn)
        out = self.conv(out)
        out = self.max_pool(out)
        return out


class Siam_AirNet2(nn.Module):
    def __init__(self, growthRate=8, num_init_features=8, bn_size=1, block_config2D=(1, 2, 4),
                 block_config3D=(8, 16, 32), batchnorm_on=True):
        super(Siam_AirNet2, self).__init__()
        self.flatten = nn.Flatten()

        self.firstlayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=growthRate, kernel_size=3, stride=1, padding=1),
            # (320 - 3 + 2*1)/1 + 1 = 320
            nn.BatchNorm2d(growthRate),  # can't understand should we use here BN or not and if yes - in which order
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  # (320 - 2)/2 + 1 = 160
        self.dense2D = nn.Sequential()
        self.dense3D = nn.Sequential()

        # Make 2D Dense Blocks and followed by Transition blocks
        for i, num_layers in enumerate(block_config2D):
            inChannels = num_layers * growthRate
            block = self._make_dense2D(inChannels, growthRate, num_layers)
            self.dense2D.add_module('dense2D%d' % (i + 1), block)
            trans = Transition_Layer2d(inChannels + num_layers * growthRate, inChannels + num_layers * growthRate)
            self.dense2D.add_module('trans2D%d' % (i + 1), trans)

            # Make 3D Dense Blocks and followed by Transition blocks
        for i, num_layers in enumerate(block_config3D):
            inChannels = num_layers * growthRate
            # num_input_features + i * growth_rate
            block = self._make_dense3D(inChannels, growthRate, num_layers)
            self.dense3D.add_module('dense3D%d' % (i + 1), block)
            trans = Transition_Layer3d(inChannels + num_layers * growthRate, inChannels + num_layers * growthRate)
            self.dense3D.add_module('trans3D%d' % (i + 1), trans)

        if batchnorm_on:
            logging.info("Batch normalization is ON")
            self.regression = nn.Sequential(
                nn.Linear(2 * 2 * 2 * 512 * 2, 1024),  # check the final dimensions
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1))
        else:
            logging.info("Batch normalization is OFF")
            self.regression = nn.Sequential(
                nn.Linear(2 * 2 * 2 * 512 * 2, 1024),  # check the final dimensions
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1))

    def _make_dense2D(self, inChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Dense_Block2D(inChannels, growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)

    def _make_dense3D(self, inChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Dense_Block3D(inChannels, growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)

    def forward_2D(self, x):
        # This function will be called for both images
        output = self.firstlayer(x)
        # dense block here should be done on the same size (160x160) of FM as in previous layer
        output = self.dense2D(output)
        # and now we are here.... 3D green part
        return output

    def forward_3D(self, x):
        output = self.dense3D(x)
        # it is expected 2x2x2x512
        output = self.flatten(output)
        return output

    def forward(self, input_t, input_ref):
        # In this function we pass in both images and obtain both vectors
        n_batches = input_t.shape[0]
        z_dim = 20
        x_y_dim = input_t.shape[-2]
        input_t = input_t.view(n_batches * z_dim, 1, x_y_dim, x_y_dim)
        input_ref = input_ref.view(n_batches * z_dim, 1, x_y_dim, x_y_dim)
        x_t = self.forward_2D(input_t)
        # print('Shape x_t after 2D ', x_t.shape) #Shape x_t after 2D  torch.Size([20, 64, 20, 20])
        x_ref = self.forward_2D(input_ref)
        # then concatenate the output for every 20 slices
        x_t = x_t.view(n_batches, x_t.shape[1], int(x_t.shape[0] / n_batches), x_t.shape[-2], x_t.shape[-2])
        # print('Shape x_t after reshaping before 3D ', x_t.shape)
        x_ref = x_ref.view(n_batches, x_ref.shape[1], int(x_ref.shape[0] / n_batches), x_ref.shape[-2], x_ref.shape[-2])
        x_t = self.forward_3D(x_t)
        x_ref = self.forward_3D(x_ref)
        # print('x_t and x_ref.shape', x_t.shape, x_ref.shape)
        input_conc = torch.cat((x_t, x_ref), dim=1)
        # print(input_conc.shape)
        res = self.regression(input_conc)
        return res

