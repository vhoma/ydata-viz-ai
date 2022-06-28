import os
import sys
import argparse
import logging
from zipfile import ZipFile, BadZipFile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import Affine3D as affine
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from kornia.geometry.transform import warp_affine3d
from PIL import Image, ImageDraw


LOG_LEVELS = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def get_device():
    if torch.cuda.is_available():
        device_name = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device_name = "mps"
    else:
        device_name = "cpu"
    return torch.device(device_name)


def parse_patient_num(s):
    return int(s.split('.')[0].split('_')[0].split('-')[-1])


def split_train_val(data_path, data_path_train, data_path_val, val_ratio=0.1):
    # create folders
    os.makedirs(data_path_train, exist_ok=True)
    os.makedirs(data_path_val, exist_ok=True)

    # calculate num files
    all_files = [f for f in os.listdir(data_path) if f.startswith("CQ") and f.endswith(".npz")]
    all_files.sort()
    num_files = len(all_files)
    num_val = int(num_files * val_ratio)
    num_train = num_files - num_val
    print(f"Num files: {num_files}, num_train: {num_train}, num_val: {num_val} ")

    # move files
    train_files = []
    i = 0
    for f in all_files:
        if i < num_train:  # first take files for train
            os.rename(os.path.join(data_path, f), os.path.join(data_path_train, f))
            train_files.append(parse_patient_num(f))
        else:  # then move files to val
            if parse_patient_num(f) in train_files:
                print(f"{f} already in train!")
                os.rename(os.path.join(data_path, f), os.path.join(data_path_train, f))
            else:
                os.rename(os.path.join(data_path, f), os.path.join(data_path_val, f))
        i += 1


def unzip_data(data_path, archive_path):
    print(f"Unzip '{archive_path}'...")
    with ZipFile(archive_path, 'r') as zip_obj:
        zip_obj.extractall(path=data_path)


def normalize(img, min_val, max_val):
    clipped_img = img.clamp(min_val, max_val)
    return (clipped_img - min_val) / (max_val - min_val)


def transform_img(img, angle, shape, device):
    m = affine.Affine3dRotateCenterMatrix(angle, shape, axis=2)
    m = torch.from_numpy(m[:-1, :]).to(device)
    new_img = warp_affine3d(torch.unsqueeze(img, dim=0).unsqueeze(dim=0), m.unsqueeze(dim=0), img.shape)
    return new_img[0][0], m


class Img3dDataSet(Dataset):
    def __init__(self, data_path, min_val, max_val, device, max_transform_angle=45):
        self.d_path = data_path
        self.min_val = min_val
        self.max_val = max_val
        self.transform = None
        self.target_transform = None
        names = [f for f in os.listdir(data_path) if f.endswith(".npz")]
        self.names_array = np.sort(np.array(names))
        self.device = device

        # this can be updated from outside to generate from different range
        self.max_transform_angle = max_transform_angle

        # generate fixed to be able to validate on consistent dataset
        self.fixed_angles = self.generate_fixed_angles()
        self.use_fixed_angles = False   # this flag can be updated outside

    def __getitem__(self, idx):
        name = self.names_array[idx]
        img3d = np.load(os.path.join(self.d_path, name))['I']
        shape_before_permute = img3d.shape   # need this because torch uses opposite dimentions order
        img3d = torch.from_numpy(img3d).float().permute(2, 0, 1).to(self.device)
        img3d = normalize(img3d, self.min_val, self.max_val)

        # transform original image twice
        if self.use_fixed_angles:
            alpha1, alpha2 = self.fixed_angles[idx]
        else:
            alpha1 = np.random.uniform(-self.max_transform_angle, self.max_transform_angle)
            alpha2 = np.random.uniform(-self.max_transform_angle, self.max_transform_angle)
        t1, m1 = transform_img(img3d, alpha1, shape_before_permute, self.device)
        t2, m2 = transform_img(img3d, alpha2, shape_before_permute, self.device)
        logging.debug("...transformed with angles: {} {}".format(alpha1, alpha2))

        # find transform matrix from 1st to 2nd
        matrix = affine.Affine3dRotateCenterMatrix(alpha2 - alpha1, shape_before_permute, axis=2)
        matrix = torch.from_numpy(matrix[:3, :]).to(self.device)

        # add file name for future identification
        file_id = str(name).split('-')[-1].split('.')[0]
        return t1, t2, matrix, file_id

    def __len__(self):
        return len(self.names_array)

    def generate_fixed_angles(self):
        res = []
        for i in range(self.__len__()):
            alpha1 = np.random.uniform(-self.max_transform_angle, self.max_transform_angle)
            alpha2 = np.random.uniform(-self.max_transform_angle, self.max_transform_angle)
            res.append((alpha1, alpha2))
        logging.info(f"Fixed transform angles:\n{res}")
        return res


def show_eval_overlap(x, y, matrix, device, max_val, epoch_num=None, scan_num=None):
    x_chan = x.unsqueeze(dim=1)  # add channels dimention
    y_chan = y.unsqueeze(dim=1)
    x_new = warp_affine3d(x_chan, matrix.reshape((matrix.shape[0], 3, 4)), x.shape[-3:])  # transform
    # take 1 slice from every image and create a grid image
    y_slice = y_chan[:, :, 10, :, :]
    x_new_slice = x_new[:, :, 10, :, :]
    chan3 = torch.zeros(x_new_slice.shape).to(device)

    n_rows = 4
    grid = make_grid(torch.cat((y_slice, x_new_slice, chan3), dim=1), nrow=n_rows, padding=1, pad_value=max_val)
    img = ToPILImage()(grid)

    # add scan numbers and epoch num
    if scan_num and epoch_num:
        d = ImageDraw.Draw(img)
        for i in range(x.shape[0]):
            # add scan id
            text_x = (i % n_rows) * 320 + 10
            text_y = (i // n_rows) * 320 + 10
            d.text((text_x, text_y), f"#{scan_num[i]}", fill=(255, 255, 255))
        text_x = (i % n_rows) * 320 + 250
        text_y = (i // n_rows) * 320 + 290
        d.text((text_x, text_y), f"Epoch#{epoch_num}", fill=(200, 200, 255))
    return img


def set_log_level(log_level):
    # set logging level
    logging.basicConfig()
    logging.getLogger().setLevel(LOG_LEVELS.get(log_level, logging.INFO))


def main(data_path, archive_path, min_val, max_val, log_level, seed=None):
    """
    consider this a unit test
    """
    np.random.seed(seed)
    set_log_level(log_level)

    # unzip_data(data_path, archive_path)
    dataset = Img3dDataSet(data_path, min_val, max_val, get_device())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y, mtrx = next(iter(dataloader))
    logging.info(f"Transformation matrix: {mtrx}")
    plt.imshow(x[0][10, :, :])
    plt.show()
    plt.imshow(y[0][10, :, :])
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--archive-path', required=True, type=str, help='Zip file with data')
    parser.add_argument('--data-path', type=str, default="./data/val", help='Working directory')
    parser.add_argument('--min-val', type=int, default=-1000, help='Min value for normalization')
    parser.add_argument('--max-val', type=int, default=1000, help='Max value for normalization')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--log-level', default="info", choices=LOG_LEVELS.keys(), help='Logging level, default "info"')
    return vars(parser.parse_args())


if '__main__' == __name__:
    args = get_args()
    print(args)
    main(**args)
