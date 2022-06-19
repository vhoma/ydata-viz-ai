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
from kornia.geometry.transform import warp_affine3d


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
    return (img - min_val) / (max_val - min_val)


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

        # convert to torch
        return t1, t2, matrix

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
