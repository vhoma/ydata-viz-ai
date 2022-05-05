import os
from zipfile import ZipFile, BadZipFile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import Affine3D as affine
import torch
from torch.utils.data import Dataset, DataLoader


def unzip_data(data_path, archive_path):
    print(f"Unzip '{archive_path}'...")
    with ZipFile(archive_path, 'r') as zip_obj:
        zip_obj.extractall(path=data_path)


def normalize(img, min_val, max_val):
    return (img - min_val) / (max_val - min_val)


def transform(img):
    alpha = np.random.randint(-45, 45)
    m = affine.Affine3dRotateCenterMatrix(alpha, img.shape, axis=2)
    return affine_transform(img, m), m, alpha


class Img3dDataSet(Dataset):
    def __init__(self, data_path, min_val, max_val):
        self.d_path = data_path
        self.min_val = min_val
        self.max_val = max_val
        self.transform = None
        self.target_transform = None
        names = [f for f in os.listdir(data_path) if f.endswith(".npz")]
        self.names_array = np.sort(np.array(names))

    def __getitem__(self, idx):
        name = self.names_array[idx]
        img3d = np.load(os.path.join(self.d_path, name))['I']
        img3d = normalize(img3d, self.min_val, self.max_val)

        # transform original image twice
        t1, m1, alpha1 = transform(img3d)
        t2, m2, alpha2 = transform(img3d)
        print("...transformed with angles: {} {}".format(alpha1, alpha2))

        # find transform matrix from 1st to 2nd
        matrix = affine.Affine3dRotateCenterMatrix(alpha2 - alpha1, img3d.shape, axis=2)

        # transpose images and remove last row from the matrix
        t1 = t1.transpose(2, 0, 1)
        t2 = t2.transpose(2, 0, 1)
        matrix = matrix[:3, :]

        # convert to torch
        return torch.from_numpy(t1).float(), torch.from_numpy(t2).float(), matrix

    def __len__(self):
        return len(self.names_array)


if __name__ == "__main__":
    curr_path = "."
    d_path = "./data"
    arch_path = os.path.join(curr_path, "data100.zip")
    #unzip_data(d_path, arch_path)
    dataset = Img3dDataSet(d_path, -1000, 1000)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y, mtrx = next(iter(dataloader))
    print(mtrx)
    plt.imshow(x[0][:, :, 10])
    plt.show()
    plt.imshow(y[0][:, :, 10])
    plt.show()
