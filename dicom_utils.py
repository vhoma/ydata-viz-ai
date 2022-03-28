import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os, shutil
from skimage.transform import resize
import wget
from zipfile import ZipFile
import pandas as pd
import scipy.ndimage
import json
import skimage.morphology as morph


def download_data(input_file, data_path, max_files=10):
    """
    :param max_files: max files to download
    :param input_file: text file with links to download data
    :param data_path: where to save the data
    :return: None
    """
    # create data_path if needed
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # change current path to download
    os.chdir(data_path)
    with open(input_file, 'r') as f:
        # start downloading!
        i = 1
        for line in f.readlines():
            if i > max_files:
                print(f"Reached max {max_files}")
                break
            i += 1
            print(line)
            wget.download(line.replace('\n', ''), data_path)


def unzip_data(data_path):
    """
    :param data_path: directory with zip files
    :return: None
    """
    all_zips = sorted(os.listdir(data_path))
    os.chdir(data_path)
    for archive in all_zips:
        if archive.startswith("."):
            continue    # ignore hidden files
        print(archive)
        with ZipFile(archive, 'r') as zip_obj:
            zip_obj.extractall()
        # delete file after unzip
        os.remove(archive)


def extract_scans_info(data_path):
    """
    :param data_path: path where data is tored
    :return: list of dicts not including pixel data
    """
    data_fields = []
    # extract basic info about the scans
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            fname = os.path.join(root, name)
            print(fname)
            if name.endswith(".dcm"):
                f = pydicom.dcmread(fname)
                keyval = {}
                for k in f.keys():
                    if not f[k].keyword == 'PixelData':
                        keyval[f[k].keyword] = f[k].value
                keyval['path'] = root
                keyval['num_slices'] = len(files)
                data_fields.append(keyval)
                break
    return data_fields


def filter_data_fields(data_fields, max_slice_thickness=2, min_height_mm=100):
    """
    Filter out scans that are not detailed enough
    :return: pandas dataframe
    """
    df = pd.DataFrame(data_fields)
    return df[(df['SliceThickness'] < max_slice_thickness) | (df['SliceThickness'] * df['num_slices'] > min_height_mm)]


def save_3d_images(df, data_path):
    """
    Will save chosen scans as .npz and remove everything else
    """
    # save every image as .npz file
    for i in df.index:
        dir_path = df.loc[i]['path']
        img3d = read_ct_scan(dir_path)
        file_name = f"{df.loc[i]['PatientID']}_{i}.npz"
        file_path = os.path.join(data_path, file_name)
        df.loc[i]['path'] = file_path
        np.savez(file_path, I=img3d)
        # now we don't need original slices files anymore
        shutil.rmtree(dir_path)
    # cleanup folders
    for name in os.listdir(data_path):
        path = os.path.join(data_path, name)
        if not name.endswith(".npz") and os.path.isdir(path):
            shutil.rmtree(path)
    df.to_csv(os.path.join(data_path, "data_fields.csv"))


def transform_to_hu_dir(data_path, csv_name="data_fields.csv"):
    # read data_fields
    df = pd.read_csv(os.path.join(data_path, csv_name), index_col=[0])

    # work with every scan one by one
    for name in os.listdir(data_path):
        file_path = os.path.join(data_path, name)
        if not name.endswith(".npz"):
            continue
        print(file_path)
        # read data
        data_fields = get_data_fields(name, df)
        image = np.load(file_path)['I']
        # transform
        image_transformed = transform_to_hu(data_fields['RescaleIntercept'], data_fields['RescaleSlope'], image)
        # write to the same file
        np.savez(file_path, I=image_transformed)


def transform_to_hu(intercept, slope, image, min_val=-1000, max_val=1000):
    #intercept = medical_image.RescaleIntercept
    #slope = medical_image.RescaleSlope
    new_image = image * slope + intercept
    return np.clip(new_image, min_val, max_val)


def mask_largest_component(mask):
    """
    Returns a mask that contains only the largest connected component in a given binary mask.
    """
    assert mask.ndim == 2
    if np.max(mask):
        (label_array, _) = morph.label(mask, background=False, return_num=True)
        label_sizes = np.bincount(label_array.ravel())
        label_of_largest_component = np.argmax(label_sizes[1:])+1
        return label_array == label_of_largest_component
    else:
        return mask


def extract_head_mask(arr, head_mask_hu_bone_threshold=900):
    """
    Given a 3D array:
        - MIP along z axis
        - Threshold the HU levels
        - Keep largest connected component
    Return 3D binary mask of the head (2D mask repeated along Z axis)
    """
    assert arr.ndim == 3
    pixelwise_mask_arr = (np.max(arr, axis=2) > head_mask_hu_bone_threshold)  # check if this value is good fo all DS cases, maybe different scans have different scalse
    pixelwise_mask_largest_component = mask_largest_component(pixelwise_mask_arr)
    final_mask = morph.convex_hull_image(pixelwise_mask_largest_component)
    mask_arr = np.repeat(np.expand_dims(final_mask, axis=2), arr.shape[2], axis=2)
    return mask_arr


def clean_noise(image, head_mask_hu_air=-1000):
    mask = extract_head_mask(image)
    new_image = image.copy()
    new_image[~mask] = head_mask_hu_air
    return new_image


def clean_noise_dir(data_path):
    # work with every scan one by one
    for name in os.listdir(data_path):
        file_path = os.path.join(data_path, name)
        if not name.endswith(".npz"):
            continue
        print(f"clean_noise {file_path}")
        image = np.load(file_path)['I']
        new_image = clean_noise(image)
        np.savez(file_path, I=new_image)   # write to the same file


def resample(image, data_fields, new_spacing=(1, 1, 1)):
    # Determine current pixel spacing
    spacing = map(float, (json.loads(data_fields.PixelSpacing) + [data_fields.SliceThickness]))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image


def resample_dir(data_path, csv_name="data_fields.csv"):
    # read data_fields
    df = pd.read_csv(os.path.join(data_path, csv_name), index_col=[0])

    # work with every scan one by one
    for name in os.listdir(data_path):
        file_path = os.path.join(data_path, name)
        if not name.endswith(".npz"):
            continue
        print(f"resample {file_path}")
        image = np.load(file_path)['I']
        data_fields = get_data_fields(name, df)
        if data_fields is None:
            continue
        new_image = resample(image, data_fields)
        np.savez(file_path, I=new_image)  # write to the same file


def get_data_fields(file_name, df):
    try:
        idx = int(file_name.split('.')[0].split('_')[-1])
    except ValueError:
        print(f"Bad file name: {file_name}")
        return None
    return df.loc[idx]


def read_ct_scan(input_dir):
    # load the DICOM files
    files = []
    print('glob: {}'.format(input_dir))
    for fname in glob.glob(os.path.join(input_dir, "*"), recursive=False):
        # print("loading: {}".format(fname))
        files.append(pydicom.dcmread(fname))

    print("file count: {}".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    return img3d


def ct_image_filter(img3d, val_from, val_to, default_value=-1000):
    res = np.clip(img3d, val_from, val_to)
    return res


def scale_ct_img(img3d, shape):
    shape = shape + (img3d['data'].shape[2],)
    img_resized = resize(img3d['data'], shape)
    return {'data': img_resized, 'aspect': img3d['aspect']}


if __name__ == "__main__":
    path = "./data/CQ500CT1 CQ500CT1/Unknown Study/CT 2.55mm"
    ct_img = read_ct_scan(path)
    filtered = ct_image_filter(ct_img, -100, 100)
    plt.imshow(filtered['data'][:, :, ct_img['data'].shape[2] // 2], cmap='gray')
    plt.show()