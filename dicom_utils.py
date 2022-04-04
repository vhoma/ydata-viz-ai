import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os, shutil
from skimage.transform import resize
import wget
from zipfile import ZipFile, BadZipFile
import pandas as pd
import scipy.ndimage
import json
import time
import skimage.morphology as morph


def download_data(input_file, data_path, max_files=10, start_idx=0, delay=10):
    """
    :param delay: delay between downloads in seconds
    :param start_idx: helps to recover if downloading stuck
    :param max_files: max files to download
    :param input_file: text file with links to download data
    :param data_path: where to save the data
    :return: None
    """

    # create this bar_progress method which is invoked automatically from wget
    # built in progress bar from wget doesn't work in local env...
    def bar_progress(current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    # create data_path if needed
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # change current path to download
    os.chdir(data_path)
    with open(input_file, 'r') as f:
        # start downloading!
        i = 1
        for line in f.readlines():
            if i < start_idx:
                i += 1
                continue
            if i > max_files:
                print(f"Reached max {max_files}")
                break
            print(f'{i}. ' + line)
            start_stamp = time.time()
            wget.download(line.replace('\n', ''), data_path, bar=bar_progress)
            print(f"\n...took {time.time() - start_stamp} sec")
            print(f'\n...Sleep{delay} sec...')
            time.sleep(delay)
            i += 1


def unzip_data(data_path):
    """
    :param data_path: directory with zip files
    :return: None
    """
    all_zips = sorted(os.listdir(data_path))
    os.chdir(data_path)
    for archive in all_zips:
        if archive.startswith(".") or not archive.endswith(".zip"):
            print(f"...ignore {archive}")
            continue    # ignore hidden files
        archive_path = os.path.join(data_path, archive)
        print(archive)
        try:
            with ZipFile(archive_path, 'r') as zip_obj:
                zip_obj.extractall()
        except BadZipFile as ex:
            print(f"Skipping {archive} with exception: {ex}")
            continue
        # delete file after unzip
        os.remove(archive_path)


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
    df = pd.DataFrame(data_fields)
    return df


def filter_data_fields(df, max_slice_thickness=2, min_height_mm=100):
    """
    Filter out scans that are not detailed enough
    :return: pandas dataframe
    """
    return df[(df['SliceThickness'] < max_slice_thickness) & (df['SliceThickness'] * df['num_slices'] > min_height_mm)]


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
        df.loc[i, 'file_path'] = file_path
        np.savez(file_path, I=img3d)
        # add original shape to df
        df.loc[i, 'orig_shape'] = str(img3d.shape)
        # now we don't need original slices files anymore
        shutil.rmtree(dir_path)
    # cleanup folders
    for name in os.listdir(data_path):
        path = os.path.join(data_path, name)
        if not name.endswith(".npz") and os.path.isdir(path):
            shutil.rmtree(path)
    df.to_csv(os.path.join(data_path, "data_fields.csv"))


def remove_bad_npz(df, data_path):
    """
    If need to filter files after they are saved to .npz
    Remove all files that are not in DF
    :param df: DataFrame with good images
    :param data_path: working dir
    :return: None
    """
    for name in os.listdir(data_path):
        if not name.endswith(".npz"):
            continue
        if get_data_fields(name, df) is None:
            file_path = os.path.join(data_path, name)
            print(f"{name} is not in DF. removing...")
            os.remove(file_path)


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


def resample_dir(data_path, csv_name="data_fields.csv", pixel_spacing_mm=(1, 1, 1)):
    """
    WARNING: Result will overwrite files in input dir
    Resample to new scale. New scale is provided in mm.
    :param data_path: dir with .npz 3d images
    :param csv_name: name of the csv file with images properties
    :param pixel_spacing_mm: new pixel spacing along 3 axes in mm (new distance between pixels in mm along x, y, z)
    :return: None
    """
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
        new_image = resample(image, data_fields, pixel_spacing_mm)
        np.savez(file_path, I=new_image)  # write to the same file
        # update DF
        idx = get_idx_by_name(name)
        df.loc[idx, 'shape_after_resampling'] = str(new_image.shape)
        df.loc[idx, 'new_pixel_spacing'] = str(pixel_spacing_mm)
    # save df with new fields
    df.to_csv(os.path.join(data_path, "data_fields.csv"))


def calculate_new_pixel_size(df, desired_arr_shape=(320, 320, 20)):
    """
    Will use new pixel size to resample images as close as possible to desired shape
    :param df: Data Frame with CT images metadata
    :param desired_arr_shape: final image shape that we want to achieve
    :return: (x, y, z) tuple with pixel spacing (pixel size) in mm
    """
    # parse original shape
    tmp = df['orig_shape'].str.split(',', expand=True)
    shape_x = tmp[0].str.replace('[^0-9.]*', '', regex=True).astype(float)
    shape_y = tmp[1].str.replace('[^0-9.]*', '', regex=True).astype(float)
    shape_z = tmp[2].str.replace('[^0-9.]*', '', regex=True).astype(float)

    # parse current pixel spacing for x and y
    tmp = df['PixelSpacing'].str.split(',', expand=True)
    pixel_spacing_x = tmp[0].str.replace('[^0-9.]*', '', regex=True).astype(float)
    pixel_spacing_y = tmp[1].str.replace('[^0-9.]*', '', regex=True).astype(float)

    # get size in mm
    df['size_x_mm'] = shape_x * pixel_spacing_x
    df['size_y_mm'] = shape_y * pixel_spacing_y
    df['size_z_mm'] = shape_z * df['SliceThickness']

    # divide by desired shape and return mean
    x = (df['size_x_mm'] / desired_arr_shape[0]).mean()
    y = (df['size_y_mm'] / desired_arr_shape[1]).mean()
    z = (df['size_z_mm'] / desired_arr_shape[2]).mean()
    return round(x, 6), round(y, 6), round(z, 6)


def adjust_size(img, new_shape=(320, 320, 20), default_value=-1000):
    """
    Achieve desired shape using crop and padding
    :param default_value: will use this value for padding
    :param img: numpy 3d array
    :param new_shape: tuple of sizes in px (x, y, z)
    :return: numpy array with new shape
    """
    # first crop (if needed)
    margin0 = [0, 0, 0]   # lower margin
    margin1 = [0, 0, 0]   # upper margin - it can be different if size diff is odd
    for i in range(3):
        margin0[i] = (img.shape[i] - new_shape[i]) // 2 if img.shape[i] >= new_shape[i] else 0
        margin1[i] = img.shape[i] - new_shape[i] - margin0[i] if img.shape[i] >= new_shape[i] else 0
    img1 = img[margin0[0]: img.shape[0] - margin1[0], margin0[1]: img.shape[1] - margin1[1], margin0[2]: img.shape[2] - margin1[2]]
    # now add padding if needed
    for i in range(3):
        margin0[i] = (new_shape[i] - img.shape[i]) // 2 if img.shape[i] < new_shape[i] else 0
        margin1[i] = new_shape[i] - img.shape[i] - margin0[i] if img.shape[i] < new_shape[i] else 0
    res = np.ones(new_shape) * default_value
    res[margin0[0]: res.shape[0] - margin1[0], margin0[1]: res.shape[1] - margin1[1], margin0[2]: res.shape[2] - margin1[2]] = img1
    return res


def adjust_size_dir(data_path, new_shape=(320, 320, 20)):
    """
    Achieve desired shape using crop and padding on all files in dir
    :param data_path: target dir
    :param new_shape: tuple of sizes in px (x, y, z)
    :return: None
    """
    for name in os.listdir(data_path):
        file_path = os.path.join(data_path, name)
        if not name.endswith(".npz"):
            continue
        print(f"adjust_size {file_path}")
        image = np.load(file_path)['I']
        new_image = adjust_size(image, new_shape)
        np.savez(file_path, I=new_image)   # write to the same file


def get_data_fields(file_name, df):
    idx = get_idx_by_name(file_name)
    try:
        res = df.loc[idx]
    except KeyError as ex:
        print(f"KeyError on index {idx}")
        return None
    return res


def get_idx_by_name(file_name):
    try:
        idx = int(file_name.split('.')[0].split('_')[-1])
    except ValueError:
        print(f"Bad file name: {file_name}")
        return None
    return idx


def get_image_by_id(df, idx, data_path):
    file_name = f"{df.loc[idx]['PatientID']}_{idx}.npz"
    return np.load(os.path.join(data_path, file_name))['I']


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