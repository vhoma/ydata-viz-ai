import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
from skimage.transform import resize


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

    return {
        'data': img3d,
        'aspect': [cor_aspect, sag_aspect, ax_aspect]
    }


def ct_image_filter(img3d, val_from, val_to, default_value=-1000):
    #res = np.copy(img3d['data'])
    #res[np.where(img3d['data'] < val_from)] = default_value
    #res[np.where(img3d['data'] > val_to)] = default_value
    res = np.clip(img3d['data'], val_from, val_to)
    return {'data': res, 'aspect': img3d['aspect']}


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