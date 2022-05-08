#!/usr/bin/env python
# coding: utf-8

import dicom_utils as dcm
from data_loader import LOG_LEVELS, set_log_level

import os
import pandas as pd
import numpy as np
import argparse
import sys
import logging


def main(data_path, input_file, log_level, seed):
    np.random.seed(seed)
    set_log_level(log_level)

    # dicom_utils.download_data(input_file, data_path, max_files=100)
    dcm.unzip_data(data_path)

    # extract metadata
    df = dcm.extract_scans_info(data_path)
    logging.info(f"Overall images: {len(df)}")
    # filter
    df = dcm.filter_data_fields(df)
    logging.info(f"...After size filtering: {len(df)}")
    # remove upside down images that fail on resampling
    df = df[~((df['SliceThickness'] * df['num_slices'] > 250) & (df['ScanOptions'] == 'HELICAL'))]
    logging.info(f"...After special case filtering: {len(df)}")
    dcm.save_3d_images(df, data_path)

    # delete unnecessary files
    dcm.clean_up(data_path)

    # to HU values
    dcm.transform_to_hu_dir(data_path)

    # more filtering!
    df = pd.read_csv(os.path.join(data_path, "data_fields.csv"), index_col=[0])
    for f in os.listdir(data_path):
        if not f.startswith("CQ") or not f.endswith(".npz"):
            continue
        path = os.path.join(data_path, f)
        image = np.load(path)['I']
        logging.info(f"{f}   {image.shape}")
        df.loc[int(f.split('.')[0].split('_')[-1]), 'last_slice_mean'] = image[:,:,-1].mean()
    # this will filter out "upside down" images
    df = df[df['last_slice_mean'] < -800]
    dcm.remove_bad_npz(df, data_path)
    df.to_csv(os.path.join(data_path, "data_fields.csv"))

    # remove noise from images
    dcm.clean_noise_dir(data_path)

    # resample closer to required pixel spacing
    new_pixel_spacing = dcm.calculate_new_pixel_size(df, desired_arr_shape=(320, 320, 20))
    logging.info(new_pixel_spacing)
    dcm.resample_dir(data_path, pixel_spacing_mm=new_pixel_spacing)

    # adjust to required size
    dcm.adjust_size_dir(data_path, new_shape=(320, 320, 20))


def get_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--data-path', required=True, type=str, help='Working directory')
    parser.add_argument('--input-file', required=False, type=str, default="./YdataDataset.txt", help='File with links to data')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--log-level', default="info", choices=LOG_LEVELS.keys(), help='Logging level, default "info"')
    return vars(parser.parse_args())


if '__main__' == __name__:
    args = get_args()
    print(args)
    main(**args)
