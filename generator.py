import cv2
import pandas as pd
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted


def generator(mri_root, l_start=0, l_end=10):
    X_train = []
    Y_train = []
    id_list = []
    # single_ori_root = '/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/single_ori'
    # single_mask_root = '/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/single_mask'
    #
    # if not os.path.exists(single_ori_root):
    #     os.mkdir(single_ori_root)
    # if not os.path.exists(single_mask_root):
    #     os.mkdir(single_mask_root)

    p_list = list(os.listdir(mri_root))
    if '.DS_Store' in p_list:
        p_list.remove('.DS_Store')

    remove_list = ['pre-A54', 'pre-B262', 'pre-B104', 'pre-B308', 'pre-C9']  # which mask is wrong
    for rmf in remove_list:
        p_list.remove(rmf)

    for i, id in enumerate(p_list[l_start:l_end]):
        p_path = os.path.join(mri_root, id)
        mask_file = os.path.join(p_path, 'N' + id + '.nii')
        mask = sitk.ReadImage(mask_file)
        mask_array = sitk.GetArrayFromImage(mask)

        mri_path = os.path.join(p_path, 'T2WI')
        # print(id)
        mri_list = sorted(os.listdir(mri_path), key=lambda s: int(s.split('-')[1].split('.')[0]))

        for h in range(mask_array.shape[0]):
            if mask_array[h, ...].sum() > 100:
                mask_single_array = mask_array[h, :, :]
                mri_file = os.path.join(mri_path, mri_list[-h - 1])
                mri = sitk.ReadImage(mri_file)
                mri_array = sitk.GetArrayFromImage(mri)
                mri_single_array = mri_array[0, ...]

                mask_1 = cv2.resize(mask_single_array, (128, 128), interpolation=cv2.INTER_CUBIC)
                mri_1 = cv2.resize(mri_single_array, (128, 128), interpolation=cv2.INTER_CUBIC)

                X_train.append(np.expand_dims(mri_1, axis=-1))
                Y_train.append(np.expand_dims(mask_1, axis=-1))
                id_list.append(id)

    return X_train, Y_train, id_list
