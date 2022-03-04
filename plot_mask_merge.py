import numpy as np
from generator import generator
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

root = '/Users/jiangxiaofeng/Downloads/Compressed/stanford/PRE/'
X_train, Y_train, id_list = generator(root, l_start=0, l_end=250)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)


np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data3/img_128.npy', X_train)
np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data3/mask_128.npy', Y_train)

X_test, Y_test, id_list = generator(root, l_start=250, l_end=300)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_test.shape)
print(Y_test.shape)

np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data3/img_test_128.npy', X_test)
np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data3/mask_test_128.npy', Y_test)

def plot_merge(X_train, Y_train, id_list):
    for i in tqdm(range(len(X_train))):
        mri_gray = X_train[i, ...]
        mri_gray = (mri_gray / (np.max(mri_gray) - np.min(mri_gray))) * 255  # [0, 255]
        mri_gray = mri_gray.astype('uint8')

        mri_rgb = np.concatenate([mri_gray] * 3, axis=-1)

        mask_gray = Y_train[i, ...] * 255
        mask_gray = mask_gray.astype('uint8')
        mask_rgb = np.concatenate([mask_gray] * 3, axis=-1)

        mask_red = np.concatenate([mask_gray, np.zeros((128, 128, 1)), np.zeros((128, 128, 1))], axis=-1).astype(
            'uint8')

        img_add = cv2.addWeighted(mri_rgb, 1, mask_rgb, 0.4, 0)
        img_add_red = cv2.addWeighted(mri_rgb, 1, mask_red, 0.2, 0)

        merge = np.concatenate((mri_rgb, img_add_red), axis=1)
        a = cv2.resize(merge, [merge.shape[1] * 4, merge.shape[0] * 4])
        im = Image.fromarray(a)
        im.save('./merge_output/' + id_list[i] + '_' + str(i) + ".jpg", quality=95, subsampling=0)


# if __name__ == '__main__':
#     plot_merge(X_train, Y_train, id_list)

