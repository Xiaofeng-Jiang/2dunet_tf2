import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm


def train_dataloader(TRAIN_PATH, train_ids, dir_path='', IMG_CHANNELS=3, IMG_HEIGHT=128, IMG_WIDTH=128):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

        # Read image files iteratively
        path = TRAIN_PATH + id_
        img = imread(dir_path + path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        # Append image to numpy array for train dataset
        X_train[n] = img

        # Read corresponding mask files iteratively
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

        # Looping through masks
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            # Read individual masks
            mask_ = imread(dir_path + path + '/masks/' + mask_file)

            # Expand individual mask dimensions
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)

            # Overlay individual masks to create a final mask for corresponding image
            mask = np.maximum(mask, mask_)

        # Append mask to numpy array for train dataset
        Y_train[n] = mask

        return X_train, Y_train


def test_dataloder(TEST_PATH, test_ids, dir_path='', IMG_CHANNELS=3, IMG_HEIGHT=128, IMG_WIDTH=128):
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_

        # Read images iteratively
        img = imread(dir_path + path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]

        # Get test size
        sizes_test.append([img.shape[0], img.shape[1]])

        # Resize image to match training data
        # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        # Append image to numpy array for test dataset
        X_test[n] = img

        return X_test, sizes_test
