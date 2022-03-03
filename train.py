import os
from random import random

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.transform import resize

from network import unet
from dataloader import train_dataloader, test_dataloder
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/Users/jiangxiaofeng/Downloads/Compressed/kaggle/unet/stage1_train/'
TEST_PATH = '/Users/jiangxiaofeng/Downloads/Compressed/kaggle/unet/stage1_test/'
FINAL_TEST_PATH = '/Users/jiangxiaofeng/Downloads/Compressed/kaggle/unet/stage2_test_final/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
final_test_ids = next(os.walk(FINAL_TEST_PATH))[1]

X_train, Y_train = train_dataloader(TRAIN_PATH, train_ids, IMG_CHANNELS=3, IMG_HEIGHT=512, IMG_WIDTH=512)
X_test, sizes_test = test_dataloder(TEST_PATH, test_ids, IMG_CHANNELS=3, IMG_HEIGHT=512, IMG_WIDTH=512)

model = unet(512, 512, 3)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

earlystopper = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint('model_unet_checkpoint.h5', verbose=1, save_best_only=True, save_weights_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=5,
                    callbacks=[earlystopper, checkpointer])


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test_t)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


ix = random.randint(0, len(preds_test_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()