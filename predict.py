import os

from skimage.transform import resize
from tensorflow.python.keras.models import load_model
from dataloader import train_dataloader, test_dataloder
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/Users/jiangxiaofeng/Downloads/Compressed/kaggle/unet/stage1_train/'
TEST_PATH = '/Users/jiangxiaofeng/Downloads/Compressed/kaggle/unet/stage1_test/'
FINAL_TEST_PATH = '/Users/jiangxiaofeng/Downloads/Compressed/kaggle/unet/stage2_test_final/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
final_test_ids = next(os.walk(FINAL_TEST_PATH))[1]

X_train, Y_train = train_dataloader(TRAIN_PATH, train_ids)
X_test, sizes_test = test_dataloder(TEST_PATH, test_ids)

model = load_model('model_unet_checkpoint.h5')
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