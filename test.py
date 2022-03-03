import numpy as np
from generator import generator

root = '/Users/jiangxiaofeng/Downloads/Compressed/stanford/PRE/'
X_train, Y_train = generator(root, l_start=0, l_end=200)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)

np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data2/img_128.npy', X_train)
np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data2/mask_128.npy', Y_train)

X_test, Y_test = generator(root, l_start=200, l_end=250)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_test.shape)
print(Y_test.shape)
#
np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data2/img_test_128.npy', X_test)
np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data2/mask_test_128.npy', Y_test)
