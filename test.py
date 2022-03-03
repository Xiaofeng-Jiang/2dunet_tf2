import numpy as np
from generator import generator

root = '/Users/jiangxiaofeng/Downloads/Compressed/stanford/PRE/'
X_train, Y_train = generator(root, l_end=10)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)

# np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data/img.npy', X_train)
# np.save('/Users/jiangxiaofeng/Downloads/Compressed/stanford/unet/data/mask.npy', Y_train)