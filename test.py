import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from generator import generator
from network import unet

root = '/Users/jiangxiaofeng/Downloads/Compressed/stanford/PRE/'
X_train, Y_train = generator(root, l_end=10)

X_train = np.array(X_train)
Y_train = np.array(X_train)

model = unet(512, 512, 1)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

earlystopper = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint('model_unet_checkpoint.h5', verbose=1, save_best_only=True, save_weights_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=5,
                    callbacks=[earlystopper, checkpointer])
