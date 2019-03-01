import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
from keras.utils import Sequence

print("Getting training and validation data...")
#training set
train = pd.read_csv("../training_set.csv")
#split train and validation data
val = train[22000:].reset_index(drop=True)
train = train[:22000]

data_gen = ImageDataGenerator(rescale=1/255.)
val_gen = data_gen.flow_from_dataframe(val,
                                         "../images",
                                         x_col="image_name", y_col=['x1', 'x2', 'y1', 'y2'],
                                         target_size=(120,160), class_mode='other',
                                         batch_size=128)

class train_aug(Sequence):
    def __init__(self, target_size=(120,160), batch_size=128):
        self.train_gen = data_gen.flow_from_dataframe(train,
                                         "../images",
                                         x_col="image_name", y_col=['x1', 'x2', 'y1', 'y2'],
                                         target_size=target_size, class_mode='other',
                                         batch_size=batch_size//8)
        self.n = self.train_gen.n
        self.batch_size = batch_size//8
        self._rot90 = iaa.Rot90(k=3)
    def reset(self):
        self.train_gen.reset()
    def on_epoch_end(self):
        self.train_gen.reset()

    def __len__(self):
        return self.n//self.batch_size+1

    def get_rotated(self, imgs, y):
        imgr = self._rot90.augment_images(imgs)
        yr = np.empty(y.shape)
        yr[:,0] = np.round(y[:,2]*4/3)
        yr[:,1] = np.round(y[:,3]*4/3)
        yr[:,2] = np.round((1-y[:,1]/640.)*480)
        yr[:,3] = np.round((1-y[:,0]/640.)*480)
        return imgr, yr

    def __data_generation(self):
        X1, y1 = self.train_gen.next()
        bs = y1.shape[0]
        X = np.empty((bs*8,X1.shape[1], X1.shape[2], X1.shape[3]))
        y = np.empty((bs*8, y1.shape[1]))
        X[:bs], y[:bs] = X1, y1
        #rotate by 90
        X[bs:bs*2], y[bs:bs*2] = self.get_rotated(X1, y1)
        #hor-flip
        X[bs*2:bs*3] = X1[:, :, ::-1, :]
        y[bs*2:bs*3,0] = 640-y1[:, 1]
        y[bs*2:bs*3,1] = 640-y1[:, 0]
        y[bs*2:bs*3,2] = y1[:, 2]
        y[bs*2:bs*3,3] = y1[:, 3]
        #rotate by 90
        X[bs*3:bs*4], y[bs*3:bs*4] = self.get_rotated(X[bs*2:bs*3], y[bs*2:bs*3])
        #vert-flip
        X[bs*4:bs*5] = X1[:, ::-1, :, :]
        y[bs*4:bs*5,0] = y1[:, 0]
        y[bs*4:bs*5,1] = y1[:, 1]
        y[bs*4:bs*5,2] = 480-y1[:, 3]
        y[bs*4:bs*5,3] = 480-y1[:, 2]
        #rotate by 90
        X[bs*5:bs*6], y[bs*5:bs*6] = self.get_rotated(X[bs*4:bs*5], y[bs*4:bs*5])
        #hor-vert-flip
        X[bs*6:bs*7] = X1[:, ::-1, ::-1, :]
        y[bs*6:bs*7,0] = 640-y1[:, 1]
        y[bs*6:bs*7,1] = 640-y1[:, 0]
        y[bs*6:bs*7,2] = 480-y1[:, 3]
        y[bs*6:bs*7,3] = 480-y1[:, 2]
        #rotate by 90
        X[bs*7:bs*8], y[bs*7:bs*8] = self.get_rotated(X[bs*6:bs*7], y[bs*6:bs*7])
        return X, y

    def __getitem__(self, index):
        return self.__data_generation()

train_gen = train_aug()
