#uses Python3
import numpy as np

#All Images have dimensions 480 x 640
from keras import layers as L
from keras.models import Model, load_model

from preprocess import train_gen, val_gen

def objloc():
    Xinp = L.Input(shape=(120, 160, 3))
    X = L.Conv2D(32, 3, padding='same', activation='relu')(Xinp)
    X = L.MaxPooling2D()(X)   #60, 80
    X = L.Conv2D(64, 3, padding='same', activation='relu')(X)
    X = L.MaxPooling2D()(X)   #30, 40
    X = L.Conv2D(128, 3, padding='same', activation='relu')(X)
    X = L.MaxPooling2D()(X)   #15, 20
    X = L.Conv2D(256, 3, padding='same', activation='relu')(X)
    X = L.MaxPooling2D()(X)   #7, 10
    X = L.Conv2D(512, 3, padding='same', activation='relu')(X)
    X = L.MaxPooling2D()(X)   #3, 5
    X = L.Conv2D(1024, 3, padding='same', activation='relu')(X)
    X = L.GlobalMaxPool2D()(X)
    X = L.Dropout(0.2)(X)
    X = L.Dense(4096, activation='relu')(X)
    X = L.Dropout(0.4)(X)
    X = L.Dense(2048, activation='relu')(X)
    X = L.Dense(4, activation='relu')(X)
    model = Model(inputs=Xinp, outputs=X)
    return model

model = objloc()
model.compile('adam', loss='mean_squared_error', metrics=['mae'])
print("Model Architecture:")
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=450, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

callbacks = [
        EarlyStoppingByLossVal(verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=2, min_delta=30),
        # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=0)
]
print("Training model...")
model.fit_generator(train_gen, steps_per_epoch=train_gen.n//train_gen.batch_size+1, epochs=100,
                    validation_data=val_gen, validation_steps=val_gen.n//val_gen.batch_size+1,
                   workers=4, callbacks=callbacks)
#open model with best loss
model = load_model("model.h5")
model.save('model.h5')
