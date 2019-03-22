DATA PREPROCESSING:
- downscaled the images (480 x 640) to 1/4th the original dimensions i.e. 120 x 160
- normalised the images in to range (0, 1) by multiplication with 1/255.

DATA AUGMENTATION:
- generated flipped image data for original image data
- three types of flips were applied: horizontal, vertical and horizontal-vertical(both)
- for each these four, generated corresponding clockwise 90-deg rotated versions.
- thus, we now had 8x training examples i.e. 176,000 (22,000 x 8) in total

MODEL ARCHITECTURE:
- used the following deep neural-net architecture :

(ReLU activation used in all Conv2D and Dense layers)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
InputLayer               (None, 120, 160, 3)            0
_________________________________________________________________
Conv2D(3x3)              (None, 120, 160, 32)           896
_________________________________________________________________
MaxPooling2D(2x2, s=2)   (None, 60, 80, 32)             0
_________________________________________________________________
Conv2D(3x3)              (None, 60, 80, 64)             18496
_________________________________________________________________
MaxPooling2D(2x2, s=2)   (None, 30, 40, 64)             0
_________________________________________________________________
Conv2D(3x3)              (None, 30, 40, 128)            73856
_________________________________________________________________
MaxPooling2D(2x2, s=2)   (None, 15, 20, 128)            0
_________________________________________________________________
Conv2D(3x3)              (None, 15, 20, 256)            295168
_________________________________________________________________
MaxPooling2D(2x2, s=2)   (None, 7, 10, 256)             0
_________________________________________________________________
Conv2D(3x3)              (None, 7, 10, 512)             1180160
_________________________________________________________________
MaxPooling2D(2x2, s=2)   (None, 3, 5, 512)              0
_________________________________________________________________
Conv2D(3x3)              (None, 3, 5, 1024)             4719616
_________________________________________________________________
GlobalMaxPooling2D       (None, 1024)                    0
_________________________________________________________________
Dropout(0.2)             (None, 1024)                    0
_________________________________________________________________
Dense(2048)              (None, 4096)                   4198400
_________________________________________________________________
Dropout(0.4)             (None, 4096)                   0
_________________________________________________________________
Dense(2048)              (None, 2048)                   8390656
_________________________________________________________________
Dense(4, output)         (None, 4)                      8196
=================================================================
Total params: 18,885,444
Trainable params: 18,885,444
Non-trainable params: 0
_________________________________________________________________

TRAINING:
- Training Strategy:
  - model trained for 100 epochs over training set (original + augmented) of 176,000
  - validated over 2000 images (original, only unaugmented)
  - early stopping, if validation loss is less than 450
  - reduced learning rate to 1/10th on plateaus i.e. where val_loss change was less than 30 for two epochs
  - used a batch size of 128

- System Specifications :
  - Hardware :
    - GPU : Nvidia TeslaK80
    - Memory Size : 14 GB
  - Software :
    - Python 3.6.7
    - Keras (with Tensorflow Backend)
    - NumPy (linear algebra)
    - Pandas (data processing, CSV I/O)
    - imgaug library (for Image Augmentation)

RESULT:
- In some cases, output comes out to be floating points more than the image size
- So, rounded off and clipped the output to integers in the range
  - (1, 640) for x, and
  - (1, 480) for y
