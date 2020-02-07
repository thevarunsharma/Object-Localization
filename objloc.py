import numpy as np
from tensorflow.compat.v1.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.misc import imresize, imsave

model = load_model("model.h5")
model._make_predict_function()

def generate_box_img(img, x1, x2, y1, y2):
    img = img/255.
    lw = max(round(max(img.shape)/640)*2, 2)
    img[y1:y1+lw, x1:x2+1] = [1, 0, 0]
    img[y2-lw+1:y2+1, x1:x2+1] = [1, 0, 0]
    img[y1:y2+1, x1:x1+lw] = [1, 0, 0]
    img[y1:y2+1, x2-lw+1:x2+1] = [1, 0, 0]
    imsave("./static/result.jpg", img)

def get_coords(img):
    img_new = imresize(img, (120, 160, 3))/255.
    dims = img.shape
    preds = model.predict(img_new.reshape(1, *img_new.shape))
    coords = np.round(preds[0]).astype('int')
    x1, x2, y1, y2 = coords
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, 639)
    y2 = min(y2, 479)
    x1 , x2 = x1*dims[1]//640, x2*dims[1]//640
    y1 , y2 = y1*dims[0]//480, y2*dims[0]//480
    generate_box_img(img, x1, x2, y1, y2)
    return x1, x2, y1, y2
