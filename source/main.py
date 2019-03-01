"""
This code assumes that 'training_set.csv', 'test.csv' and 'images' folder
are located one-directory-up relative to the current directory.
Submission file generated as 'submit.csv' will be saved in the current directory.
"""

import numpy as np
import pandas as pd
from keras.preprocessing import image

from modeltrain import model

test = pd.read_csv("../test.csv")
l = list(test['image_name'])

print("Generating test predictions...")
ans = np.zeros((test.shape[0], 4))
cnt = 0
for i in l:
    img = image.img_to_array(image.load_img(
        "../images/"+i,
                        target_size=(120,160)))/255.
    pred = model.predict(img.reshape((1, 120, 160, 3)))
    ans[cnt] = pred
    cnt+=1
    if cnt%1000==0: print(cnt)

ans[:, (0, 1)] = np.round(ans[:, (0, 1)]).astype('int')
ans[:, (2, 3)] = np.round(ans[:, (2, 3)]).astype('int')
test['x1'] = ans[:, 0].astype(int)
test['x2'] = ans[:, 1].astype(int)
test['y1'] = ans[:, 2].astype(int)
test['y2'] = ans[:, 3].astype(int)
test.x1 = test.x1.clip_lower(1)
test.x2 = test.x2.clip_upper(640)
test.y1 = test.y1.clip_lower(1)
test.y2 = test.y2.clip_upper(480)

test.to_csv('submit.csv', index=False, index_label=False)
