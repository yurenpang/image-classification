"""
Author: Yuren "Rock" Pang
Predict if a picture is in a particular class
"""
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model('third_model.h5')

test_image = image.load_img('../intel-image-classification/seg_pred/seg_pred/23842.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

category = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

for i in range(6):
    if result[0][i] > 0.5:
        print(category[i])
