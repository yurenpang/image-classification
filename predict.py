
import numpy as np
from keras.models import load_model

model = load_model('my_model.h5')

from keras.preprocessing import image
test_image = image.load_img('./intel-image-classification/seg_pred/21820.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

category = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

for i in range(6):
    if result[0][i] > 0.5:
        print(category[i])