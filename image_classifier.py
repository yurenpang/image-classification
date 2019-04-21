from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64,64,3), activation="relu"))

# Pooling using MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2)) # Dropout

# Flatten
classifier.add(Flatten())

# Compile the CNN
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units = 6, activation="softmax"))

classifier.compile(optimizer='adam', loss = "categorical_crossentropy", metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('./intel-image-classification/seg_train', target_size=(64,64), batch_size=32,class_mode="categorical")

test_set = train_datagen.flow_from_directory("./intel-image-classification/seg_test", target_size=(64,64), batch_size=32, class_mode="categorical")

classifier.fit_generator(
    training_set, steps_per_epoch=8000, epochs=10,validation_data=test_set, validation_steps=800
)

classifier.save("my_model.h5")
