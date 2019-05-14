"""
Author: Yuren "Rock" Pang, Jufeng Yan
Deep learning model for multi-class image classification task
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from numpy.random import seed
from tensorflow import set_random_seed
import matplotlib.pyplot as plt

# set seeds for numpy random and tensorflow backend
seed(1)
set_random_seed(3)

# Initialize the classifier
classifier = Sequential()

# Convolution, with 3*3 filter
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Pooling using MaxPooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
classifier.add(Flatten())

# Compile the CNN with 2 hidden layers
# use softmax for multiple classes
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(64, activation="relu"))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units=6, activation="softmax"))

# Compile the classifier with the loss function: cross-entropy and accuracy metric
classifier.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Get the training set and test set from correct directory
training_set = train_datagen.flow_from_directory('../intel-image-classification/seg_train', target_size=(64,64), batch_size=32,class_mode="categorical")
test_set = train_datagen.flow_from_directory("../intel-image-classification/seg_test", target_size=(64,64), batch_size=32, class_mode="categorical")

# Train the model, return a History object
history = classifier.fit_generator(
    training_set, steps_per_epoch=1000, epochs=60, validation_data=test_set, validation_steps=800
)
# Store the model which takes hours to run
classifier.save("my_model.h5")

# Plot the accuracy and loss function
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

