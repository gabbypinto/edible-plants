import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing import image
from keras import models
from keras import layers



base_dir = '/Users/katherinehansen/CPSC393/edible-plants/allplants' 
edibleTrain = os.path.join(base_dir, 'edible_plants/train')
poisonousTrain = os.path.join(base_dir, 'poisonous_plants/train')

# edibleTest = os.path.join(base_dir, 'edible_plants/test')
# poisonousTest = os.path.join(base_dir, 'poisonous_plants/test')

test = os.path.join(base_dir, 'test')

edibleValidation = os.path.join(base_dir, 'edible_plants/validation')
poisonousValidation = os.path.join(base_dir, 'poisonous_plants/validation')



datagen = ImageDataGenerator(rescale=1./255) 
batch_size = 20

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

