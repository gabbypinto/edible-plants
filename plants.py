import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD

from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import numpy as np
from keras.regularizers import l2


labels = ['edible_plants','poisonous_plants']
img_size =224

def getData(base_dir):
    data = []
    for label in labels:
        path = os.path.join(base_dir, label)
        class_num = labels.index(label)
        if label == 'edible_plants' and base_dir == r'C:\Users\gabri\Documents\edible-plants\allplants\validation':
            for img in os.listdir(path):
                try:
                    imgArr = cv2.imread(os.path.join(path,img))[...,::-1] #convert BGR to RGB format
                    resizedArr = cv2.resize(imgArr,(img_size,img_size)) #reshaping images to preferred size
                    data.append([resizedArr,class_num])
                except Exception as e:
                    print(e)
        else:
            for folder in os.listdir(path):
                # print(folder)
                current_path = path+"\\"+folder
                for img in os.listdir(current_path):
                    # print(img)
                    try:
                        imgArr = cv2.imread(os.path.join(current_path,img))[...,::-1] #convert BGR to RGB format
                        resizedArr = cv2.resize(imgArr,(img_size,img_size)) #reshaping images to preferred size
                        data.append([resizedArr,class_num])
                    except Exception as e:
                        print(e)
    return np.array(data,dtype=object)
train_data_path = r'C:\Users\gabri\Documents\edible-plants\allplants\train'
train = getData(r'C:\Users\gabri\Documents\edible-plants\allplants\train')
test = getData(r'C:\Users\gabri\Documents\edible-plants\allplants\test')
val = getData(r'C:\Users\gabri\Documents\edible-plants\allplants\validation')

xTrain = []
yTrain = []
xVal = []
yVal = []

for f,l in train:
    xTrain.append(f)
    yTrain.append(l)

for f,l in val:
    xVal.append(f)
    yVal.append(l)

xTrain = np.array(xTrain)/255
xVal = np.array(xVal)/255

xTrain.reshape(-1,img_size,img_size,1)
yTrain = np.array(yTrain)

xVal.reshape(-1,img_size,img_size,1)
yVal=np.array(yVal)

batch_size = 30

datagen = ImageDataGenerator(
featurewise_center=False,
samplewise_center=False,
featurewise_std_normalization=False,
samplewise_std_normalization=False,
zca_whitening=False,
rotation_range=30,
zoom_range = 0.2,
width_shift_range=0.1,
height_shift_range=0.1,
horizontal_flip=True,
vertical_flip=False
)

train_generator = datagen.flow_from_directory(
train_data_path,
target_size = (224,224),
batch_size = batch_size,
class_mode = 'categorical',
subset = 'training',
shuffle=True
)
train_generator.fit(xTrain);
# datagen.fit(xTrain)

model = Sequential()
model.add(Conv2D(32,3,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,3,padding="same",activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,3,padding="same",activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
# model.add(Dense(2,activation="softmax"))
model.add(Dense(2, activation='sigmoid'))

# model.summary()

# opt = Adam(lr=0.00001)
opt = SGD(lr=0.01)
model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = model.fit(xTrain,yTrain,epochs = 10, validation_data = (xVal,yVal))

acc = history.history['accuracy']
val_acc = history.history['val_acc']
loss=history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.plot(epochs_range,acc,label="training accuracy")
plt.plot(epochs_range,val_acc,label="validation accuracy")
plt.legend(loc='lower right')
plt.title('training vs validation accuracy')

plt.subplot(2,2,2)
plt.plot(epochs_range,acc,label="training loss")
plt.plot(epochs_range,val_acc,label="validation loss")
plt.legend(loc='upper right')
plt.title('training vs validation loss')
plt.show()
