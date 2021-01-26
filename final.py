import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing import image
from keras import models
from keras import layers
from keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16
from keras.layers import GaussianNoise
from keras.regularizers import l2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
import cv2

tf.compat.v1.disable_eager_execution()

#instantiate the VGG16 model
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

base_dir = r'C:\Users\gabri\Documents\edible-plants\allplants'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


#extracting features/grabbing features and labels
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        else:
            print(i * batch_size , sample_count)
    return features, labels

#feature and labels
train_features, train_labels = extract_features(train_dir, 6879)
validation_features, validation_labels = extract_features(validation_dir, 358)
test_features, test_labels = extract_features(test_dir, 416)

#reshaping
train_features = np.reshape(train_features, (6879, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (358, 4 * 4 * 512))
test_features = np.reshape(test_features, (416, 4 * 4 * 512))

# constructing our model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu', activity_regularizer=l2(0.001)))
model.add(layers.Dense(1, activation='sigmoid'))


#freezing convolutional base
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#compiling and fitting the model
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(train_features, train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels))

#test accuracy
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(150, 150),
batch_size=20,
class_mode='binary')
score = model.evaluate(test_features,test_labels,batch_size=20)
print('test acc:', score[1])

#generate heat map
def heatMapPlot(img_path,final_path):
    img = image.load_img(img_path, target_size=(224, 224))
    model = VGG16(weights='imagenet')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    pic_output = model.output[:, np.argmax(preds[0])]
    last_conv_layer = model.get_layer('block5_conv3')

    grads = K.gradients(pic_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(final_path, superimposed_img)
    plt.clf()

#edible plant
heatMapPlot(r"C:\Users\gabri\Documents\edible-plants\allplants\train\edible_plants\train\Dandellion\510677438_73e4b91c95_m.jpg",r"C:\Users\gabri\Documents\edible-plants\heatmaps\dandelion.jpg")
#non edible plant
heatMapPlot(r"C:\Users\gabri\Documents\edible-plants\allplants\train\poisonous_plants\train\lilies\1f340c5e9c379ba637ea6b51476c3d29--lilies-flowers-the-flowers.jpg",r"C:\Users\gabri\Documents\edible-plants\heatmaps\lilies.jpg")

#plots - loss and accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
