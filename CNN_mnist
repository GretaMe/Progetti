from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
#from tensorflow.keras.models import model_from_json
#from tensorflow.keras.models import load_model


#json_file = open('modello.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()

#loaded_model =model_from_json(loaded_model_json)
#model = load_model('model.h5')

(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()
class_names = ['Sandal', 'Trouser', 'T-Shirt/Top', 'Sneaker', 'Dress', 'Shirt', 'Bag', 'Pullover', 'Coat', 'Ankle boot']

plt.figure()
plt.imshow(train_images[0])
plt.grid(False)
plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

train_images_1= tf.reshape(train_images, [-1,28,28,1])
test_images_1= tf.reshape(test_images, [-1,28,28,1])

model = Sequential()
model.add(Input(shape=(28,28,1)))
model.add(Conv2D(32, (5,5), strides=(1,1)))
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Conv2D(32, (5,5), strides=(1,1)))
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Flatten())
model.add(Dense(256, activation = 'tanh'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
callback = keras.callbacks.EarlyStopping(monitor = 'accuracy', patience = 3)
history = model.fit(train_images_1,train_labels, epochs= 100,callbacks=[callback], batch_size = 50)
test_loss, test_accuracy = model.evaluate(test_images_1,test_labels, verbose=2)
print('\n Test accuracy: ', test_accuracy )
predictions = model.predict(test_images_1)

#acc = history.history['accuracy']
#loss = history.history['loss']
#val_acc = history.history['val_accuracy']
#val_loss = history.history['val_loss']
#plt.plot(acc,label='accuracy', color = 'red')
#plt.plot(val_acc, label='test_accuracy', color = 'blue')
#plt.xlabel('Epoch')
#plt.ylabel('accuracy')
