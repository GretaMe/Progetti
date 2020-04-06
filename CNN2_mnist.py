from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from scipy.fftpack import fft2
from tensorflow.keras.models import load_model

model1 = load_model('/home/meglioli/model.h5')

(train_images, train_labels), (test_images, test_labels)= mnist.load_data()
class_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure()
plt.imshow(train_images[0], cmap = 'Greys')
plt.grid(False)
plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

fourier_images = fft2(train_images)
fourier_images = tf.math.real(fourier_images)

fourier_testimages = fft2(test_images)
fourier_testimages = tf.math.real(fourier_testimages)

train_images_1= tf.reshape(train_images, [-1,28,28])
test_images_1= tf.reshape(test_images, [-1,28,28])
fourier_images_1 = tf.reshape(fourier_images, [-1,28,28])
fourier_testimages_1 = tf.reshape(fourier_testimages, [-1,28,28])


model = Sequential()
model.add(model1.layers[0])
#model.add(Input(shape=(28,28,1)))
model.add(Reshape([28,28,1]))
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
history = model.fit(fourier_images_1,train_labels, epochs= 3,callbacks=[callback], batch_size = 50)
test_loss, test_accuracy = model.evaluate(fourier_testimages_1,test_labels, verbose=2)
print('\n Test accuracy: ', test_accuracy )
predictions = model.predict(fourier_testimages_1)
