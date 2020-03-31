import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
#import h5py

N =784 #(28x28) per le immagini
batch = 1000

# Generate random input data and desired output data  (training set)
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()


train_images = train_images/255.0
test_images = test_images/255.0

train_images= tf.reshape(train_images, [-1,784,1])
test_images= tf.reshape(test_images, [-1,784,1])

train_images_1 = keras.backend.eval(train_images[0:100,:,:])
test_images_1 = keras.backend.eval(test_images)
fourier_images = fft(train_images_1)

train_images_1= tf.reshape(train_images_1, [-1,28,28,1])
test_images_1= tf.reshape(test_images_1, [-1,28,28,1])
fourier_images= tf.reshape(fourier_images, [-1,28,28,1])

train_images_1 = keras.backend.eval(train_images_1)
test_images_1 = keras.backend.eval(test_images_1)
fourier_images = keras.backend.eval(fourier_images)

# First half of inputs/outputs is real part, second half is imaginary part
X = train_images_1
Y = fourier_images

# Create model with no hidden layers, same number of outputs as inputs.
# No bias needed.  No activation function, since FFT is linear.
model = Sequential()
model.add(Input(shape=(28,28,1)))
model.add(Dense(N, use_bias=False))
#model.add(Dense(N, use_bias=False, activation = 'tanh'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#plt.figure()
#plt.imshow(model.get_weights()[0], vmin=-1, vmax=1, cmap='coolwarm') #lista dei pesi di ogni singolo layer
callback = keras.callbacks.EarlyStopping(monitor = 'accuracy', patience = 10)
history = model.fit(X, Y, epochs=50,callbacks = [callback], batch_size=30)
weights = model.get_weights()
plt.figure()
plt.imshow(weights[0], vmin=-1, vmax=1, cmap='coolwarm')

# Confirm that it works (test set)
data = test_images_1[0:10]
data_fourier = fft(data)
predictions = model.predict(data)
risultati = model.evaluate(data,data_fourier)
print('test loss, test accuracy:', risultati)
ANN = predictions
FFT = data_fourier

def MSE(imageA,imageB):
    err= np.sum((imageA.astype( "float" ) - imageB.astype( "float" ))**2)
    err/= float(imageA.shape[0] * imageA.shape[1])
    return err

print('\n Il parametro che indica il livello di sovrapposizione tra le immagini corrisponde a : ',MSE(ANN,FFT))
## Heat map of neuron weights
##plt.figure()
##plt.imshow(model.get_weights()[0], vmin=-1, vmax=1, cmap='coolwarm')
model_json= model.to_json()
with open('modello.json','w') as json_file:
    json_file.write(model_json)

#model.save_weights('model.h5')
