import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Input,Flatten, Conv2D, MaxPool2D
from tensorflow.keras.datasets import mnist
import numpy as np
import scipy 
import matplotlib.pyplot as plt
#from skimage.filters import gaussian


(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

N = 784
batch_size = 20
sigma = 1

train_images = train_images/255.0
test_images = test_images/255.0

plt.figure()
plt.imshow(train_images[0], cmap = 'Greys')
plt.grid(False)
plt.show()

filtered_images = []
for image in train_images:
    filtered_image = scipy.ndimage.gaussian_filter(image, sigma)
    filtered_images.append(filtered_image)
    
gaussian_images = np.array(filtered_images)

plt.figure()
plt.imshow(gaussian_images[0], cmap = 'Greys')
plt.grid(False)
plt.show()

gaussian_images = tf.reshape(gaussian_images, [-1,28,28,1])
train_images = tf.reshape(train_images, [-1,28,28,1])
test_images = tf.reshape(test_images, [-1,28,28,1])

X = train_images
Y = gaussian_images

model = Sequential()
model.add(Input(shape = (28,28,1)))
model.add(Conv2D(1,(6,6), padding = 'same',data_format = 'channels_last'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
#callback = keras.callbacks.EarlyStopping(monitor = 'accuracy', patience = 3)
model.fit(X,Y, epochs = 5, batch_size = batch_size)

data = test_images[0:10]
filtered_data = []
for image in data:
    filtered_image = scipy.ndimage.gaussian_filter(image, sigma)
    filtered_data.append(filtered_image)
    
data_gauss = np.array(filtered_data)
data_gauss_re = tf.reshape(data_gauss, [-1,28,28,1])
predictions = model.predict(data)
risultati = model.evaluate(data,data_gauss_re)
print('test loss, test accuracy:', risultati)
ANN = predictions
FFT = data_gauss_re.numpy()

def MSE(imageA,imageB):
    err= np.sum((imageA.astype( "float" ) - imageB.astype( "float" ))**2)
    err/= float(imageA.shape[0] * imageA.shape[1])
    return err

print('\n Il parametro che indica il livello di sovrapposizione tra le immagini corrisponde a : ',MSE(ANN,FFT))
save_model(model,'/home/meglioli/model_0.h5')
