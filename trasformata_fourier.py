#Addestramento di una rete a calcolare la Trasformata di Fourier su un determinato datase.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Input,Flatten, Conv2D
from tensorflow.keras.datasets import mnist
import numpy as np
#from scipy.fftpack import fft2
import matplotlib.pyplot as plt

#Caricamento dati dal dataset mnist.
#Suddivisione in train set e test set con le rispettive labels.
(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

N = 784
batch_size = 20

#Noemalizzazione delle immagini per ottenere vali tra 0 e 1.
train_images = train_images/255.0
test_images = test_images/255.0
print('\n I valori minimi e massimi dei pixel sono: ',train_images.min(), train_images.max())

#Disegno del primo esempio del train set.
plt.figure()
plt.imshow(train_images[0], cmap = 'Greys')
plt.grid(False)
plt.show()

train_images_tensor = tf.convert_to_tensor(train_images) 
train_images_tensor = tf.cast(train_images_tensor, dtype= tf.complex64)
#Calcolo manuale della Trasformata di Fourier Bidimensionale.
#Estrazione della sola parte reale per poter mostrare l'immagine a cui viene applicata la fft2.
fourier_images = tf.signal.fft2d(train_images_tensor)
fourier_images = tf.math.real(fourier_images)

#Disegno del primo esempio del train set trasformato tramite Fourier.
plt.figure()
plt.imshow(fourier_images[0], cmap = 'Greys')
plt.grid(False)
plt.show()

#Trasformazione dimensionale delle immagini trasformate.
train_images = tf.reshape(train_images, [-1,28,28,1])
fourier_images = tf.reshape(fourier_images, [-1,28,28,1])

#Nomina degli input e delle etichette corrispondenti.
X = train_images
Y = fourier_images

#Creazione della rete in maniera sequenziale.
model = Sequential()
model.add(Input(shape = (28,28,1))) #Dimensione dell'immagine di input.
model.add(Conv2D(10,(3,3), padding = 'same',data_format = 'channels_last'))
model.add(Conv2D(10,(3,3), padding = 'same',data_format = 'channels_last'))
model.add(Conv2D(10,(3,3), padding = 'same',data_format = 'channels_last'))
model.add(Conv2D(10,(3,3), padding = 'same',data_format = 'channels_last'))
#model.add(Flatten()) #Appiattimento in un vettore monodimensionale dell'immagine.
#model.add(Dense(N, use_bias = False)) #Creazione di uno strato denso.
#Compilazione del metodo con binary_crossentropy per il confronto tra due immagini, conottimizzatore adam
#e con metrica accuracy per valutare il grado di accuratezza della rete calcolato ad ogni epoca.
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
#Introduzione dell'Early Stopping per evitare la ripetizione degli stessi valory di accuracy.
#callback = keras.callbacks.EarlyStopping(monitor = 'accuracy', patience = 3)
#Crazione vera a propria del modello con mini batch.
model.fit(X,Y, epochs = 1, batch_size = batch_size)

#Conferma del buon funzionamento della rete.
data = test_images[0:10] #Selezione 10 immagini del test set.
data = tf.reshape(data, [-1,28,28,1])
data_tensor = tf.convert_to_tensor(data)
data_tensor = tf.cast(data_tensor, dtype= tf.complex64)
data_fourier = tf.signal.fft2d(data_tensor) #Trasformazione immagini tramite fft2.
data_fourier = tf.math.real(data_fourier) #Estrapolazione parte reale.
data_fourier = tf.reshape(data_fourier, [-1,28,28,1]) #Ridimensionamento per adattarsi alla rete.
predictions = model.predict(data) #Calcolo predizioni tramite la rete.
results = model.evaluate(data,data_fourier) #Valutazione del modello.
print('test loss, test accuracy:', results)

#confronto tra immagini tramite mean_squared_error.
ANN = predictions
FFT = data_fourier.numpy() #TRasformazione da tensore a array.
print('\n I valori minimi e massimi dei pixel sono: ',FFT.min(), FFT.max())
def MSE(imageA,imageB):
    err= np.sum((imageA.astype( "float" ) - imageB.astype( "float" ))**2)
    err/= float(imageA.shape[0] * imageA.shape[1])
    return err

print('\n Il parametro che indica il livello di sovrapposizione tra le immagini corrisponde a : ',MSE(ANN,FFT))

#Salvataggio del modello per poterne utilizzare i pesi congelandoli nella rete CNN_mnist
save_model(model,'/home/meglioli/model.h5')
