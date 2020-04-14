import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.datasets import mnist
import numpy as np
import scipy 
import matplotlib.pyplot as plt

#from numpy.random import seed
#seed(1)
#tf.compat.v1.set_random_seed(2)
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
 
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()
 
    return kernel_2D

def my_init(shape, dtype=tf.float32):
    my_kernel = tf.convert_to_tensor(gaussian_kernel(5,sigma = 1))
    my_kernel = tf.reshape(my_kernel,(5,5,1,1))
    my_kernel = tf.cast(my_kernel,dtype= dtype)
    return my_kernel

#suddivido il dataset in trainset e in testset
(train_images,train_labels),(test_images, test_labels) = mnist.load_data()
#iperparametri
N = 784
batch_size = 100
sigma = 1
#normalizzazione
train_images = train_images/255.0
test_images = test_images/255.0
#mostro la prima immagine del trainset
plt.figure()
plt.imshow(train_images[0], cmap = 'Greys')
plt.grid(False)
plt.colorbar()
plt.show()

#faccio un ciclo per evitare che il gaussian_filter sovrascriva le immagini per la sua proprietà di multidimensionalità
filtered_images = []
for image in train_images:
    filtered_image = scipy.ndimage.gaussian_filter(image, sigma)
    filtered_images.append(filtered_image)
    
gaussian_images = np.array(filtered_images)
#mostro la prima immagine delle immagini trasformate con il filtro gaussiano
plt.figure()
plt.imshow(gaussian_images[0], cmap = 'Greys')
plt.grid(False)
plt.colorbar()
plt.show()
#ridimensiono le immagini in modo che possano andare bene con la rete
gaussian_images = tf.reshape(gaussian_images, [-1,28,28,1])
train_images = tf.reshape(train_images, [-1,28,28,1])
test_images = tf.reshape(test_images, [-1,28,28,1])
#nomino gli insiemi su cui faccio l'addestramento
X = train_images
Y = gaussian_images
#costruisco il modello con un solo strato convolutivo 5x5
#il kernel_initializer permette di settare i pesi iniziali della rete come desidero (distribuzione normale troncata)
model = Sequential()
model.add(Input(shape = (28,28,1)))
#layer=Conv2D(1,(5,5), kernel_initializer= TruncatedNormal(mean =0.0, stddev = 0.05, seed= 1), padding = 'same',data_format = 'channels_last', use_bias = False)
#creo un initializer in modo che la matrice dei pesi sia di mio piacimento
layer=Conv2D(1,(5,5), kernel_initializer= my_init, padding = 'same',data_format = 'channels_last', use_bias = False)
layer.trainable = True
model.add(layer)
#utilizzo come metrica 'mse' dato che questo è un problema di regressione e pertanto non è saggio 
#valutare i progressi della rete con l'accuracy
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['mse'])
callback = keras.callbacks.EarlyStopping(monitor = 'mse', patience = 3)
#attraverso il validation_split prendo una parte elle immagini di training e le uso per addestrare la rete (prendo il 10% delle immagini)
model.fit(X,Y,validation_split=0.1, epochs = 10, callbacks = [callback] ,batch_size = batch_size, verbose = 1)

#guardo se la rete funziona prendendo le mie immagini di testing 
data = test_images[0:100]
#faccio un ciclo per calcolare la gaussiana su queste immagini
filtered_data = []
for image in data:
    filtered_image = scipy.ndimage.gaussian_filter(image, sigma)
    filtered_data.append(filtered_image)
    
data_gauss = np.array(filtered_data)
#ridimensiono in modo che le immagini possano entrare nella rete
data_gauss_re = tf.reshape(data_gauss, [-1,28,28,1])
predictions = model.predict(data)
risultati = model.evaluate(data,data_gauss_re)
print('test loss, test accuracy:', risultati)
ANN = predictions
GAUSS = data_gauss_re.numpy()

def MSE(imageA,imageB):
    err= np.sum((imageA.astype( "float" ) - imageB.astype( "float" ))**2)
    err/= float(imageA.shape[0] * imageA.shape[1])
    return err

print('\n Il parametro che indica il livello di sovrapposizione tra le immagini corrisponde a : ',MSE(ANN,GAUSS))
#estraggo i pesi della rete e li visualizzo ridimensionandoli prima
weights= model.get_weights()[0]
weights = tf.reshape(weights,(5,5))
plt.figure()
plt.imshow(weights)
plt.colorbar()
#salvo il modello per poterlo utilizzare in un altro programma
save_model(model,'/home/meglioli/model_0.h5')
