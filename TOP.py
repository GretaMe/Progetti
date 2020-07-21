import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from pathlib import Path
from skimage import io as skio
from sklearn.model_selection import train_test_split
from skimage.transform import resize

data_dir = '/Users/meglioli/Documents/MATLAB/data_new/original/'
data_deb_dir = '/Users/meglioli/Documents/MATLAB/data_new/deblurred/'

img_width = 2048
img_height = 64
input_depth = 1

#img_dir_path = Path(data_dir)
#print(list(img_dir_path.glob("*.png")))
img_dir_path = os.path.join(data_dir,"*.png")
train_dir_paths = glob(img_dir_path)
a = list(train_dir_paths)
print(len(a))

deb_dir_path = os.path.join(data_deb_dir,"*.png")
deb_dir_paths = glob(deb_dir_path)
b = list(deb_dir_paths)
print(len(b))

def img_gen(img_paths, img_size = (img_width, img_height, input_depth)):
  # fai un glob delle della cartella in cui cerchi i jpg
  #img_paths = img_dir_path.glob("*.png")

  # iteri sul glob
  for img_path in img_paths:
    img = skio.imread(img_path)/255. #carichi l'immagine
    img = resize(img, img_size, preserve_range=True)
    yield img #fai yield dell'immagine 
    
image_generator = img_gen(train_dir_paths)
next_image = next(image_generator)
print(next_image.shape)
#plt.imshow(next_image, cmap='gray')

def image_batch_generator(img_paths, deb_paths, batchsize=32):
    while True:
        ig = img_gen(img_paths)
        ig_deb = img_gen(deb_paths)
        batch_img= []
        batch_deb = []
        
        for index,img in enumerate(ig):
            # Add the image and mask to the batch
            batch_img.append(img)
            batch_deb.append(next(ig_deb))
            # If we've reached our batchsize, yield the batch and reset
            if len(batch_img) == batchsize:
                yield np.stack(batch_img, axis=0),np.stack(batch_deb, axis=0)
                batch_img = []
                batch_deb = []
    
        
        # If we have an nonempty batch left, yield it out and reset
        if len(batch_img) != 0 and len(batch_deb) != 0:
            yield np.stack(batch_img, axis=0), np.stack(batch_deb, axis=0)
            batch_img =[]  
            batch_deb = []  
            


BATCHSIZE = 32

# Split the data into a train and validation set
train_img_paths, val_img_paths, train_deb_paths, val_deb_paths= train_test_split(train_dir_paths,deb_dir_paths, test_size=0.15, shuffle=False)

# Create the train and validation generators
#traingen = image_batch_generator(train_img_paths, batchsize=BATCHSIZE)
#valgen = image_batch_generator(val_img_paths, batchsize=BATCHSIZE)
#traingen_deb = image_batch_generator(train_deb_paths, batchsize=BATCHSIZE)
#valgen_deb = image_batch_generator(val_deb_paths, batchsize=BATCHSIZE)

def calc_steps(data_len, batchsize):
    return (data_len + batchsize - 1) // batchsize

# Calculate the steps per epoch
train_steps = calc_steps(len(train_img_paths), BATCHSIZE)
val_steps = calc_steps(len(val_img_paths), BATCHSIZE)

train_deb_steps = calc_steps(len(train_deb_paths), BATCHSIZE)
val_deb_steps = calc_steps(len(val_deb_paths), BATCHSIZE)

# Train the model

model = Sequential()
input_shape = (img_width,img_height,input_depth)
model.add(Conv2D(1,(3,3), padding = 'same', input_shape = input_shape, activation = 'relu', data_format = 'channels_last'))
model.add(Conv2D(1,(3,3), padding = 'same', activation = 'relu', data_format = 'channels_last'))
model.add(Dense(1))
model.add(Conv2D(1,(3,3), padding = 'same', activation = 'relu', data_format = 'channels_last'))
model.add(Conv2D(1,(3,3), padding = 'same', activation = 'relu', data_format = 'channels_last'))

model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mse'])
#callback = tf.keras.callbacks.EarlyStopping(monitor = 'mse', patience = 3)

#history = model.fit_generator(
#    traingen, 
#    traingen_deb,
#    steps_per_epoch=train_steps, 
#    epochs=100, # Change this to a larger number to train for longer
#    validation_data=(valgen,valgen_deb), 
#    validation_steps=val_steps
#)

history = model.fit(
    image_batch_generator(train_img_paths,train_deb_paths, batchsize=BATCHSIZE),
    steps_per_epoch=train_steps, 
    epochs=100, # Change this to a larger number to train for longer
    validation_data=image_batch_generator(val_img_paths,val_deb_paths, batchsize=BATCHSIZE), 
    validation_steps=val_steps
)
