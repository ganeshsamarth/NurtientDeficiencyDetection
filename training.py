# training models
from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import h5py
import random
import cv2
import h5py

from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras import __version__
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout,Conv2D,Activation,Reshape, Flatten, MaxPooling2D, BatchNormalization, Input, concatenate, merge, AveragePooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from models import resnet,squeezenet,inceptionv3,alexnet

output_directory = '/home/student/Desktop/NurtientDeficiencyDetection/'
h5f = h5py.File("{}data_training.h5".format(output_directory), 'r')
train_X = h5f['X_train']
train_Y = h5f['Y_train']
h5f1 = h5py.File("{}data_testing.h5".format(output_directory), 'r')
test_X = h5f1['X_test']
print(test_X.shape)
test_Y = h5f1['Y_test']
input_shape=[224,224,3]
model = squeezenet.SqueezeNet(2)
optimizer1 = keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer = optimizer1, loss = 'categorical_crossentropy',metrics = ['accuracy'])

# add this line before model.fit
checkpointer = ModelCheckpoint(filepath='/home/student/Desktop/ganesh/weights_squeezenet_3.h5', verbose=1, save_best_only=True)




#change model.fit in this way
model.fit(train_X, train_Y, batch_size=128, epochs=50,validation_split=0.2, callbacks=[checkpointer])

#code for loading the weights saved
model.load_weights('/home/student/Desktop/ganesh/weights_squeezenet_3.h5')

#this should print your accuracy on the terminal

results = model.evaluate(test_X,test_Y)
print('test loss, test acc:', results)




