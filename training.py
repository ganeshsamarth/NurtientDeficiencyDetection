# training models
from PIL import Image
import numpy as np
import os
import h5py
import random
import cv2
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
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

model = ResNet50((224,224,3),2)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit(x=X,y=Y,batch_size = 32, epochs = 20,validation_split=0.2)
