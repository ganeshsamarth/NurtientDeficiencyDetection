#alexnet model
from keras.models import Model,Sequential
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input, Flatten,Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image

def AlexNetModel():
  model = Sequential()

  # 1st Convolutional Layer
  model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
  strides=(4,4), padding='valid'))
  model.add(Activation('relu'))
  # Pooling
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
  # Batch Normalisation before passing it to the next layer
  model.add(BatchNormalization())

  # 2nd Convolutional Layer
  model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
  model.add(Activation('relu'))
  # Pooling
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
  # Batch Normalisation
  model.add(BatchNormalization())

  # 3rd Convolutional Layer
  model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
  model.add(Activation('relu'))
  # Batch Normalisation
  model.add(BatchNormalization())

  # 4th Convolutional Layer
  model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
  model.add(Activation('relu'))
  # Batch Normalisation
  model.add(BatchNormalization())

  # 5th Convolutional Layer
  model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
  model.add(Activation('relu'))
  # Pooling
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
  # Batch Normalisation
  model.add(BatchNormalization())

  # Passing it to a dense layer
  model.add(Flatten())
  # 1st Dense Layer
  model.add(Dense(4096, input_shape=(224*224*3,)))
  model.add(Activation('relu'))
  # Add Dropout to prevent overfitting
  model.add(Dropout(0.4))
  # Batch Normalisation
  model.add(BatchNormalization())

  # 2nd Dense Layer
  model.add(Dense(4096))
  model.add(Activation('relu'))
  # Add Dropout
  model.add(Dropout(0.4))
  # Batch Normalisation
  model.add(BatchNormalization())

  # 3rd Dense Layer
  model.add(Dense(1000))
  model.add(Activation('relu'))
  # Add Dropout
  model.add(Dropout(0.4))
  # Batch Normalisation
  model.add(BatchNormalization())

  # Output Layer
  model.add(Dense(2))
  model.add(Activation('softmax'))

  return model
