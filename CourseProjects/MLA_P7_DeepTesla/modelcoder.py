from keras.layers.pooling import MaxPooling2D 
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Activation,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras import optimizers
import params

def get_nvidia_model():
    print("Start to create model")
    model = Sequential()
    model.add(Lambda(lambda x:x/255., input_shape=(params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c)))
    model.add(Convolution2D(24,(5,5), activation='relu', strides=(2,2), padding='valid'))
    model.add(Convolution2D(36,(5,5), activation='relu', strides=(2,2), padding='valid'))
    model.add(Convolution2D(48,(5,5), activation='relu', strides=(2,2), padding='valid'))
    model.add(Convolution2D(64,(3,3), activation='relu', strides=(1,1), padding='valid'))
    model.add(Convolution2D(64,(3,3), activation='relu', strides=(1,1), padding='valid'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='Adam')
    print("Model created")
    return model

def get_nvidia_model_2():
    print("Start to create model")
    model = Sequential()
    model.add(Lambda(lambda x:x/255., input_shape=(params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c)))
    
    model.add(Convolution2D(24,(3,3), activation='elu',  padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32,(3,3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(48,(3,3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64,(3,3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64,(3,3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(1164, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))

    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam)
    print("Model created")
    return model