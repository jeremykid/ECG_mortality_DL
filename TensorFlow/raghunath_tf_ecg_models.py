import numpy as np
import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from itertools import cycle
import gzip
import tensorflow as tf
# import neurokit2 as nk

class parallel_model_A(object):
    def __init__(self, kernel_initializer = 'he_normal'):
        self.kernel_initializer = kernel_initializer
    
    def _conv_block(self, x, filters, kernel_size):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def __call__(self, inputs):
        x = inputs
        x = self._conv_block(x, 3, 2521)
        x = self._conv_block(x, 16, 1261)
        x = self._conv_block(x, 32, 631)
        x = self._conv_block(x, 64, 316)
        x = self._conv_block(x, 128, 158)
        x = self._conv_block(x, 256, 1)
        x = self._conv_block(x, 512, 1)

        x = AveragePooling1D(pool_size=2,
           strides=3, padding='valid')(x)
        x = Flatten()(x)
        return x
    
class parallel_model_B(object):
    def __init__(self, kernel_initializer = 'he_normal'):
        self.kernel_initializer = kernel_initializer
    
    def _conv_block(self, x, filters, kernel_size):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def __call__(self, inputs):
        x = inputs
        x = self._conv_block(x, 3, 625)
        x = self._conv_block(x, 64, 313)
        x = self._conv_block(x, 128, 157)
        x = self._conv_block(x, 256, 1)
        x = self._conv_block(x, 512, 1)

        x = AveragePooling1D(pool_size=2,
           strides=3, padding='valid')(x)
        x = Flatten()(x)
        return x
    
    
def get_raghunath_model(n_classes):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal_0 = Input(shape=(5040, 3), dtype=np.float32, name='signal_0')
    x0 = parallel_model_A()(signal_0)
    signal_1 = Input(shape=(1248, 3), dtype=np.float32, name='signal_1')
    x1 = parallel_model_B()(signal_1)
    signal_2 = Input(shape=(1248, 3), dtype=np.float32, name='signal_2')
    x2 = parallel_model_B()(signal_2)
    signal_3 = Input(shape=(1248, 3), dtype=np.float32, name='signal_3')
    x3 = parallel_model_B()(signal_3)
    signal_4 = Input(shape=(1248, 3), dtype=np.float32, name='signal_4')
    x4 = parallel_model_B()(signal_4)
    signal_5 = Input(shape=(1248, 3), dtype=np.float32, name='signal_5')
    x5 = parallel_model_B()(signal_5)
    
    tabular = Input(shape=(2), dtype=np.float32, name='tabular')
    x_tab = Dense(64,activation='relu')(tabular)
#     x_tab = Dense(10,activation='relu')(x_tab)
    
    concat = tf.keras.layers.Concatenate()([x_tab, x0, x1, x2, x3, x4, x5])
    concat = Dense(256, activation='relu', 
              kernel_initializer=kernel_initializer)(concat)
    concat = Dropout(0.2)(concat)

    concat = Dense(128, activation='relu', 
              kernel_initializer=kernel_initializer)(concat)
    concat = Dropout(0.2)(concat)
    
    concat = Dense(64, activation='relu', 
              kernel_initializer=kernel_initializer)(concat)
    
    concat = Dense(32, activation='relu', 
              kernel_initializer=kernel_initializer)(concat)
    
    concat = Dense(8, activation='relu', 
              kernel_initializer=kernel_initializer)(concat)
    
    result = Dense(n_classes, activation='sigmoid', 
                  kernel_initializer=kernel_initializer)(concat)
    model = Model(inputs=[signal_0, signal_1, signal_2, signal_3, 
                          signal_4, signal_5, tabular], outputs=result)
    return model

if __name__ == "__main__":
    model = get_raghunath_model(1)
    model.summary()