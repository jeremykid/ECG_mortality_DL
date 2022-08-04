import numpy as np
import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense, AveragePooling1D)
from tensorflow.keras.models import Model
import tensorflow as tf
import gzip

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
    
    
def get_raghunath_model(n_classes, demographic_size):
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
    
    tabular = Input(shape=(demographic_size), dtype=np.float32, name='tabular')
    x_tab = Dense(64,activation='relu')(tabular)
#     x_tab = Dense(10,activation='relu')(x_tab)
    
    concat = tf.keras.layers.Concatenate()([x_tab, x0, x1, x2, x3, x4])
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
                          signal_4, tabular], outputs=result)
    return model

class DataGenerator_raghunath_lab(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, ecg_labels, demographic=False, batch_size=64,
                 n_classes=10, shuffle=False, features_complete=False, demographic_size=2,
                np_path = "./ecgs_np/%s.xml.npy.gz"):
        'Initialization'
        self.batch_size = batch_size
        self.labels = ecg_labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.agsx_df = demographic
        self.lab_size = demographic_size
        self.np_path = np_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, list_IDs_temp = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, 5040, 12))
        X_agsx = np.zeros((self.batch_size, self.lab_size))
        y = np.zeros((self.batch_size, self.n_classes))
        for i, ID in enumerate(list_IDs_temp):
            try: 
                np_path = self.np_path%ID
                f = gzip.GzipFile(np_path, "r")
                enp = np.load(f)[:,0:5040]
#                 print (enp)
            except Exception as e:
                enp = np.zeros([12,5040])
#                 print (e)
                pass
                
            X[i] = enp
            y[i] = self.labels.loc[ID].to_numpy()
            try:
                X_agsx[i] = self.agsx_df.loc[ID].to_numpy()
            except:
                print("please supply demographic dataframe to the data generator", ID)
                break
                
        tabular = X_agsx # Age, Sex
        signal_0 = X[:, :, [7, 1, 10]] # V1, II, V5
        signal_1 = X[:, :1248, [0, 1, 2]] # I, II, III
        signal_2 = X[:, :1248, [3, 4, 5]] # aVR, aVL, aVF
        signal_3 = X[:, :1248, [6, 7, 8]] # V1, V2, V3
        signal_4 = X[:, :1248, [9, 10, 11]] # V4, V5, V6
        
        return (signal_0, signal_1, signal_2, signal_3, signal_4, tabular), y, list_IDs_temp

if __name__ == "__main__":
    model = get_raghunath_model(1,2)
    model.summary()