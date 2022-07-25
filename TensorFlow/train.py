import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from os import listdir
from os.path import isfile, join
import importlib
import pandas as pd
# import amir_tools as at
from sklearn import preprocessing
import gzip
import TF_resnet_model as tflow_resnet
import raghunath_tf_ecg_models as tflow_cnn
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import os


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
import argparse
import datetime
import imp

# python3 train_reg.py --method raghunath_cnn_ecg --label_path /home/weijiesun/ECG_survival/data/demo/label.pickle --demographic_path /home/weijiesun/ECG_survival/data/demo/demographic_df.pickle --train_path /home/weijiesun/ECG_survival/data/demo/train.pickle --val_path /home/weijiesun/ECG_survival/data/demo/val.pickle
# python3 train_reg.py --method automatic_ecg --label_path /home/weijiesun/ECG_survival/data/demo/label.pickle --demographic_path /home/weijiesun/ECG_survival/data/demo/demographic_df.pickle --train_path /home/weijiesun/ECG_survival/data/demo/train.pickle --val_path /home/weijiesun/ECG_survival/data/demo/val.pickle
if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument("--method", type=str, default='automatic_ecg') # first, last, random, all
    parser.add_argument("--label_path", type=str) # first, last, random, all
    parser.add_argument("--demographic_path", type=str) # first, last, random, all
    parser.add_argument("--train_path", type=str) # training data path
    parser.add_argument("--val_path", type=str) # training data path
    parser.add_argument("--ecg_np_path", type=str, default = "/home/padmalab/ecg/data/processed/ecgs_compressed/ecgs_np/%s.xml.npy.gz") # ECG data path
    
    args = parser.parse_args()
    method = args.method
    label_df = pd.read_pickle(args.label_path)
    demographic_df = pd.read_pickle(args.demographic_path)
    train = pd.read_pickle(args.train_path)
    val = pd.read_pickle(args.val_path)
    ecg_np_path = args.ecg_np_path
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]='0'

    demographic_size = demographic_df.shape[1]
    label_number = label_df.shape[1]
    batch_size = 64
############################# loss  ####################################
    # loss functions: 
    w_pos = (len(label_df)-label_df.sum())/label_df.sum()
    w_pos = w_pos.to_numpy()
    w_pos = w_pos/min(w_pos)
    w_pos = tf.constant([w_pos]*512,dtype = 'float32')

    def complete_weighted_bincrossentropy(true, pred):
        # calculate the binary cross entropy
        bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

        # apply the weights
        weights = w_pos
        weighted_bin_crossentropy = weights * bin_crossentropy 
        return keras.backend.mean(weighted_bin_crossentropy)

    # batch_weighted: weights determined per batch
    def batch_weighted_bincrossentropy(true, pred):
        # source: https://github.com/huanglau/Keras-Weighted-Binary-Cross-Entropy/blob/master/DynCrossEntropy.py
        num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)
        zero_weight =  keras.backend.sum(true)/ num_pred +  keras.backend.epsilon() 
        one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred +  keras.backend.epsilon()

        weights =  (1.0 - true) * zero_weight +  true * one_weight 
        bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

        weighted_bin_crossentropy = weights * bin_crossentropy 

        return keras.backend.mean(weighted_bin_crossentropy)
    def get_loss(loss_name):
        if loss_name == "Weighted":
            return complete_weighted_bincrossentropy
        if loss_name ==  "WeightedPerBatch":
            return batch_weighted_bincrossentropy
        if loss_name == "Notweighted":
            return 'binary_crossentropy'
        
############################# loss end  ####################################

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    if method == 'raghunath_cnn_ecg':
        training_generator = tflow_cnn.DataGenerator_raghunath_lab(train, label_df, demographic=demographic_df, 
                                                            n_classes=label_number, batch_size=batch_size, demographic_size=demographic_size,
                                                            np_path = ecg_np_path)
        validation_generator = tflow_cnn.DataGenerator_raghunath_lab(val, label_df, demographic=demographic_df, 
                                                            n_classes=label_number, batch_size=batch_size, demographic_size=demographic_size,
                                                            np_path = ecg_np_path)

        model = tflow_cnn.get_raghunath_model(label_number, demographic_size)
    elif method == 'automatic_ecg':
        training_generator = tflow_resnet.DataGenerator_lab(train, label_df, demographic=demographic_df, 
                                                            n_classes=label_number, batch_size=batch_size, demographic_size=demographic_size,
                                                            np_path = ecg_np_path)

        validation_generator = tflow_resnet.DataGenerator_lab(val, label_df, demographic=demographic_df, 
                                                            n_classes=label_number, batch_size=batch_size, demographic_size=demographic_size,
                                                            np_path = ecg_np_path)
        model = tflow_resnet.get_model_lab(label_number, lab_number=demographic_size)

    else:
        print ('method error, should be one of automatic_ecg or raghunath_cnn_ecg')
        
    model.compile(loss=loss, optimizer=opt)
    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                  ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
    # Train neural network
    history = model.fit(training_generator,
                        epochs=70,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        verbose=1)
    