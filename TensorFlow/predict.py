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
import TF_dnn_ecg_models as tflow_cnn
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

# python predict.py --method METHOD_NAME --model_path MODEL_PATH --label_path LABEL_PATH --tabular_path TABULAR_PATH --test_path TEST_PATH --ouput_file PATH_TO_OUTPUT_FILE --ecg_np_path ECG_PATH`

# python3 predict.py --method ResNet --model_path ./backup_model_best.hdf5 --label_path ../demo_data/label.pickle --tabular_path ../demo_data/demographic_df.pickle --test_path ../demo_data/val.pickle --ouput_file ./result.pickle --ecg_np_path ../demo_data/demo_ecg/%s.xml.npy.gz 

if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument("--method", type=str, default='ResNet') # 
    parser.add_argument("--model_path", type=str) #  backup_model_best
    parser.add_argument("--label_path", type=str) #  
    parser.add_argument("--tabular_path", type=str) # 
    parser.add_argument("--test_path", type=str) # 
    parser.add_argument("--ouput_file", type=str) # 
    parser.add_argument("--ecg_np_path", type=str, default = "../demo_data/%s.xml.npy.gz") # ECG data path
    
    args = parser.parse_args()
    method = args.method
    model_path = args.model_path
    label_df = pd.read_pickle(args.label_path)
    demographic_df = pd.read_pickle(args.tabular_path)
    test = pd.read_pickle(args.test_path)
    output_file = args.ouput_file
    ecg_np_path = args.ecg_np_path
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]='0'

    demographic_size = demographic_df.shape[1]
    label_number = label_df.shape[1]
    batch_size = 64
    
    ################################# 
    
    if method == 'DNN':
        test_generator = tflow_cnn.DataGenerator_raghunath_lab(test, label_df, demographic=demographic_df, 
                                                            n_classes=label_number, batch_size=batch_size, demographic_size=demographic_size,
                                                            np_path = ecg_np_path)

        model = tflow_cnn.get_raghunath_model(label_number, demographic_size)
    elif method == 'ResNet':
        test_generator = tflow_resnet.DataGenerator_lab(test, label_df, demographic=demographic_df, 
                                                            n_classes=label_number, batch_size=batch_size, demographic_size=demographic_size,
                                                            np_path = ecg_np_path)

        model = tflow_resnet.get_model_lab(label_number, lab_number=demographic_size)

    else:
        print ('method error, should be one of ResNet or DNN')
    
    ################################# 
    model = tf.keras.models.load_model(model_path, compile=False)#, custom_objects={'batch_weighted_bincrossentropy': batch_weighted_bincrossentropy})
    y_list = []
    predict_list = []
    IDs = []
    for X, y in test_generator:
        predict = model.predict(X, batch_size=64, verbose=0)
        y_list.append(y)
        predict_list.append(predict)
    labels = np.concatenate(y_list, axis=0)
    pred = np.concatenate(predict_list, axis=0)
    
    pred_df = pd.DataFrame(pred)
    pred_df.to_pickle(output_file)
