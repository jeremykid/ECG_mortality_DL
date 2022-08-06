# Scripts and modules to  train deep learning models for mortality prediction using ECGs
This is the companion code base for the paper 'Towards artificial intelligence based learning health system for population-level prediction of short and longer-term mortality using electrocardiograms'. 

This study focused on developing and evaluating ECG based mortality models to predict the probability of a patient dying within 30-days, 1-year and 5-year, starting from the day of ECG acquisition. ECGs to be used in these models can be acquired at any time point during a healthcare episode . The goal of the prediction model is to output a calibrated probability of mortality, which could be then used as the patient's risk score for clinical management or resource allocation during patient's stay in the hospital or for prognostic planning after patient's discharge from the hospital. 

<Prediction task picture>

![Screen Shot 2022-08-05 at 10 29 32 PM](https://user-images.githubusercontent.com/10427900/183233812-a8ea8824-6156-4554-8c37-e449541ff245.png)


## Requirements

For Tensorflow : 
This code was tested on Python 3 with tensorflow=2.4.1 or tensorflow-gpu=2.4.1

numpy>=1.20.3
pandas>=1.3.4
tensorflow==2.4
scipy>=1.6.2
scikit-learn>=0.24.2
tqdm>=4.26
xarray>=0.19
seaborn>=0.11.2
openpyxl>=3.0

## Deep learning algorithm

### ResNet (Based on the [1])

![image1](https://user-images.githubusercontent.com/10427900/180932275-5d9c976c-5ef5-4b51-a847-c97602460f44.png)

Input: X_ecg shape = (N, 4096, 12), tabular_feature shape = (N, tabular_size)
    
(Where X_ecg: 12 lead ECG voltage time series traces; tabular_feature: Age, Sex and Lab values; N: Number of instances)
    
In X_ecg: The leads are in following order: {I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}. 

All ECG signals and tabular features are represented as 32 bits floating point numbers. 

Output: shape = (N, label_size). Each entry of output is a prediction score between 0 and 1, that can be considered as the probability of mortality for a given patient.


## Synthetic Dataset
<Data Confidentiality Statement>
Original training and testing dataset couldn't be shared due to patient confidentiality and privacy reasons. We have included an ECG dataset that is artificially  generated using variational autoencoders for the purpose of code demonstration only. They are not expected to accurately represent real ECG signals. 

## Script

### train.py: Script for training the neural network. 
To train the neural network run:

`$ python3 train.py --method METHOD_NAME --label_path LABEL_PATH --tabular_path TABULAR_PATH --train_path TRAIN_PATH --val_path VAL_PATH --ecg_np_path ECG_PATH`

METHOD\_NAME: Choice of model architecture - 'ResNet' or 'DNN'

'ResNet' is default choice.
Optional: 'DNN': Deep Convolutional Neural Net based on [2], where X_ecg length is at least 5040 for Leads V1, II, V5, at least 1248 for other leads.   
    
LABEL\_PATH : path to the label dataframe, index is ECG_ID, shape is (N, label_size)
    
TABULAR\_PATH: path to the tabular dataframe, index is ECG_ID, shape is (N, feature_size)
    
TRAIN\_PATH: path to the list of training ECG_IDs
    
VAL\_PATH: path to the list of validation ECG_IDs

ECG\_PATH: each ECG file is in npy.gz format and named as {ECG_ID}.npy.gz 

### predict.py: Script for generating the neural network predictions on a given (test) dataset.

`$ python predict.py --method METHOD_NAME --model_path MODEL_PATH --label_path LABEL_PATH --tabular_path TABULAR_PATH --test_path TEST_PATH --ouput_file PATH_TO_OUTPUT_FILE --ecg_np_path ECG_PATH`

MODEL\_PATH: The trained model hdf5 file path.
    
PATH\_TO\_OUTPUT\_FILE: Output results (N, label_size) and each entry contains the probability scores between 0 and 1

TEST\_PATH: path to the list of testing ECG_IDs

## Reference

[1] Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4

[2] S. Raghunath, A. E. Ulloa Cerna, L. Jing, D. P. VanMaanen, J. Stough, D. N. Hartzel, J. B. Leader, H. L. Kirchner, M. C. Stumpe, A. Hafez, et al., “Prediction of mortality from 12-lead electrocardiogram voltage data using a deep neural network,” Nature medicine, vol. 26, no. 6, pp. 886–891, 2020.


