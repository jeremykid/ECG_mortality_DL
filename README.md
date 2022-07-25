# Scripts and modules for training and testing of deep learning models for predicting mortality using ECGs
<This is the companion code base for the paper the 'title of the paper'. 
<Prediction task picture>

## Requirements

For Tensorflow Version 
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

For PyTorch Version
<list out the requirements>


## Deep learning algorithms
We used two different deep learning frameworks as listed below. 
This is single or multiple binary label prediction task - eg: 30-day mortality, 365-day mortality, 5-year mortality

### 1) ResNet

<Model arch figure>

Input: X_ecg shape = (N, 4096, 12), tabular_feature shape = (N, tabular_size)
<Where X_ecg: 12 lead ECG voltage time series traces, tabular_feature: Age, Sex and Lab, N: Number of instances>
In X_ecg: The leads are following order: {I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}. 
All ECG signals and tabular features are represented as 32 bits floating point numbers. 

Output: shape = (N, label_size). Each entry of output is the probablities between 0 and 1, and  can be understood as the probability of mortality for a given patient.


### 2) Raghunath_DNN

Input: X_ecg shape = (N, 5040, 12) since 
<same as above>

# Synthetic Dataset
<Data Confidentiality Statement>
<to do... >
<This dataset is artifically generated using variational autoencoders for purpose of code demostration only. They are not expected to accurately represent real ECG signals> 

## Script

train.py: Script for training the neural network. To train the neural network run:

python3 train.py --method automatic_ecg --label_path label_df.pickle --tabular_path tabular_df.pickle --train_path train_ID_list.pickle --val_path val_ID_list.pickle

method: Choice of model architecture - 'automatic_ecg' or 'raghunath_cnn_ecg' <change names>
label_path : path to label dataframe, index is ECG_IDs, shape is (N, label_size)
tabular_path: path to tabular dataframe, index is ECG_IDs, shape is (N, feature_size)
train_path: path to list of training ECG_IDs
val_path: path list of validation ECG_IDs

ECG path: each ECG file is npy.gz format and name as {ECG_ID}npy.gz <update>

predict.py: Script for generating the neural network predictions on a given dataset.
$ python predict.py --method automatic_ecg --test_path test_ID_list.pickle --ouput_file PATH_TO_OUTPUT_FILE 


## Reference

[1] Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4

[2] S. Raghunath, A. E. Ulloa Cerna, L. Jing, D. P. VanMaanen, J. Stough, D. N. Hartzel, J. B. Leader, H. L. Kirchner, M. C. Stumpe, A. Hafez, et al., “Prediction of mortality from 12-lead electrocardiogram voltage data using a deep neural network,” Nature medicine, vol. 26, no. 6, pp. 886–891, 2020.


