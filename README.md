# ECG_mortality_model



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

## automatic ResNet Model[1]

Input: X_ecg shape = (N, 4096, 12), demographic_feature shape = (N, demographic_size)

In X_ecg: The leads are following order: {I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}. All signal are represented as 32 bits floating point numbers. 

In demographic_feature: The features are represent as 32 bits floating point numbers. 

Output: shape = (N, label_size)

In training dataset, there are binary bits for each entry. In model output, output is the probablity between 0 and 1

## raghunath_model[2]

Input: X_ecg shape = (N, 5040, 12) since 


## Script

In tensorflow

Method: 'automatic_ecg' or 'raghunath_cnn_ecg'

label_df: dataframe, index is ECG_IDs, shape is (N, label_size)

demographic_df: dataframe, index is ECG_IDs, shape is (N, feature_size)

train_ID_list: list of training ECG_IDs

val_ID_list: list of validation ECG_IDs

ECG path: each ECG file is npy.gz format and name as {ECG_ID}npy.gz

python3 train.py --method automatic_ecg --label_path label_df.pickle --demographic_path demographic_df.pickle --train_path train_ID_list.pickle --val_path val_ID_list.pickle

## Reference

[1] Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4

[2] S. Raghunath, A. E. Ulloa Cerna, L. Jing, D. P. VanMaanen, J. Stough, D. N. Hartzel, J. B. Leader, H. L. Kirchner, M. C. Stumpe, A. Hafez, et al., “Prediction of mortality from 12-lead electrocardiogram voltage data using a deep neural network,” Nature medicine, vol. 26, no. 6, pp. 886–891, 2020.


