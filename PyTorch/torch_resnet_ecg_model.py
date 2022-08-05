import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from typing import Type, Any, Callable, Union, List, Optional

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
class ResidualUnit(nn.Module):
    """Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default it uses
        'he_normal'.
    dropout_keep_prob: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu'.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(
        self,
        n_samples_out, 
        n_filters_out, 
        n_filters_in,
        kernel_initializer='he_normal',
        dropout_keep_prob=0.8, 
        kernel_size=17, 
        preactivation=True,
        postactivation_bn=False, 
        activation_function='relu'
    ):
        super(ResidualUnit, self).__init__()
        self.n_samples_out = n_samples_out
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        ##### layer ######
        self.conv1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        # @TODO downsample = 4 always? 
        self.conv2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, kernel_size=1, stride=4, padding=0, bias= False)
        self.drop1 = nn.Dropout(p=self.dropout_rate)
        self.bn1 = nn.BatchNorm1d(self.n_filters_out)   
        self.conv_skip = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        self.activition = nn.ReLU(inplace=True) 
        self.conv_skip = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        
    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            maxpool1 = nn.MaxPool1d(downsample, stride=downsample)
            y = maxpool1(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.

            y = self.conv_skip(y)

        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = self.activition(x)
            x = self.bn1(x)
        else:
            x = self.bn1(x)
            x = self.activition(x)
        return x
    
    def forward(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[2]#y.shape[2]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[1]#y.shape[1]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
#         self.conv1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        x = self.conv1(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = self.drop1(x)

        # 2nd layer
#         self.conv2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, kernel_size=1, stride=4, padding=0, bias= False)
        x = self.conv2(x)
        if self.preactivation:
            x = torch.add(x,y)  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = self.drop1(x)
        else:
            x = self.bn1(x)
            x = torch.add(x,y)  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = self.drop1(x)
            y = x
        return [x, y]

class ECG_ResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
    ):
        super(ECG_ResNet, self).__init__()
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        self.kernel_size = 16
        self.conv1 = nn.Conv1d(12, 64, kernel_size=1, stride=1, padding=0, bias= False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.kernel_initializer = 'he_normal'
        #4096
        self.res1 = ResidualUnit(1024, 128, 64, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res2 = ResidualUnit(256, 196, 128, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res3 = ResidualUnit(64, 256, 196, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res4 = ResidualUnit(16, 320, 256, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.flatten = nn.Flatten()
        self.dense_agsx = nn.Linear(2, 10)
        self.dense = nn.Linear(5130, num_classes)

    def forward(self, x):
        signal = x[0]
        age_sex = x[1]
        
        x1 = self.conv1(signal)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x1, y = self.res1([x1, x1])
        x1, y = self.res2([x1, y])
        x1, y = self.res3([x1, y])
        x1, _ = self.res4([x1, y])
        x1 = self.flatten(x1)

        x2 = self.dense_agsx(age_sex)
        x = torch.cat([x1, x2], dim=1)
        
        x = self.dense(x)
        result = self.sigmoid(x)
        # print (result.shape)        
        return result
