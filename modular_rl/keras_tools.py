from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation, Reshape, Permute
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.layers.normalization import  BatchNormalization
from keras.layers import Dense, Input

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.visualize_util import plot as keras_plot
from keras import backend as K

import numpy as np

def get_ch_axis(verbose=False):
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        if verbose: print('TF ORDERING IS USED')
        ch_axis = 3
    else:
        if verbose: print('TH ORDERING IS USED')
        ch_axis = 1
    return ch_axis


###############################################################################################
## GENERAL TOOLS

def spatial_softmax(x):
    """
    Applies softmax over the spatial dimensions of the image
    Instead of just the last dimension
    :param x:
    :return:
    """
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        x = Permute((1,3))

    x_shape = K.shape(x)
    x = Reshape((x_shape[1], x_shape[2]*x_shape[3]))(x)
    x = Activation('softmax')(x)
    x = Reshape((x_shape[1], x_shape[2], x_shape[3]))(x)

    if dim_ordering == 'tf':
        x = Permute((3,1))

    return x

def mean_coord(x):
    """
    Given a 4D tensor extracts mean coordinate of every feature map
    It is done by multiplying vector of coordinates of every pixel by
    normalized spatial features for every channel
    (normalization is done over spatial dimensions)
    :param x:
    :return: (Tensor [B,CH,XY]) XY - is dimension of height/width coordinate
    """
    # Applying spatial softmax on input features
    x = spatial_softmax(x)

    # Adjusting for dim ordering
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        if verbose: print('TF ORDERING IS USED')
        x = Permute((1,3))(x)

    # Finding parameters of the feature maps
    ch_axis = 1
    h_axis = 2
    w_axis = 3
    x_shape = K.shape(x)

    height = x_shape[h_axis]
    width  = x_shape[w_axis]

    # Forming vectors of coordinates of every pixel
    cnt_h_array = K.variable(np.array([i for i in range(0, height)]))
    cnt_w_array = K.variable(np.array([i for i in range(0, width)]))

    # Finding weights for coordinates
    h_weights = K.sum(axis=h_axis)(x)
    w_weights = K.sum(axis=w_axis)(x)

    # Weighting coordinates
    cnt_h_weighted = K.dot(h_weights * cnt_h_array)
    cnt_w_weighted = K.dot(w_weights * cnt_w_array)

    # Finding mean values
    h_mean = K.sum(cnt_h_weighted, axis=2)
    w_mean = K.sum(cnt_w_weighted, axis=2)

    # Packing everything together into a single Tensor
    hw_mean = K.merge([h_mean, w_mean], mode='concat', axis=2)

    if dim_ordering == 'tf':
        if verbose: print('TF ORDERING IS USED')
        hw_mean = Permute((2,1))(hw_mean)
    return hw_mean

####################################################################################################
# ARCHITECTURES

def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf', activation='relu'):
    s_id = 'fire' + str(fire_id) + '/'
    if dim_ordering == 'tf':
        print('TF ORDERING IS USED')
        c_axis = 3
    else:
        print('TH ORDERING IS USED')
        c_axis = 1


    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = BatchNormalization()(x)
    x = Activation(activation, name=s_id + activation + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = BatchNormalization()(left)
    left = Activation(activation, name=s_id + activation + exp1x1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = BatchNormalization()(right)
    right = Activation(activation, name=s_id + activation + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
    return x


def oclmnist_vis_feat(input_shape, out_num=128, activation='relu'):
    """
    Functional model for shared visual features
    :param out_num: (Int) number of classes
    :param input_shape:
    :param dim_ordering:
    :return:
    """
    input_img = Input(shape=input_shape)

    x = Convolution2D(32, 3, 3, border_mode='valid', name='conv1')(input_img)
    x = BatchNormalization()(x)
    x = Activation(activation, name='act_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64, dim_ordering=dim_ordering)

    ###############################################################
    # Non spatial features
    x_nonspat = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x_nonspat)

    x_nonspat = fire_module(x_nonspat, fire_id=4, squeeze=32, expand=128, dim_ordering=dim_ordering)
    x_nonspat = fire_module(x_nonspat, fire_id=5, squeeze=32, expand=128, dim_ordering=dim_ordering)
    x_nonspat = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x_nonspat)

    x_nonspat = Convolution2D(out_num, 1, 1, border_mode='valid', name='conv6')(x_nonspat)
    x_nonspat = Activation(activation, name='act_conv6')(x_nonspat)
    x_nonspat = GlobalAveragePooling2D()(x_nonspat)
    x_nonspat = Dense(128)(x_nonspat)
    x_nonspat = BatchNormalization(mode=1)(x_nonspat)

    ###############################################################
    # Spatial features
    x_spat = Convolution2D(32, 3, 3, border_mode='same', name='conv4_spat')(input_img)
    x_spat = mean_coord(x_spat)
    x_spat = Dense(128)(x_spat)
    x_spat = BatchNormalization(mode=1)(x_spat)
    ###############################################################
    # Merging
    x_merged = K.merge([x_spat, x_nonspat], mode='sum')
    out = Activation(activation, name='act_out')(x_merged)

    model = Model(input=input_img, output=[out])
    return model
