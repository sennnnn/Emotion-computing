import tensorflow as tf
import tensorflow.layers as layers

from model_3D_block import *

def VGG16_3D(input, initial_channel=64, rate=0.5, top=True):
    """
    3-dimension VGG16 network
    Args:
        input:tensor that wait to be operated.
        initial_channel:the benchmark of the channels.
        rate:the dropout layer keep probability.
    Return:
        input:the network output.
    """
    c = initial_channel
    for i in range(2):
        input = CBR3D(input, c, kernel_size=3)
    
    input = downS3D(input, 3)
    c = c*2
    for i in range(2):
        input = CBR3D(input, c, kernel_size=3)
    
    input = downS3D(input, 3)
    c = c*2
    for i in range(2):
        input = CBR3D(input, c, kernel_size=3)
    input = CBR3D(input, c, kernel_size=1)

    input = downS3D(input, 3)
    c = c*2
    for i in range(2):
        input = CBR3D(input, c, kernel_size=3)
    input = CBR3D(input, c, kernel_size=1)
    
    input = downS3D(input, 3)
    for i in range(2):
        input = CBR3D(input, c, kernel_size=3)
    input = CBR3D(input, c, kernel_size=1)

    if(top):
        input = output_layer3D(input, rate)

    return input

def output_layer(input, rate):
    """
    2DCNN output layer or feature extraction layer.
    Args:
        input:tensor that need to be operated.
        rate:dropout layer keep probability.
    Return:
        input:tensor that has been operated.
    """
    input = layers.average_pooling2d(input, pool_size=7, strides=7, padding='same')
    input = layers.flatten(input)
    input = layers.dense(input, 2048, activation=None, use_bias=True)
    input = tf.nn.dropout(input, rate)

    return input

def output_layer3D(input, rate):
    """
    CNN3D output layer or feature extraction layer
    This isn't same as the regular CNN3D output layer, and this will output a stacked feature
    Args:
        input:tensor that need to be operated.
        rate:dropout layer keep probability.
    Return:
        out:tensor that has been operated.
    """
    input_list = tf.unstack(input, axis=1)
    out = []
    for single in input_list:
        out.append(output_layer(single, rate))
    
    out = tf.stack(out, axis=1)

    return out