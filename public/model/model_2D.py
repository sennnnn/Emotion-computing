import tensorflow as tf
import tensorflow.layers as layers

from .model_2D_block import *

def res50(input, last_channel, initial_channel=256, rate=0.5, top=True):
    """
    resnet 50 network architecture
    backbone
    Args:
        input:tensor that wait to be operated.
        last_channel:last layer output channel number.
        initial_channel:the benchmark of the channels.
        rate:the dropout layer keep probability.
    Return:
        input:the network output.
    """
    c = initial_channel
    # conv1
    input = CBR(input, c, strides=2, kernel_size=7)
    input = downS(input, 3, 'max')

    # conv2
    c = c*1
    input = bottle_neck_downS(input, c)
    for i in range(2):
        input = bottle_resB(input, c)

    # conv3
    c = c*2
    input = bottle_neck_downS(input, c)
    for i in range(3):
        input = bottle_resB(input, c)

    # conv4
    c = c*2
    input = bottle_neck_downS(input, c)
    for i in range(5):
        input = bottle_resB(input, c)

    # conv5
    c = c*2
    input = bottle_neck_downS(input, c)
    for i in range(2):
        input = bottle_resB(input, c)

    if(top):
        input = output_layer(input, rate, last_channel)

    return input

def VGG16(input, last_channel, initial_channel=64, rate=0.5, top=True):
    """
    VGG16 network architecture
    Args:
        input:tensor that wait to be operated.
        last_channel:last layer output channel number.
        initial_channel:the benchmark of the channels.
        rate:the dropout layer keep probability.
    Return:
        input:the network output.
    """
    c = initial_channel
    for i in range(2):
        input = CBR(input, c, kernel_size=3)
    
    input = downS(input, 3)
    c = c*2
    for i in range(2):
        input = CBR(input, c, kernel_size=3)
    
    input = downS(input, 3)
    c = c*2
    for i in range(2):
        input = CBR(input, c, kernel_size=3)
    input = CBR(input, c, kernel_size=1)

    input = downS(input, 3)
    c = c*2
    for i in range(2):
        input = CBR(input, c, kernel_size=3)
    input = CBR(input, c, kernel_size=1)
    
    input = downS(input, 3)
    for i in range(2):
        input = CBR(input, c, kernel_size=3)
    input = CBR(input, c, kernel_size=1)

    if(top):
        input = output_layer(input, rate, last_channel)

    return input

def densenet(input, last_channel, initial_channel=64, rate=0.5, top=True):
    """
    Densenet network architecture
    backbone
    Args:
        input:tensor that wait to be operated.
        last_channel:last layer output channel number.
        initial_channel:the benchmark of the channels.
        rate:the dropout layer keep probability.
    Return:
        input:the network output.
    """
    c = initial_channel
    input = C(input, c, strides=2, kernel_size=7)
    input = downS(input, 3)
    # Dense block 1
    input = dense_block(input, c, 6)
    # Trainsition layer 1
    input = transitionB(input)
    # Dense block 2
    input = dense_block(input, c, 12)
    # Trainsition layer 2
    input = transitionB(input)
    # Dense block 3
    input = dense_block(input, c, 48)
    # Trainsition layer 3
    input = transitionB(input)
    # Dense block 4
    input = dense_block(input, c, 32)

    if(top):
        input = output_layer(input, rate, last_channel)

    return input

def output_layer(input, rate, output_units):
    """
    CNN output layer or feature extraction layer.
    Args:
        input:tensor that need to be operated.
        rate:dropout layer keep probability.
        output_units:last fully connection layer output size.
    Return:
        input:tensor that has been operated.
    """
    input = layers.average_pooling2d(input, pool_size=7, strides=7, padding='same')
    input = layers.flatten(input)
    input = layers.dense(input, output_units, use_bias=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    input = tf.nn.dropout(input, rate)

    return input

def VGG16_stacked(input, last_channel, initial_channel=64, rate=0.5, reuse=True):
    if(reuse):
        reuse = tf.AUTO_REUSE
    else:
        reuse = False
    out = []
    input_list = tf.unstack(input, axis=1)
    index = 0
    for single in input_list:
        if(index == 0):
            with tf.variable_scope('VGG16', reuse=False):
                out.append(VGG16(single, last_channel, rate=rate, top=True))
            index += 1
        else:
            with tf.variable_scope('VGG16', reuse=reuse):
                out.append(VGG16(single, last_channel, rate=rate, top=True))

    out = tf.stack(out, axis=1)

    return out

def VGG16_LSTM(input, rnn_units=1024, initial_channel=64, rate=0.5, RNN_layer=1):
    """
    stacked VGG16 cascade LSTM Cell.
    input must shape as [batch_size, sequence time length, state length(RNN unit size)]
    Args:
        input:the network input.
        RNN_layer:the RNN layer count.
    Return:
        input:the network output.
    """
    input = VGG16_stacked(input, rnn_units, initial_channel=initial_channel, rate=rate)
    LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_units)
    LSTM_cell = tf.nn.rnn_cell.MultiRNNCell([LSTM_cell]*RNN_layer)
    predict,state = tf.nn.dynamic_rnn(LSTM_cell, input, dtype=tf.float32)

    return predict