import tensorflow as tf
import tensorflow.layers as layers

from model_2D_block import *

def res50(input, last_channel, initial_channel=64, rate=0.5, top=True):
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

    # conv2
    c = c*1
    input = downS(input, 1, 'max')
    for i in range(3):
        input = bottleB(input, c)

    # conv3
    c = c*2
    input = downS(input, 1, 'con', c)
    for i in range(4):
        input = bottleB(input, c)

    # conv4
    c = c*2
    input = downS(input, 1, 'con', c)
    for i in range(6):
        input = bottleB(input, c)

    # conv5
    c = c*2
    input = downS(input, 1, 'con', c)
    for i in range(3):
        input = bottleB(input, c)

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
    print(input)
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
    input = layers.dense(input, output_units, activation=None, use_bias=True)
    input = tf.nn.dropout(input, rate)

    return input

def VGG16_stacked(input, initial_channel=64, rate=0.5):
    out = []
    input_list = tf.unstack(input, axis=1)
    for single in input_list:
        out.append(VGG16(single, top=True))

    out = tf.stack(out, axis=1)

    return out

def construct_network(input, model, initial_channel=64, rate=0.1):
    """
    deperacted
    This way can't let the variable share easy.
    regression layer
    Args:
        input:the picture that wait to be processed.
        model:the neural network model.
        rate:the last layer dropout rate of the neural network model.
    Return:
        None
    """
    with tf.variable_scope('network'):
        predict = model(input, initial_channel=initial_channel, rate=rate)
    
    predict = layers.dense(predict, 512, use_bias=True)
    predict = layers.dense(predict, 512, use_bias=True)
    predict = layers.dense(predict, 1, use_bias=True, name='regression_layer')
    predict = tf.identity(predict, name='predict')

def VGG16_LSTM(input, rnn_units,  RNN_layer=1):
    """
    stacked VGG16 cascade LSTM Cell.
    input must shape as [batch_size, sequence time length, state length(RNN unit size)]
    Args:
        input:the network input.
        RNN_layer:the RNN layer count.
    Return:
        input:the network output.
    """
    LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_units)
    LSTM_cell = tf.nn.rnn_cell.MultiRNNCell([LSTM_cell]*RNN_layer)
    out,state = tf.nn.dynamic_rnn(LSTM_cell, input, dtype=tf.float32)
    
    predict = layers.dense(predict, 512, use_bias=True)
    predict = layers.dense(predict, 512, use_bias=True)
    predict = layers.dense(predict, 1, use_bias=True, name='regression_layer')
    predict = tf.identity(predict, name='predict')

    return predict

if __name__ == "__main__":
    # test region
    input = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # input = VGG16_stacked(input)
    # input = VGG16(input)
    print(densenet(input))