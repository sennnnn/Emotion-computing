import tensorflow as tf
import tensorflow.layers as layers

def C3D(input, filters, strides=1, kernel_size=3):
    """
    3-dimension convolution only
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor that has been operated.
    """
    input = layers.conv3d(input, filters, kernel_size, use_bias=True, strides=[1, strides, strides], padding='same', \
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))

    return input

def CB3D(input, filters, strides=1, kernel_size=3):
    """
    3-dimension convolution + batch normalization
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor that has been operated.
    """
    input = layers.conv3d(input, filters, kernel_size, use_bias=True, strides=[1, strides, strides], padding='same', \
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def CBR3D(input, filters, strides=1, kernel_size=3):
    """
    3-dimension convolution + batch normalization + leakyrelu
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor that has been operated.
    """
    input = layers.conv3d(input, filters, kernel_size, use_bias=True, strides=[1, strides, strides], padding='same', \
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')

    return input

def downS3D(input, kernel_size, pattern='max', filters=None):
    """
    3-dimension way
    Downsampling in order to concentrate the information containing in the input.
    The default pattern is maxPooling.
    Args:
        input:tensor that need to be operated.
        pattern:the method about how to downsampling.
                * max--maxPooling
                * avg--averagePooling
                * sto--stochasticPooling
                * con--convolution(stride=2)
    Return:
        input:tensor that has been operated.
    """
    if(pattern == 'avg'):
        input = layers.avg_pooling3d(input, kernel_size, strides=[1, 2, 2], padding='same')
    elif(pattern == 'sto'):
        pass
    elif(pattern == 'con'):
        filters = input.get_shape().as_list()[-1]*2 if filters == None else filters
        input = CBR3D(input, filters, kernel_size=kernel_size, strides=2)
    else:
        input = layers.max_pooling3d(input, kernel_size, strides=[1, 2, 2], padding='same')

    return input

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