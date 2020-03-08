import tensorflow as tf
import tensorflow.layers as layers

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def C(input, filters, strides=1, kernel_size=3):
    """
    convolution only
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor that has been operated.
    """
    # add kernel regular to avoid overfitting.
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    
    return input

def CB(input, filters, strides=1, kernel_size=3):
    """
    convolution + batch_normalization
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor that has been operated.
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def CBR(input, filters, strides=1, kernel_size=3):
    """
    convolution + batch normalization + leaky relu operation
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor that has been operated.
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    
    return input

def resB(input, filters):
    """
    The first generation residual block.
    Args:
        input:tensor that need to be operated.
        filters:the benchmark of the channel number.
    Return:
        input:tensor that has been operated.
    """
    raw = input
    input = CBR(input, filters)
    input = CB(input, filters)
    input = input + raw
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')

    return input

def bottleB(input, filters):
    """
    The second generation residual block.
    Args:
        input:tensor that need to be operated.
        filters:the benchmark of the channel number.
    Return:
        input:tensor that has been operated.
    """
    raw=input
    input = CBR(input, filters//4, kernel_size=1)
    input = CBR(input, filters//4, kernel_size=3)
    input = CB(input, filters, kernel_size=1)
    input = input + raw
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')

    return input

def downS(input, kernel_size, pattern='max'):
    """
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
        input = layers.avg_pooling2d(input, kernel_size, strides=2, padding='same')
    elif(pattern == 'sto'):
        pass
    elif(pattern == 'con'):
        filters = input.get_shape().as_list()[-1]
        input = CBR(input, filters, kernel_size, strides=2)
    else:
        input = layers.max_pooling2d(input, kernel_size, strides=2, padding='same')

    return input

def res50(input, initial_channel=64, downsampling_pattern='max'):
    """
    resnet 50 network architecture
    backbone
    Args:
        input:tensor that wait to be operated.
        initial_channel:the benchmark of the channels.
        downsampling_pattern:the pattern of the downsampling.
    Return:
        input:the network output.
    """
    c = initial_channel
    p = downsampling_pattern
    # conv1
    input = CBR(input, c, strides=2, kernel_size=7)
    # conv2
    i = i*1
    input = downS(input, kernel_size=3, p)
    for i in range(3):
        input = bottleB(input, i)
    # conv3
    i = i*2
    input = downS(input, kernel_size=3, p)
    for i in range(4):
        input = bottleB(input, i)
    # conv4
    i = i*2
    input = downS(input, kernel_size=3, p)
    for i in range(6):
        input = bottleB(input, i)
    # conv5
    i = i*2
    input = downS(input, kernel_size=3, p)
    for i in range(3):
        input = bottleB(input, i)

    input = output_layer(input)

    return input

def output_layer(input):
    """
    resnet output layer
    Args:
        input:tensor that need to be operated.
    Return:
        input:tensor that has been operated.
    """
    input = layers.average_pooling2d(input, pool_size=7, strides=7, padding='same')
    input = layers.flatten(input)
    input = layers.dense(input, 1000, activation=tf.nn.softmax, use_bias=True,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    
    return input