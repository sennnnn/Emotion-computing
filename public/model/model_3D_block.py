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