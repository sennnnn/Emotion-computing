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

def bottle_resB(input, filters):
    """
    The second generation residual block.
    Args:
        input:tensor that need to be operated.
        filters:the benchmark of the channel number.
    Return:
        input:tensor that has been operated.
    """
    raw = input
    input = CBR(input, filters//4, kernel_size=1)
    input = CBR(input, filters//4, kernel_size=3)
    input = CB(input, filters, kernel_size=1)
    input = input + raw
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')

    return input

def bottle_neck_downS(input, filters):
    """
    The res50 downsampling block.
    Args:
        input:tensor that need to be operated.
        filters:the benchmark of the channel number.
    Return:
        input:tensor that has been operated.
    """
    shortcut = CB(input, filters, strides=2)
    input = CBR(input, filters//4, strides=2, kernel_size=1)
    input = CBR(input, filters//4, kernel_size=3)
    input = CB(input, filters, kernel_size=1)
    input = input + shortcut
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')

    return input

def bottle_neckB(input, filters):
    """
    Densenet block
    Args:
        input:tensor that need to be operated.
        filters:the benchmark of the channel number.
    Return:
        input:tensor that has been operated.
    """
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    input = C(input, filters, kernel_size=1)
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    input = C(input, filters, kernel_size=3)

    return input

def transitionB(input):
    """
    Densenet trainsition layer
    Args:
        input:tensor that need to be operated.
    Return:
        input:tensor that has been operated.
    """
    filters = input.get_shape().as_list()[-1]
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = C(input, filters, kernel_size=1)
    input = layers.average_pooling2d(input, 3, strides=2, padding='same')

    return input

def downS(input, kernel_size, pattern='max', filters=None):
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
        input = layers.average_pooling2d(input, kernel_size, strides=2, padding='same')
    elif(pattern == 'sto'):
        pass
    elif(pattern == 'con'):
        filters = input.get_shape().as_list()[-1]*2 if filters == None else filters
        input = CBR(input, filters, kernel_size=kernel_size, strides=2)
    else:
        input = layers.max_pooling2d(input, kernel_size, strides=2, padding='same')

    return input

def dense_block(input, filters, recurrent_round):
    """
    Densenet main dense connect block
    Args:
        input:tensor that need to be accumalated.
        filters:the benchmark of the channel number.
        recurrent_round:the dense connect depth.
    Return:
        out:tensor that has been concentrated.
    """
    last_concat = []
    for _ in range(recurrent_round):
        last_concat.append(input)
        input = tf.concat(last_concat, axis=-1)
        print(input.get_shape().as_list())
        input = bottle_neckB(input, filters)

    return input

if __name__ == "__main__":
    pass