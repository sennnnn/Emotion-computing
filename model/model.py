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
    raw=input
    input = CBR(input, filters//4, kernel_size=1)
    input = CBR(input, filters//4, kernel_size=3)
    input = CB(input, filters, kernel_size=1)
    input = input + raw
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

def transitionB(input, filters):
    """
    Densenet trainsition layer
    Args:
        input:tensor that need to be operated.
        filters:the benchmark of the channel number.
    Return:
        input:tensor that has been operated.
    """
    input = 

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
        input = layers.avg_pooling2d(input, kernel_size, strides=2, padding='same')
    elif(pattern == 'sto'):
        pass
    elif(pattern == 'con'):
        filters = input.get_shape().as_list()[-1]*2 if filters == None else filters
        input = CBR(input, filters, kernel_size=kernel_size, strides=2)
    else:
        input = layers.max_pooling2d(input, kernel_size, strides=2, padding='same')

    return input

def res50(input, initial_channel=64, rate=0.5, top=True):
    """
    resnet 50 network architecture
    backbone
    Args:
        input:tensor that wait to be operated.
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
        input = output_layer(input, rate)

    return input

def VGG16(input, initial_channel=64, rate=0.5, top=True):
    """
    VGG16 network architecture
    Args:
        input:tensor that wait to be operated.
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

    input = output_layer(input, rate)

    return input

def VGG16_stacked(input, initial_channel=64, rate=0.5):
    out = []
    input_list = tf.unstack(input, axis=1)
    for single in input_list:
        out.append(VGG16(single))

    out = tf.stack(out, axis=1)

    return out

def output_layer(input, rate):
    """
    CNN output layer or feature extraction layer.
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
    
def construct_network(input, model, initial_channel=64, rate=0.1):
    """
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

# def densenet(input, initial_channel=64, rate=0.1):
#     """
#     resnet 50 network architecture
#     backbone
#     Args:
#         input:tensor that wait to be operated.
#         initial_channel:the benchmark of the channels.
#         rate:the dropout layer keep probability.
#     Return:
#         input:the network output.
#     """

if __name__ == "__main__":
    # test region
    input = tf.placeholder(tf.float32, [None, 224, 224, 224, 3])
    # input = VGG16_stacked(input)
    # input = VGG16(input)
    input = VGG16_3D(input, top=True)
    print(input)