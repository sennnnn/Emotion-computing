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

def res50(input, initial_channel=64, rate=0.1):
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

    input = output_layer(input, rate)

    return input

def output_layer(input, rate):
    """
    resnet output layer
    Args:
        input:tensor that need to be operated.
        rate:dropout layer keep probability.
    Return:
        input:tensor that has been operated.
    """
    input = layers.average_pooling2d(input, pool_size=7, strides=7, padding='same')
    input = layers.flatten(input)
    input = layers.dense(input, 1000, activation=tf.nn.softmax, use_bias=True,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = tf.nn.dropout(input, rate)

    return input

def restore_from_pb(sess, frozen_graph, meta_graph):
    """
    Ckpt restore from frozen model file.
    Args:
        sess:The Session that is connected with meta_graph.
        frozen_graph:The graph which will be used to restore.
        meta_graph:The graph which will be restored.
    Return:
        sess:The Session after processing.
    """ 
    # frozen_graph 与 meta_graph 应该是相互匹配的
    ops = frozen_graph.get_operations()
    ops_restore = [x.name.replace('/read','') for x in ops if('/read' in x.name)]
    tensors_constant = [frozen_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    tensors_variables = [meta_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    do_list = []
    sess_local = tf.Session(graph=frozen_graph)
    with tf.variable_scope('frozen_save'):
        for i in range(len(ops_restore)):
            try:
                temp = sess_local.run(tensors_constant[i])
                op = tf.assign(tensors_variables[i], temp)
                do_list.append(op)
            except:
                print('{} => {} error, frozen graph tensor name {} => meta graph tensor name {}'\
                      .format(tensors_constant[i].get_shape(), tensors_variables[i].get_shape(), 
                              tensors_constant[i].name, tensors_variables[i].name))
                exit()
        sess_local.close()
        sess.run(do_list)

    return sess

def load_graph(graph_def_filename):
    """
    To get graph from graph_def file.
    Args:
        graph_def_filename:The graph_def file save path.
    Return:
        graph:The graph loaded from graph_def file.
    """
    with tf.gfile.GFile(graph_def_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    print("load graph {} ...".format(graph_def_filename))

    return graph

def frozen_graph(sess, output_graph_path):
    """
    Extracting the sub graph of the current computing graph.
    Args:
        sess:The current session.
        output_graph_path:The frozen graph file save path.
    Return:
        string:The saving information.
    """
    # 因为计算图上只有ops没有变量，所以要通过会话，来获取那些变量
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   ["predict"])

    with open(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    return "{} ops written to {}.\n".format(len(output_graph_def.node), output_graph_path)


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

def construct_network(input, model, rate=0.1):
    """
    regression layer
    Args:
        input:the picture that wait to be processed.
        model:the neural network model.
        rate:the last layer dropout rate of the neural network model.
    Return:
        None
    """
    with tf.variable_score('network'):
        predict = model(input, rate=rate)
    
    predict = layers.dense(predict, 1, use_bias=True, name='regression_layer', \
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    predict = tf.identify(predict, name='predict')

if __name__ == "__main__":
    pass