import os
import tensorflow as tf

from test import test
from train import train
from process import train_generator,test_generator,args_process
from util.util import open_readlines,train_valid_split,get_newest
from model.model_2D import VGG16
from model.model_util import load_graph

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

ret_dict = args_process()

if(ret_dict['task'] == 'train'):
    # rarely changing options
    input_shape = (448, 448)
    initial_channel = 64
    max_epoches = 200

    # usually changing options
    strategy = ret_dict['strategy']
    last = ret_dict['last']
    start_epoch = ret_dict['start_epoch']
    pattern = ret_dict['model_pattern']
    batch_size = 16
    learning_rate = 0.0001
    keep_prob = 0.5

    frozen_model_path = "{}/build/frozen_model".format(strategy)
    ckpt_path = "{}/build/ckpt".format(strategy)

    train_path_list,valid_path_list = train_valid_split(open_readlines('dataset/trainval.txt'))

    train_batch_generator = train_generator(train_path_list, input_shape, batch_size)
    valid_batch_generator = train_generator(valid_path_list, input_shape, batch_size)

    # load graph or init graph
    train_object = train(last, pattern, VGG16, frozen_model_path, ckpt_path, initial_channel, strategy)

    train_object.training(learning_rate, max_epoches, len(train_path_list)//batch_size, \
                        start_epoch, train_batch_generator, valid_batch_generator, 3, 1, keep_prob, config)

elif(ret_dict['task'] == 'test'):
    
    strategy = ret_dict['strategy']
    input_shape = (448, 448)
    batch_size = 16
    keep_prob = 0.5

    test_path_list = open_readlines('dataset/test.txt')

    test_batch_generator = test_generator(test_path_list, input_shape, batch_size)
    
    graph = load_graph(get_newest('build/frozen_model'.format(target)))
    test_object = test(graph, 0.5, config)

    # test_object.video_test('MEDIAEVAL18_54', test_batch_generator.videowise_iter('MEDIAEVAL18_54'))

else:
    print('Sorry,{} isn’t a valid option'.format(sys.argv[1]))
    exit()

