import sys
sys.path.append('..')

import os
import tensorflow as tf

from test import test
from train import train
from process import train_valid_generator,test_generator
from public.util.arg_parser import args_process
from public.util.util import open_readlines,train_valid_split,get_newest
from public.model.model_2D import res50
from public.model.model_util import load_graph

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

ret_dict = args_process()

if(ret_dict['task'] == 'train'):
    # rarely changing options
    input_shape = (224, 224)
    initial_channel = 256
    max_epoches = 20

    # usually changing options
    last = ret_dict['last']
    start_epoch = ret_dict['start_epoch']
    pattern = ret_dict['model_pattern']
    target = ret_dict['target']
    batch_size = 4
    learning_rate = 0.0001
    keep_prob = 0.5

    frozen_model_path = "{}/build/frozen_model".format(target)
    ckpt_path = "{}/build/ckpt".format(target)

    train_path_list,valid_path_list = train_valid_split(open_readlines('../dataset/trainval.txt'))

    train_batch_generator = train_valid_generator(train_path_list, input_shape, batch_size)
    valid_batch_generator = train_valid_generator(valid_path_list, input_shape, batch_size)

    # load graph or init graph
    train_object = train(last, pattern, res50, frozen_model_path, ckpt_path, target, initial_channel, input_shape)

    train_object.training(learning_rate, max_epoches, len(train_path_list)//batch_size, \
                        start_epoch, train_batch_generator, valid_batch_generator, 3, 100, keep_prob, config)

elif(ret_dict['task'] == 'test'):
    
    strategy = ret_dict['strategy']
    input_shape = (224, 224)
    batch_size = 4
    keep_prob = 0.5

    test_path_list = open_readlines('../dataset/test.txt')

    test_batch_generator = test_generator(test_path_list, input_shape, batch_size)
    
    graph = load_graph(get_newest('build/frozen_model'.format(target)))
    test_object = test(graph, 0.5, config)

    # test_object.video_test('MEDIAEVAL18_54', test_batch_generator.videowise_iter('MEDIAEVAL18_54'))

else:
    print('Sorry,{} isn’t a valid option'.format(sys.argv[1]))
    exit()

