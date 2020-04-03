import sys
sys.path.append('..')

import os
import re
import tensorflow as tf

from test import test
from train import train
from process import train_valid_generator,test_generator
from public.util.arg_parser import args_process
from public.util.util import open_readlines,train_valid_split,get_newest
from public.model.model_2D import VGG16_LSTM
from public.model.model_util import load_graph

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

ret_dict = args_process()

def record_split_by_video_name(path_list):
    record_dict = {}
    for one_record in path_list:
        video_name = re.search('MEDIAEVAL18_[0-9]*', one_record).group(0)
        if(video_name not in list(record_dict.keys())):
            record_dict[video_name] = []
        record_dict[video_name].append(one_record)

    return record_dict

def split_time_block(lines, time_block_length):
    record_dict = record_split_by_video_name(lines)
    time_block_list = []
    for key in record_dict.keys():
        videowise_record_list = record_dict[key]
        for i in range(0, len(videowise_record_list)-time_block_length+1):
            # print(i)
            time_block_list.append(videowise_record_list[i:i+time_block_length])
    
    return time_block_list

if(ret_dict['task'] == 'train'):
    # rarely changing options
    input_shape = (5, 224, 224)
    initial_channel = 64
    max_epoches = 20

    # usually changing options
    model_key = 'VGG16-LSTM'
    last = ret_dict['last']
    start_epoch = ret_dict['start_epoch']
    pattern = ret_dict['model_pattern']
    target = ret_dict['target']
    batch_size = 2
    learning_rate = 0.0001
    keep_prob = 0.5

    frozen_model_path = "build/{}-{}/frozen_model".format(target, model_key)
    ckpt_path = "build/{}-{}/ckpt".format(target, model_key)

    record_list = split_time_block(open_readlines('../dataset/trainval.txt'), 5)
    train_list,valid_list = train_valid_split(record_list, valid_rate=0.2, ifrandom=True)

    train_batch_generator = train_valid_generator(train_list, input_shape, batch_size, target=target)
    valid_batch_generator = train_valid_generator(valid_list, input_shape, batch_size, target=target)

    # load graph or init graph
    train_object = train(last, pattern, model_key, frozen_model_path, ckpt_path, target, initial_channel, input_shape)

    train_object.training(learning_rate, max_epoches, len(train_list)//batch_size, \
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

