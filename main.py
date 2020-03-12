import os
import tensorflow as tf

from train import train
from process import train_generator,args_process
from util.util import open_readlines,train_valid_split
from model.model_old import res50

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
    last = ret_dict['last']
    start_epoch = ret_dict['start_epoch']
    pattern = ret_dict['model_pattern']
    batch_size = 16
    learning_rate = 0.0001
    keep_prob = 0.5

    frozen_model_path = "build/frozen_model"
    ckpt_path = "build/ckpt"

    train_path_list,valid_path_list = train_valid_split(open_readlines('dataset/trainval.txt'))

    train_batch_generator = train_generator(train_path_list, input_shape, batch_size, 'v')
    valid_batch_generator = train_generator(valid_path_list, input_shape, batch_size, 'v')

    # load graph or init graph
    train_object = train(last, pattern, res50, frozen_model_path, ckpt_path, initial_channel)

    train_object.training(learning_rate, max_epoches, len(train_path_list)//batch_size, \
                        start_epoch, train_batch_generator, valid_batch_generator, 3, 100, keep_prob, config)

elif(ret_dict['task'] == 'test'):
    pass
else:
    print('Sorry,{} isn’t a valid option'.format(sys.argv[1]))
    exit()

