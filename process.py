import os
import re
import cv2
import random
import numpy as np

from util.arg_parser import arg_parser
from util.util import dict_load

def args_process():

    ret_dict = {}

    a = arg_parser()

    a.add_val('--task', 'train')

    model_pattern_dict = {'ckpt':'ckpt', 'pb':'pb'}

    a.add_map('--pattern', model_pattern_dict)

    strategy_dict = {'0':'frame-base', '1':'rnn-base'}

    a.add_map('--strategy', strategy_dict)

    parse_dict = a()

    training_metric_loss_only_log_path = os.path.join(parse_dict['--strategy'], 'build', 'valid_metric_loss_only.log')
    start_epoch = 1
    if(not os.path.exists(training_metric_loss_only_log_path)):
        last = False
        dir_path = os.path.dirname(training_metric_loss_only_log_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, 0x777)
    else:
        try:
            temp_dict = dict_load(training_metric_loss_only_log_path)
            start_epoch = len(temp_dict['epochwise']['metric']) + 1
            if(start_epoch == 0):
                last = False
            else:
                last = True
        except:
            last = False

    ret_dict['strategy'] = parse_dict['--strategy']
    ret_dict['task'] = parse_dict['--task']
    ret_dict['model_pattern'] = parse_dict['--pattern'] if('--pattern' in parse_dict.keys()) else None
    ret_dict['last'] = last
    ret_dict['start_epoch'] = start_epoch

    return ret_dict

def z_score(pic, smooth=0.000001):
    """
    z-score standardization
    Args:
        pic:the picture that wait to be processed.
    Return:
        pic:the picture that has been processed.
    """
    mean,stdDev = cv2.meanStdDev(pic)
    mean,stdDev = mean[0][0],stdDev[0][0]
    pic = (pic-mean+smooth)/(stdDev+smooth)

    return pic

def slice_process(slice, input_shape):
    slice = slice.astype(np.float32)
    slice = cv2.resize(slice, input_shape)
    slice[:,:,0] = z_score(slice[:,:,0])
    slice[:,:,1] = z_score(slice[:,:,1])
    slice[:,:,2] = z_score(slice[:,:,2])

    return slice

def parse_va(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        va = lines[0].split()
        va = [float(x) for x in va]

        return va

class test_generator():
    def __init__(self, test_list, input_shape, batch_size):
        self.test_list = test_list
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.target = v_or_a

    def videowise_split(self):
        self.videowise_path_list = {}
        for line in self.test_list:
            instance_name = re.search('MEDIAEVAL18_[0-9]*', line).group(0)
            if(instance_name not in list(self.videowise_path_list.keys())):
                self.videowise_path_list[instance_name] = []
            else:
                self.videowise_path_list[instance_name].append(line)

    def videowise_iter(self, video_name):
        self.videowise_split()
        video_data_list = self.videowise_path_list[video_name]
        data_batch_block = []
        label_batch_block = []
        count = 0
        for item in video_data_list:
            count += 1
            pic_path = item + '.jpg'
            txt_path = item + '.txt'
            pic = cv2.imread(pic_path)
            pic = slice_process(pic, self.input_shape)
            va = parse_va(txt_path)
            data_batch_block.append(pic)
            label_batch_block.append(va)
            if(count == self.batch_size):
                yield data_batch_block,label_batch_block
                data_batch_block.clear()
                label_batch_block.clear()
                count = 0

        if(count != 0): yield data_batch_block,label_batch_block

    def __iter__(self):
        random.shuffle(self.test_list)
        data_batch_block = []
        label_batch_block = []
        count = 0
        for item in self.test_list:
            count += 1
            pic_path = item + '.jpg'
            txt_path = item + '.txt'
            pic = cv2.imread(pic_path)
            pic = slice_process(pic, self.input_shape)
            va = parse_va(txt_path)
            data_batch_block.append(pic)
            label_batch_block.append(va)
            
            if(count == self.batch_size):
                yield data_batch_block,label_batch_block
                data_batch_block.clear()
                label_batch_block.clear()
                count = 0

        if(count != 0): yield data_batch_block,label_batch_block

    def test_generator_timewise(self):
        while(1):
            yield from self.__iter__()

class train_generator():
    def __init__(self, train_list, input_shape, batch_size, ifrandom=True, v_or_a='v'):
        self.train_list = train_list
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.ifrandom = ifrandom
        self.count = len(train_list)
        self.target = v_or_a
    
    def __iter__(self):
        if(self.ifrandom):
            random.shuffle(self.train_list)
        
        data_batch_block = []
        label_batch_block = []
        count = 0
        for item in self.train_list:
            count += 1
            pic_path = item + '.jpg'
            txt_path = item + '.txt'
            pic = cv2.imread(pic_path)
            pic = slice_process(pic, self.input_shape)
            va = parse_va(txt_path)
            data_batch_block.append(pic)
            label_batch_block.append(va)

            if(count == self.batch_size):
                yield data_batch_block,label_batch_block
                data_batch_block.clear()
                label_batch_block.clear()
                count = 0

        if(count != 0): yield data_batch_block,label_batch_block

    def epochwise_generator(self):
        while(1):
            yield from self.__iter__()

if __name__ == "__main__":
    # example about how to use train generator.
    pass