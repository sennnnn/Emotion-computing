import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util.loss_metric import MSE_np,r_coefficient_np,MSE,r_coefficient
from process import slice_process,parse_va

class test(object):
    def __init__(self, graph, rate, tf_config):
        self.graph = graph
        if(target not in self.target_list): assert False, "The target input error."
        self.target = target
        self.tf_config = tf_config
        self.rate = rate

    def predict_op(self):
        self.p_op = self.graph.get_tensor_by_name('predict:0')

    def video_test(self, video_name, video_generator):
        self.predict_op()
        self.test_out_path = 'test/{}.txt'.format(video_name)
        pr_all = []
        label_all = []
        with tf.Session(config=self.tf_config, graph=self.graph) as sess:
            for pic,va in xxxx:
                pr_va = sess.run(self.p_op, feed_dict={'data:0':[pic], 'rate:0':self.rate})
                pr_all.append(pr_va[0])
                label_all.append(va)
            pr_all = np.array(pr_all, dtype=np.float32)
            label_all = np.array(label_all, dtype=np.float32)
            print(sess.run([MSE(pr_all, label_all), r_coefficient(pr_all, label_all)]))
        print(MSE_np(pr_all, label_all), r_coefficient_np(pr_all, label_all))

    def random_test(self):
        pass

if __name__ == "__main__":
    import tensorflow as tf
    import cv2

    from process import train_generator,test_generator,args_process,slice_process
    from util.util import open_readlines,train_valid_split,get_newest
    from model.model_util import load_graph

    input_shape = (448, 448)
    batch_size = 16
    keep_prob = 0.5

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    target = 'v'

    test_path_list = open_readlines('dataset/test.txt')

    test_batch_generator = test_generator(test_path_list, input_shape, batch_size, target)
    
    graph = load_graph(get_newest('{}/build/frozen_model'.format(target)))

    rate = 0.5

    pic = slice_process(pic, input_shape)

    with tf.Session(config=config, graph=graph) as sess:
        pr_va = sess.run(graph.get_tensor_by_name('predict:0'), feed_dict={'data:0':[pic], 'rate:0':keep_prob})
        print(pr_va)


    # test_object = test(graph, target, rate, config)

    # test_object.video_test('MEDIAEVAL18_54', test_batch_generator.videowise_iter('MEDIAEVAL18_54'))
    # def normal_test(self, test_time):
