import sys
sys.path.append('..')

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from process import slice_process,parse_va
from public.util.loss_metric import MSE_np,r_coefficient_np,MSE,r_coefficient

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
