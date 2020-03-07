import os
import cv2

root_path = r'E:\dataset\emotion_computating\MediaEval\2018'
data_path = os.path.join(root_path, 'data')
label_path = os.path.join(root_path, 'label')
train_data_path = os.path.join(data_path, 'trainval')
train_label_path = os.path.join(label_path, 'trainval')

train_data_list = os.listdir(train_data_path)
train_label_list = os.listdir(train_label_path)

build_path = 'dataset'
build_data_path = os.path.join(build_path, 'data')
build_label_path = os.path.join(build_path, 'label')

for vi,tx in zip(train_data_list, train_label_list):
    vi_path = os.path.join(train_data_path, vi)
    tx_path = os.path.join(train_label_path, tx)
    
    cap = cv2.VideoCapture(vi_path)
