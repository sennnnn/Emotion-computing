import os
import sys
sys.path.append('..')
import cv2

from util import list_to_dict

root_path = r'E:\dataset\emotion_computating\MediaEval\2018'
data_path = os.path.join(root_path, 'data')
label_path = os.path.join(root_path, 'label')
train_data_path = os.path.join(data_path, 'trainval')
train_label_path = os.path.join(label_path, 'trainval')

train_data_list = os.listdir(train_data_path)
train_label_list = os.listdir(train_label_path)

build_path = ''
build_train_path = os.path.join(build_path, 'trainval')
build_label_path = os.path.join(build_path, 'test')

for vi,tx in zip(train_data_list, train_label_list):
    instance = os.path.splitext(vi)[0]
    # 失误操作导致其标签被废了。
    if(instance == 'MEDIAEVAL18_00'): continue
    instance_path = os.path.join(build_train_path, instance)
    
    vi_path = os.path.join(train_data_path, vi)
    tx_path = os.path.join(train_label_path, tx)

    if(not os.path.exists(instance_path)):
        os.makedirs(instance_path)
    
    va_info = open(tx_path, 'r').readlines()
    va_info = [x.strip().split('\t') for x in va_info]
    va_info = list_to_dict(va_info)

    cap = cv2.VideoCapture(vi_path)
    if(cap.isOpened()):
        index = 0
        while(1):
            flag,frame = cap.read()
            if(not flag): break
            if(cap.get(0)/1000 > index):
                txt_path = '{}/{}.txt'.format(instance_path, va_info['Time'][index])
                pic_path = os.path.join(instance_path, va_info['Time'][index] + '.jpg')
                if(os.path.exists(txt_path) and os.path.exists(pic_path)): 
                    index += 1
                    continue
                print('{} {}/{}'.format(os.path.splitext(vi)[0], index, len(va_info['Time'])-1))
                v = va_info['Valence'][index]
                a = va_info['Arousal'][index]
                index += 1
                f = open(txt_path, 'w')
                f.write('{} {}\n'.format(v, a))
                f.close()
                cv2.imwrite(pic_path, frame)
    else:
        assert False, "Error, {} can't be opened".format(vi)