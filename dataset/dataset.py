import os
import sys
sys.path.append('..')
import cv2

from util import list_to_dict,write_va

root_path = r'E:\dataset\emotion_computating\MediaEval\2018'
data_path = os.path.join(root_path, 'data')
label_path = os.path.join(root_path, 'label')

task = 'test'

task_data_path = os.path.join(data_path, task)
task_label_path = os.path.join(label_path, task)
task_data_list = os.listdir(task_data_path)
task_label_list = os.listdir(task_label_path)

build_path = ''
build_task_path = os.path.join(build_path, task)

for vi,tx in zip(task_data_list, task_label_list):
    instance = os.path.splitext(vi)[0]
    # 失误操作导致其标签被废了。
    instance_path = os.path.join(build_task_path, instance)
    
    data_path = os.path.join(task_data_path, vi)
    label_path = os.path.join(task_label_path, tx)

    if(not os.path.exists(instance_path)):
        os.makedirs(instance_path)
    else: continue
    va_info = open(label_path, 'r').readlines()
    va_info = [x.strip().split('\t') for x in va_info]
    va_info = list_to_dict(va_info)

    Time_list = va_info['Time']
    v_list = va_info['Valence']
    a_list = va_info['Arousal']
    time_length = len(Time_list)
    cap = cv2.VideoCapture(data_path)
    if(cap.isOpened()):
        index = 0
        while(1):
            flag,frame = cap.read()
            if(not flag): break
            if(cap.get(0)/1000 >= index):
                txt_path = '{}/{}.txt'.format(instance_path, Time_list[index])
                pic_path = os.path.join(instance_path, Time_list[index] + '.jpg')
                if(os.path.exists(txt_path) and os.path.exists(pic_path)): 
                    index += 1
                    continue
                print('{} {}/{}'.format(os.path.splitext(vi)[0], index, time_length))
                v = v_list[index]
                a = a_list[index]
                cv2.imwrite(pic_path, frame)
                write_va(txt_path, v, a)
                index += 1
                if(index == time_length): break
    else:
        assert False, "Error, {} can't be opened".format(vi)
