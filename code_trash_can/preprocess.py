import os
import cv2
import time
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm


basepath = 'E:\\dataset\\datasets\\MediaEval\\2018'
data_dopath   = 'data/Videos/trainval'
label_dopath  = 'label/txts/trainval'


list_nosuffix_label= [x+'.txt' for x in sorted([os.path.splitext(x)[0] for x in os.listdir(os.path.join(basepath,label_dopath))],key=lambda x: int(x.split('_')[1]))]
list_nosuffix_data = [x+'.mp4' for x in sorted([os.path.splitext(x)[0] for x in os.listdir(os.path.join(basepath,data_dopath))],key=lambda x: int(x.split('_')[1]))]

if not os.path.exists('./build'):
    os.mkdir('./build')

if not os.path.exists('./build/data'):
    os.mkdir('./build/data')

if not os.path.exists('./build/label'):
    os.mkdir('./build/label')

for filename_data,filename_label in zip(list_nosuffix_data,list_nosuffix_label):

    print(filename_label)
    filename_data_ = os.path.join(basepath,data_dopath,filename_data)
    filename_label_ = os.path.join(basepath,label_dopath,filename_label)

    cap = cv2.VideoCapture(filename_data_)
    annotations = open(filename_label_,"r")
    lines = annotations.readlines()
    for line,i in zip(lines,tqdm(range(len(lines)))):
        line = line.strip('\n')
        line = line.split('\t')
        if(line[0] == 'Time'):
            continue
        if not os.path.exists('./build/label/{}'.format(filename_label)):
            os.mkdir('./build/label/{}'.format(filename_label))
        np.save(os.path.join('./build/label/{}/{}.npy'.format(filename_label,int(float(line[0])))),np.array(line[1:3],dtype=np.float32))
    annotations.close()

    
    if(cap.isOpened()):
        print(filename_data)
        all_seconds = []
        one_second = []
        index = 1
        while(1):

            flag,one_frame = cap.read()
            if(not flag):
                print('\n')
                break
            one_second.append(one_frame)
            if(cap.get(0)/1000 > index):
                # if(index == 1100):
                #     break
                one_second.clear()
                if not os.path.exists('./build/data/{}'.format(filename_data)):
                    os.mkdir('./build/data/{}'.format(filename_data))
                temp=np.array(one_second,dtype=np.uint8)
                np.save(os.path.join('./build/data/{}/{}.npy'.format(filename_data,index-1)),temp)
                print('\r','当前处理的视频时间戳:{}:{}:{}'.format(int((index-1)/3600),int(((index-1)%3600)/60),(index-1)%60),end="",flush=True)
                index += 1
                
        if(len(one_second) != 0):
            np.save(os.path.join('./build/data/{}/{}.npy'.format(filename_data,index-1)),np.array(one_second,dtype=np.uint8))