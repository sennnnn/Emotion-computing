import cv2
import random
import numpy as np

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

def open_readlines(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

        return lines

def parse_va(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        va = lines[0].split()
        va = [float(x) for x in va]

        return va

class train_generator():
    def __init__(self, train_txt_path, input_shape, batch_size, ifrandom=True):
        self.train_list = open_readlines(train_txt_path)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.ifrandom = ifrandom
    
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

        if(count == 0): yield data_batch_block,label_batch_block

    def epochwise_generator(self):
        while(1):
            yield from self.__iter__()

if __name__ == "__main__":
    # example about how to use train generator.
    # train_ge_obj = train_generator('dataset/trainval.txt', (224, 224), 4)
    # epochwise_gene = train_ge_obj.epochwise_generator()
    # for i,j in epochwise_gene:
    #     print(i, j)
    pass