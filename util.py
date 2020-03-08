import os
import random

def list_to_dict(lines):
    ret_dict = {x:[] for x in lines[0]}
    for line in lines[1:]:
        for i,j in zip(list(ret_dict.keys()), line):
            ret_dict[i].append(j)
    
    return ret_dict

def write_va(txt_path, v, a):
    f = open(txt_path, 'w')
    f.write('{} {}\n'.format(v, a))
    f.close()

def saveInfo(info, txt_path):
    f = open(txt_path, 'w')
    f.write(str(info))
    f.close()

def loadInfo(txt_path):
    f = open(txt_path, 'r')
    return eval(f.read())

def dict_save(dict, txt_path):
    with open(txt_path, 'w') as f:
        f.write(str(dict))
        f.close()

def dict_load(txt_path):
    with open(txt_path, 'r') as f:
        return eval(f.read())

def get_newest(dir_path):
    file_list = os.listdir(dir_path)
    newest_file = os.path.join(dir_path,file_list[0])
    for filename in file_list:
        one_file = os.path.join(dir_path, filename)
        if(get_ctime(newest_file) < get_ctime(one_file)):
            newest_file = one_file

    return newest_file 

def get_ctime(file_path, ifstamp=True):
    if(ifstamp):
        return os.path.getctime(file_path)
    else:
        timeStruct = time.localtime(os.path.getctime(file_path))
        return time.strftime("%Y-%m-%d %H:%M:%S",timeStruct)

def ifsmaller(price_list, price):
    if(len(price_list) == 0):
        return 0
    else:
        if(price <= price_list[-1]):
            return 1
        else:
            return 0

def iflarger(price_list, price):
    if(len(price_list) == 0):
        return 0
    else:
        if(price >= price_list[-1]):
            return 1
        else:
            return 0

def open_readlines(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

        return lines

def train_valid_split(*args, valid_rate=0.2, ifrandom=True):
    ret_train = []
    ret_valid = []
    for arg in args:
        # 计算训练集和验证集的条目数量
        all_count = len(arg)
        valid_count = int(all_count * valid_rate)

        # 为训练集和验证集分配数据条目
        if(ifrandom):
            random.shuffle(arg)
        
        valid_list = arg[:valid_count]
        train_list = arg[valid_count:]
        ret_train.append(train_list)
        ret_valid.append(valid_list)
    
    return tuple(ret_train + ret_valid)