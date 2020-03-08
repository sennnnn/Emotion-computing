import os

task = 'trainval'

task_list = os.listdir(task)

f = open('{}.txt'.format(task), 'w')
for one in task_list:
    one_video_path = os.path.join(task, one)
    item_list = os.listdir(one_video_path)
    item_list = sorted(item_list, key=lambda x: int(os.path.splitext(x)[0]))
    train_list = []
    for item in item_list:
        item = os.path.splitext(item)[0]
        if item not in train_list: train_list.append(item)
        else: continue
        item_path = os.path.join(one_video_path, item)
        f.write('{}/{}\n'.format('dataset', item_path))

        