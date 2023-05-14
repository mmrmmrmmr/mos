import all_root
import cv2
import numpy as np
import os
import sys

w = 360
h = 240

name_list = ['highway']

data_org = []

# data_train = []
# data_label = []

# data_train_all = []
# data_label_all = []
# data = []
def read_data(name,n=3,per=0.7):
    data_train = []
    data_label = []
    data_train_all = []
    data_label_all = []
        
    place = ['_input.txt', '_groundtruth.txt']
    for i in range(2):
        data = []
        root = all_root.root_list+name+place[i]
        f = open(root,'r')
        data = f.read().splitlines()
        begin = int(data[0])
        data = data[begin+30:]
        if i == 0:
            d = []
            for i in range(len(data)-n+1):
                d.append(data[i:i+n])
            for j in d:
                j.append(all_root.root_fig+name+'/b.jpg')
            data_train_all.extend(d)
            # data_org.extend(data)
        else:
            data = data[n-1:]
            data_label_all.extend(data)
            x = int(len(data)*per)
            data_label.extend(data[0:x])
            data_train.extend(d[0:x])
    return data_label, data_label_all, data_train, data_train_all
            
# a = read_data('highway')

def cal_back(name,n=100):
    root_save = all_root.root_fig+name+'/b.jpg'
    root = all_root.root_list+name+'_input.txt'
    f = open(root,'r')
    data = f.read().splitlines()
    begin = int(data[0])
    data = data[begin:begin+n]
    add = np.zeros([h,w,3],dtype=float)
    for i in range(n):
        f = cv2.imread(data[i])
        f = cv2.resize(f, (w,h)).astype(float)
        add += f/n
    cv2.imwrite(root_save, add.astype(np.uint8))
    
# a = cal_back('highway')

# import cv2

# cv2.imshow('1', cv2.imread(data_label[1],flags=2))
# cv2.waitKey(1)
        
# for i in all_root.all_name_list:
#     cal_back(i, 50)

def progress_bar(finish_tasks_number, tasks_number, epoch, a, b):
    os.system('clear')
    percentage = round(finish_tasks_number / tasks_number * 100)
    # print("\r now : {:.4f}  before : {:.4f} \n".format(a,b), end="")
    # print("\r 进度 {}: ".format(epoch), "▓" * (percentage // 2), end="",flush=True)
    print("\r now:{:.4f} before:{:.4f} epoch:{:4d}".format(a,b,epoch), "▓" * (percentage // 2), end="",flush=True)
    
    
    # sys.stdout.flush()

if __name__ == "__main__":
    for i in all_root.all_name_list:
        all_root.create_txtroot(i)
        cal_back(i,50)
    # a = read_data('highway')

