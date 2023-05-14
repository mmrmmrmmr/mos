import numpy as np
import cv2
from all_net import all_net, BinaryDiceLoss
from create_data import *
import all_root
import torch
import torch.optim as optim
import torch.nn as nn
# from show import show

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 2
root_save = all_root.root_model

# data_list = all_root.all_name_list
data_list = ['highway',
             'pedestrians',
             'office',
             'busStation',
             'canoe',
             'overpass']

data_train = []
data_label = []
data_train_all = []
data_label_all = []

def init_data():
    global data_label, data_label_all, data_train, data_train_all
    data_train = []
    data_label = []
    data_train_all = []
    data_label_all = []
    for i in data_list:
        a,b,c,d = read_data(i)
        data_label.extend(a)
        data_label_all.extend(b)
        data_train.extend(c)
        data_train_all.extend(d)
    
    data_label_all = np.array(data_label_all)
    data_train_all = np.array(data_train_all)
    data_label = np.array(data_label)
    data_train = np.array(data_train)
        
init_data()

def image2tensor(root):
    f = cv2.imread(root)
    f = cv2.resize(f,(w,h))
    f = torch.from_numpy(f.astype(np.float32))/255
    return f.permute(2,0,1)

def read_image_from_root(root):
    batch = len(root)
    flag = type(root[0])
    data = None
    if flag == np.ndarray:
        data = torch.zeros(batch,12,h,w)
        for i in range(batch):
            for j in range(0,12,3):
                data[i,j:j+3] = image2tensor(root[i][int(j/3)])
    else:
        data = torch.zeros(batch,h,w)
        for i in range(batch):
            data[i] = image2tensor(root[i])[0]
    return data

model = all_net().to(device)
# criterion = nn.CrossEntropyLoss().to(device)
criterion = BinaryDiceLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(num_e=100):
    tot = 0.0
    epoch_loss = 0.0
    before_loss = 0.0
    l = len(data_train)
    l = int(l/batch_size)
    for epoch in range(num_e):
        running_loss = 0.0    
        for i in range(l):
            progress_bar(i, l, epoch+1, epoch_loss, before_loss)
            if device != 'cpu':
                torch.cuda.empty_cache()

            x = np.random.randint(0,l,batch_size)
            inputs = read_image_from_root(data_train[x]).to(device)
            labels = read_image_from_root(data_label[x]).to(device)
            # zeros the paramster gradients
            # return labels
            optimizer.zero_grad()       # 
            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            # return outputs, labels
            loss = criterion(outputs.squeeze(dim=1), labels)  # 计算loss
            loss.backward()     # loss 求导
            optimizer.step()    # 更新参数
            # print statistics
            running_loss += loss.item()  # tensor.item()  获取tensor的数值
        
        before_loss = epoch_loss
        epoch_loss = running_loss
        tot += epoch
            
        if epoch%5 == 4:
            print(epoch+1, tot/5)
            before_loss = epoch_loss
            tot = 0.0
            torch.save(model.state_dict(), root_save+'model' + str(epoch+1)+'.pth')
    torch.save(model.state_dict(), root_save+'model_end' + '.pth')
    print('Finished Training')
    if device != 'cpu':
        torch.cuda.empty_cache()
        
def show():
    l = len(data_label_all)
    # l = 1
    for i in range(l):
        # i = 5
        inputs = read_image_from_root([data_train[i]])
        labels = read_image_from_root([data_label[i]])
        c = model(inputs.to(device)).to('cpu').permute(0,2,3,1).squeeze(dim=0).detach()
        cv2.imshow('groundtruth',np.array(labels.squeeze(dim=0).detach()))
        # return c
        c = c/c.max()
        c = np.array(c)
        c = c*255
        c = c.astype(np.uint8)
        cv2.imshow('out', c)
        inputs = inputs[:,6:9,:,:].permute(0,2,3,1).squeeze(dim=0).detach()
        c = np.array(inputs)
        c = c*255
        c = c.astype(np.uint8)
        cv2.imshow('in', c)
        cv2.waitKey(1)      
        

train(25)

# model.load_state_dict(torch.load(root_save+'modelend2.pth')) 