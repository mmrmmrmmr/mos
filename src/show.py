from train_net import *

def show():
    l = len(data_label_all)
    # l = 1
    for i in range(l):
        # i = 5
        inputs = read_image_from_root([data_train[i]])
        labels = read_image_from_root([data_label[i]])
        c = model(inputs.to(device)).to('cpu').permute(0,2,3,1).squeeze(dim=0).detach()
        cv2.imshow('groundtruth',np.array(labels.squeeze(dim=0).detach()))
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
        cv2.waitKey(5)
        