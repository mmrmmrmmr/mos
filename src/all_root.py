import os
import re

root = os.path.join(os.path.dirname(__file__))
root = os.path.join(os.path.dirname(root))

root_fig = root+'/fig/'
root_model = root+'/model/'
root_list = root+'/list/'

all_name_list = ['highway',
                'fountain02',
                'fall',
                'pedestrians',
                'office',
                'boats',
                'overpass',
                'canoe',
                'fountain01',
                'PETS2006',
                'busStation']

def create_txtroot(name,flag=1):
    if flag == 1:
        f_root = root_fig + name
        f = open(f_root+'/temporalROI.txt','r')
        begin = f.readline().replace(' ','\n')
        f_name = []
        f_mask = []
        for r, dirs, f in os.walk(f_root + '/input/', topdown=False):
            f_name.append(f)
        f_name[0].sort()
        f_name = [f_root + '/input/'+str(i) for i in f_name[0]]
        for r, dirs, f in os.walk(f_root + '/groundtruth/', topdown=False):
            f_mask.append(f)
        f_mask[0].sort()
        f_mask = [f_root + '/groundtruth/'+str(i) for i in f_mask[0]]
        
        f_input = open(root_list+name+"_input.txt",'w+')
        f_input.writelines(begin+'\n')
        for line in f_name:
            f_input.write(line+'\n')
        f_input.close()
        f_ground = open(root_list+name+"_groundtruth.txt",'w+')
        f_ground.writelines(begin+'\n')
        for line in f_mask:
            f_ground.write(line+'\n')
        f_ground.close()
        return 
    # else:

if __name__ == "__main__":
    for i in all_name_list:
        create_txtroot(i)
                
        