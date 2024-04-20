import os
import torch
import numpy as np
import torch.utils.data as data
# from .io import IO
from .build import DATASETS
from utils.logger import *
import h5py
import json
import random

@DATASETS.register_module()
class PartNormal(data.Dataset):
    def __init__(self, config):

        self.npoints = config.N_POINTS
        self.root = config.PC_PATH + "/shapenetcore_partanno_segmentation_benchmark_v0_normal"
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        if( config.subset == "train" ):
            split = "trainval"
        else:
            split = config.subset
        self.split = split

        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

        print_log('The number of data: %d' % len( self.datapath ), logger = 'PartNormal')

        # self.calc_class_weight()

    # def calc_class_weight(self):
    #     y = []
    #     for i in range( len( self.datapath ) ):
    #     # for i in range( 10 ):
    #         print(i)
    #         fn = self.datapath[i]
    #         data = np.loadtxt(fn[1]).astype(np.float32)
    #         # point_set = data[:, 0:6]
    #         seg = data[:, -1].astype(np.int32)
    #         y.append( seg )
    #     y = np.concatenate( y )
    #     from sklearn.utils.class_weight import compute_class_weight
    #     labels = np.unique(y)
    #     class_weights = compute_class_weight( 'balanced', labels, y )
    #     for i in range( labels.size ):
    #         print( class_weights[ i ] )
    #     quit()

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        pos = pc[:, 0:3 ]
        ori = pc[:, 3:6 ]
        centroid = np.mean( pos, axis=0 )
        pos = pos - centroid
        m = np.max(np.sqrt(np.sum(pos**2, axis=1)))
        pos = pos / m
        pc = np.hstack( [ pos, ori ] )
        return pc

    def __getitem__(self, index):

        # ##############################
        # # balanced minibatch sampling
        # if( self.split == "trainval" or self.split == "train" ):
        #     cls_id = index % len( self.seg_classes )
        #     candidates = []
        #     for i, fn in enumerate( self.datapath ):
        #         if( self.classes[ fn[0] ] == cls_id ):
        #             candidates.append( i )
        #     index = random.choice( candidates )
        # ##############################

        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        point_set = self.pc_norm( point_set )

        # resample
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)
