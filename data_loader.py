import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
import ipdb

class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
        data = np.load(data_path) #data['kdd'] is an array with [feature_1, feature_2,..., label]
        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        N, D = features.shape 
        normal_data = features[labels==1] # normal is lablled as 1 
        normal_labels = labels[labels==1]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==0] # attack is lablled as 0
        attack_labels = labels[labels==0]

        # due to this kdd dataset has a larger number of attacks than normal, so we treat the attacks as training data
        
        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack) # shuffle the data
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        self.train = attack_data[randIdx[:N_train]]            # train data
        self.train_labels = attack_labels[randIdx[:N_train]]

        self.test = attack_data[randIdx[N_train:]]             # test data
        self.test_labels = attack_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, normal_data),axis=0) # concatenate the test and normal as consequent testing data
        self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])


class KittingExperimentLoader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
        data = np.load(data_path) # data  is an array with [feature_1, feature_2,..., label]
        labels = data[:,-1]
        features = data[:,:-1]
        N, D = features.shape 
        normal_data = features[labels==0] # normal is lablled as 0 
        normal_labels = labels[labels==0]


        anomaly_data = features[labels==1] # anomaly is lablled as 1
        anomaly_labels = labels[labels==1]

        N_normal = normal_data.shape[0]
        
        randIdx = np.arange(N_normal) # shuffle the data
        np.random.shuffle(randIdx)
        N_train = N_normal // 2

        self.train = normal_data[randIdx[:N_train]]            # train data
        self.train_labels = normal_labels[randIdx[:N_train]]

        self.test = normal_data[randIdx[N_train:]]             # test data
        self.test_labels = normal_labels[randIdx[N_train:]]
        self.test = np.concatenate((self.test, anomaly_data),axis=0) # concatenate the normal and anomaly as consequent testing data
        self.test_labels = np.concatenate((self.test_labels, anomaly_labels),axis=0)

        print 
        print("shape of train:{}, shape of test:{}".format(self.train.shape, self.test.shape))
        print
        print("shape of normal_data:{}, shape of anomaly_data:{}".format(normal_data.shape, anomaly_data.shape))
        print
        # input("Press ENTER to continue!")
        

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])
       
       

def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""

    if 'kitting' in data_path.lower():
        dataset = KittingExperimentLoader(data_path, mode)        
    elif 'kdd' in data_path.lower():
        dataset = KDD99Loader(data_path, mode)
    else:
        raise("can't find the dataset")
        
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
