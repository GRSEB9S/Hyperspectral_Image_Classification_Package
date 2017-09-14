#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:39:47 2017

@author: root
"""

# only post-process
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils import load_data, split_train_test, data_summary,draw,draw_part,unaries_reshape,Post_Processing
import numpy as np
import time
import scipy.io
import os

## Load data
data_all, label_all, X, y, height, width, num_classes, GT_Label,ind,ind_each_class = \
            load_data('indian_pines',feature_type='raw',ispca=False)
            


## train-test-split
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, random_state=0)  

# my own split_train_test
train_size = 0.05
X_train, X_test, y_train, y_test, train_indexes, test_indexes = \
       split_train_test(X, y, train_size, ind_each_class, random_state=0)

train_map = np.zeros(len(data_all))
test_map  = np.zeros(len(data_all))
train_indexes = train_indexes.astype(int)
test_indexes = test_indexes.astype(int)
train_map[train_indexes] = label_all[train_indexes]
test_map[test_indexes] = label_all[test_indexes]
train_map = train_map.reshape(GT_Label.shape[1],GT_Label.shape[0]).transpose(1,0).astype(int)
test_map  = test_map.reshape(GT_Label.shape[1],GT_Label.shape[0]).transpose(1,0).astype(int)

DATA_PATH = os.getcwd()
train_ind = {}
train_ind['train_indexes'] = train_indexes
scipy.io.savemat(os.path.join(DATA_PATH, 'train_indexes.mat'),train_ind)

test_ind = {}
test_ind['test_indexes'] = test_indexes
scipy.io.savemat(os.path.join(DATA_PATH, 'test_indexes.mat'),test_ind)


## Data Summary
df = data_summary(y_train,y,num_classes)
print('----------------------------------')
print('Data Summary:')
print(df)
print('----------------------------------')
print("Training samples: %d" % len(y_train))
print("Test samples: %d" % len(y_test))
print('----------------------------------')

DATA_PATH = os.path.join(os.getcwd(),"datasets")
prob_map= scipy.io.loadmat(os.path.join(DATA_PATH, 'p.mat'))['p']
prob_map = np.transpose(prob_map)

# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(prob_map,height,width,\
                                          num_classes,y_test,test_indexes)
print(seg_accuracy)
