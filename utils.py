# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:39:00 2017

@author: Xiangyong Cao
"""

import scipy.io
import numpy as np
import os
import math
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import spectral as spy
from pygco import cut_simple, cut_simple_vh
from sklearn.metrics import accuracy_score


def fstack(Data,n,random_state):
    n = int(n)
    h,w,b,num = Data.shape[0],Data.shape[1],Data.shape[2],Data.shape[3]
    np.random.seed(random_state)
    if n==1:
        ind = np.array([np.random.randint(low=0,high=52)])
        Data = Data[:,:,:,ind[0]]
    else:
        ind = np.random.permutation(52)
        Temp = np.zeros((h,w,1))
        for i in range(n):
            Temp = np.dstack((Temp,Data[:,:,:,ind[i]]))
        Data = np.delete(Temp,(0),axis=2)
    return Data,ind[0:n]
    
def load_data(string,feature_type='raw',ispca=False):
    feature_type = feature_type.split('_')
    DATA_PATH = os.path.join(os.getcwd(),"datasets")
    if feature_type[0]=='raw':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'.mat'))[string]
        ind = 0
    if feature_type[0]=='3ddwt':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_3ddwt.mat'))['Data_dwt3d']
        ind = 0
    if feature_type[0]=='3dgabor':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_3dgabor.mat'))[string+'_3dgabor']
        Data,ind = fstack(Data,feature_type[1],random_state=0)
    Label = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_gt.mat'))[string+'_gt']
    Data = Data.astype(float)
    # some constant paramaters
    height, width, band = Data.shape[0], Data.shape[1], Data.shape[2]
    num_classes = len(np.unique(Label))-1

#    # Normalizations
#    for b in range(band):
#        temp = Data[:,:,b]
#        Data[:,:,b] = (temp-np.min(temp))/(np.max(temp)-np.min(temp))


    ## Transform tensor data into matrix data, each row represents a spectral vector  
    # transform 3D into 2D 
    data_all = Data.transpose(2,0,1).transpose(0,2,1).reshape(band,-1).transpose(1,0)
    
    # transform 2D into 1D
    label_all = Label.transpose(1,0).flatten()
    
    # dimension reduction using PCA
    if ispca:
        pca = PCA(n_components=40)
        data_all = pca.fit_transform(data_all)
    # remove the sepctral vectors whose labels are 0
    data = data_all[label_all!=0]
    label = label_all[label_all!=0]
    
    label_list = list(label_all)
    ind_each_class=[]
    for i in range(1,num_classes+1):
        ind_each_class.append([index for index, value in enumerate(label_list) if value == i]) 
    ind_each_class = np.asarray(ind_each_class)
    return data_all, label_all, data, label, height, width, num_classes, Label, ind, ind_each_class
    

def split_each_class(samples_class_k, labels_class_k, ind_each_class_k,
                     num_train_class_k,one_hot=False,random_state=0):
    idx = np.arange(0, len(samples_class_k))  # get all possible indexes
    np.random.seed(random_state)
    np.random.shuffle(idx)  # shuffle indexes
    num_train_class_k = int(num_train_class_k)
    idx_train = idx[0:num_train_class_k]
    idx_test = idx[num_train_class_k:]  
    X_train_class_k = np.asarray([samples_class_k[i] for i in idx_train])  
    X_test_class_k = np.asarray([samples_class_k[i] for i in idx_test])  
    if one_hot:
        y_train_class_k = np.asarray([labels_class_k[i] for i in idx_train])
        y_test_class_k = np.asarray([labels_class_k[i] for i in idx_test])
    else:
        y_train_class_k = np.asarray([labels_class_k[i] for i in idx_train]).reshape(len(idx_train),1)
        y_test_class_k = np.asarray([labels_class_k[i] for i in idx_test]).reshape(len(idx_test),1)
        tr_index_k = np.asarray([ind_each_class_k[i] for i in idx_train]).reshape(len(idx_train),1)
        te_index_k = np.asarray([ind_each_class_k[i] for i in idx_test]).reshape(len(idx_test),1)
    return X_train_class_k, y_train_class_k,X_test_class_k, y_test_class_k, tr_index_k, te_index_k
    
def list2array(X,isdata=True,one_hot=False):
    if isdata:
        Y = np.zeros(shape=(1,X[0].shape[1]))
        for k in range(len(X)):
            Y = np.vstack((Y,X[k]))
        Y = np.delete(Y,(0),axis=0)
    else:
        if one_hot:
            Y = np.zeros(shape=(1,X[0].shape[1]))
            for k in range(len(X)):
                Y = np.vstack((Y,X[k]))
            Y = np.delete(Y,(0),axis=0)                
        else:
            Y = np.zeros(shape=(1,))
            for k in range(len(X)):
                Y = np.vstack((Y,X[k]))
            Y = np.delete(Y,(0),axis=0)
    return Y 
    
def split_train_test(X, y, train_size, ind_each_class, one_hot=False, random_state=0):
    #sample_each_class, label_each_class, num_train_each, train_rate, proportion):
    num_classes = len(np.unique(y))
    sample_each_class = np.asarray([X[y==k] for k in range(1,num_classes+1)]) 
    if one_hot:
        y_0 = y - 1
        y_onehot = convertToOneHot(y_0)
        label_each_class = np.asarray([y_onehot[y_0==k] for k in range(num_classes)]) 
    label_each_class  = np.asarray([y[y==k] for k in range(1,num_classes+1)]) 
    num_each_class = [len(sample_each_class[k]) for k in range(num_classes)]
    if train_size>=0 and train_size<=1:
        num_train = [math.ceil(train_size * i) for i in num_each_class]
    else:
        num_train = [train_size/num_classes] * num_classes
    X_train, y_train, X_test, y_test, train_indexes, test_indexes = [],[],[],[],[],[]
    for k in range(num_classes):
        X_train_class_k, y_train_class_k, X_test_class_k, y_test_class_k, tr_index_k, te_index_k =\
               split_each_class(sample_each_class[k], label_each_class[k],
                                ind_each_class[k],num_train[k], one_hot, random_state)
        X_train.append(X_train_class_k)
        y_train.append(y_train_class_k)
        X_test.append(X_test_class_k)
        y_test.append(y_test_class_k) 
        train_indexes.append(tr_index_k)
        test_indexes.append(te_index_k)
    X_train = list2array(X_train)
    X_test  = list2array(X_test)
    y_train = list2array(y_train,isdata=False,one_hot=False)
    y_test  = list2array(y_test,isdata=False,one_hot=False)
    train_indexes  = list2array(train_indexes,isdata=False,one_hot=False)
    test_indexes   = list2array(test_indexes,isdata=False,one_hot=False)
    if one_hot==False:
        y_train=y_train.reshape((y_train.shape[0],))
        y_test=y_test.reshape((y_test.shape[0],))
        train_indexes = train_indexes.reshape((train_indexes.shape[0],))
        test_indexes = test_indexes.reshape((test_indexes.shape[0],))
    return X_train,X_test, y_train, y_test, train_indexes, test_indexes
 
def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)    
    

def data_summary(y_train,y,num_classes):
    df = pd.DataFrame(np.random.randn(num_classes, 3),
                      index=['class_'+np.str(i) for i in range(1,1+num_classes)],
                  columns=['Train', 'Test', 'Total'])
    df['Train'] = [sum(y_train==i) for i in range(1,num_classes+1)]
    df['Total'] = [sum(y==i) for i in range(1,num_classes+1)]
    df['Test'] = np.array(df['Total']) - np.array(df['Train'])
    return df
    
def draw(GT_Label,ES_Label,Seg_Label,train_map,test_map):

    fig = plt.figure(figsize=(12,6))

    p = plt.subplot(1, 5, 1)
    v = spy.imshow(classes=GT_Label, fignum=fig.number)
    p.set_title('Ground Truth')
    p.set_xticklabels([])
    p.set_yticklabels([])

    p = plt.subplot(1, 5, 2)
    spy.imshow(classes = train_map , fignum=fig.number)
    p.set_title('Training Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 5, 3)
    v = spy.imshow(classes = test_map, fignum=fig.number)
    p.set_title('Testing Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 5, 4)
    v = spy.imshow(classes = ES_Label * (GT_Label != 0), fignum=fig.number)
    p.set_title('Classification Map')
    p.set_xticklabels([])
    p.set_yticklabels([])

    p = plt.subplot(1, 5, 5)
    v = spy.imshow(classes = Seg_Label * (GT_Label != 0), fignum=fig.number)
    p.set_title('Segmentation Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
def draw_part(GT_Label,ES_Label,train_map,test_map):

    fig = plt.figure(figsize=(12,6))

    p = plt.subplot(1, 4, 1)
    v = spy.imshow(classes=GT_Label, fignum=fig.number)
    p.set_title('Ground Truth')
    p.set_xticklabels([])
    p.set_yticklabels([])

    p = plt.subplot(1, 4, 2)
    spy.imshow(classes = train_map , fignum=fig.number)
    p.set_title('Training Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 4, 3)
    v = spy.imshow(classes = test_map, fignum=fig.number)
    p.set_title('Testing Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 4, 4)
    v = spy.imshow(classes = ES_Label * (GT_Label != 0), fignum=fig.number)
    p.set_title('Classification Map')
    p.set_xticklabels([])
    p.set_yticklabels([])

def unaries_reshape(unaries,height,width,num_classes):
    una = []
    for i in range(num_classes):
        temp = unaries[:,i].reshape(height,width).transpose(1,0)
        una.append(temp)
    return np.dstack(una).copy("C")

def Post_Processing(prob_map,height,width,num_classes,y_test,test_indexes):
    unaries = (-100*np.log(prob_map+1e-4)).astype(np.int32)
    una = unaries_reshape(unaries,width,height,num_classes)
    one_d_topology = (np.ones(num_classes)-np.eye(num_classes)).astype(np.int32).copy("C")
    Seg_Label = cut_simple(una, 50 * one_d_topology)
    Seg_Label = Seg_Label + 1
    seg_Label = Seg_Label.transpose().flatten()
    seg_accuracy = accuracy_score(y_test,seg_Label[test_indexes])
    return Seg_Label, seg_accuracy
    
    
    
    