#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 01:17:03 2017

@author: root
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils import load_data, split_train_test, data_summary,draw,draw_part,unaries_reshape,Post_Processing
import numpy as np
import time
import scipy.io
import os


## Load data
data_all, label_all, X, y, height, width, num_classes, GT_Label,ind,ind_each_class = \
            load_data('indian_pines',feature_type='lowrank',ispca=False)
            


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

## Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test) 
data_all_scaled = scaler.transform(data_all)

## Classifiers
# KNN
from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
KNN = KNeighborsClassifier(n_neighbors=7).fit(X_train_scaled,y_train)
KNN_Label = KNN.predict(data_all_scaled).reshape(width,height).astype(int).transpose(1,0)
KNN_predict_prob = KNN.predict_proba(data_all_scaled)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(KNN_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(KNN) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (KNN.score(X_train_scaled,y_train),KNN.score(X_test_scaled,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,KNN_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Naive Bayes: GaussianNB
from sklearn.naive_bayes import GaussianNB
start_time = time.time()
GaussNB = GaussianNB().fit(X_train,y_train)
GaussNB_Label = GaussNB.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
GaussNB_predict_prob = GaussNB.predict_proba(data_all)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(GaussNB_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(GaussNB) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (GaussNB.score(X_train,y_train),GaussNB.score(X_test,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,GaussNB_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# discriminant_analysis - linear discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
start_time = time.time()
LDA = LinearDiscriminantAnalysis().fit(X_train,y_train)
LDA_Label = LDA.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
LDA_predict_prob = LDA.predict_proba(data_all)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(LDA_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(LDA) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (LDA.score(X_train,y_train),LDA.score(X_test,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,LDA_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Logistic Regression
from sklearn.linear_model import LogisticRegression
start_time = time.time()
LR = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=10).fit(X_train_scaled,y_train)
LR_Label = LR.predict(data_all_scaled).reshape(width,height).astype(int).transpose(1,0)
LR_predict_prob = LR.predict_proba(data_all_scaled)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(LR_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(LR) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (LR.score(X_train_scaled,y_train),LR.score(X_test_scaled,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,LR_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Linear SVM
from sklearn.svm import LinearSVC
start_time = time.time()
LSVM = LinearSVC(multi_class='ovr').fit(X_train_scaled,y_train)
LSVM_Label = LSVM.predict(data_all_scaled).reshape(width,height).astype(int).transpose(1,0)
print('(Linear SVM) Train Acc=%.3f, Cla_Acc==%.3f (Time_cost=%.3f)'\
      % (LSVM.score(X_train_scaled,y_train),LSVM.score(X_test_scaled,y_test),\
         (time.time()-start_time)))
# draw classification map
draw_part(GT_Label,LSVM_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Kernel SVM
from sklearn.svm import SVC
start_time = time.time()
SVM = SVC(C=200,probability=True).fit(X_train_scaled, y_train)
SVM_Label = SVM.predict(data_all_scaled).reshape(width,height).astype(int).transpose(1,0)
SVM_predict_prob = SVM.predict_proba(data_all_scaled)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(SVM_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(Kernel SVM) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (SVM.score(X_train_scaled,y_train),SVM.score(X_test_scaled,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,SVM_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
start_time = time.time()
DTree = DecisionTreeClassifier(max_depth=50).fit(X_train,y_train)
DTree_Label = DTree.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
DTree_predict_prob = DTree.predict_proba(data_all)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(DTree_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(Decision Tree) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (DTree.score(X_train,y_train),DTree.score(X_test,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,DTree_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Random Forest
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()
RF = RandomForestClassifier(n_estimators=200).fit(X_train,y_train)
RF_Label = RF.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
RF_predict_prob = RF.predict_proba(data_all)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(RF_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(Random Forest) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (RF.score(X_train,y_train),RF.score(X_test,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,RF_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier
start_time = time.time()
GBC = GradientBoostingClassifier(n_estimators=300,learning_rate=0.1).fit(X_train,y_train)
GBC_Label = GBC.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
GBC_predict_prob = GBC.predict_proba(data_all)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(GBC_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(Gradient Boosting) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (GBC.score(X_train,y_train),GBC.score(X_test,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,GBC_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

# Neural Network - MLP
from sklearn.neural_network import MLPClassifier
start_time = time.time()
MLP = MLPClassifier(hidden_layer_sizes=[200,350]).fit(X_train_scaled,y_train)
MLP_Label = MLP.predict(data_all_scaled).reshape(width,height).astype(int).transpose(1,0)
MLP_predict_prob = MLP.predict_proba(data_all_scaled)
# Post-processing using Graph-Cut
Seg_Label, seg_accuracy = Post_Processing(MLP_predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
print('(MLP) Train_Acc=%.3f, Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
      % (MLP.score(X_train_scaled,y_train),MLP.score(X_test_scaled,y_test),\
         seg_accuracy, (time.time()-start_time)))
# draw classification map
draw(GT_Label,MLP_Label,Seg_Label,train_map,test_map)
print('--------------------------------------------------------------------')

