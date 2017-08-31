# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:27:37 2017

@author: Xiangyong Cao
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils import load_data, split_train_test, data_summary
import numpy as np
import time

## Load data
X, y, num_classes, GT_Label, ind = load_data('indian_pines',feature_type='3dgabor_1',ispca=False)

## train-test-split
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, random_state=0)  

# my own
train_size = 0.05
X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=train_size, 
                                                    random_state=0)
## Data Summary
df = data_summary(y_train,y,num_classes)
print('----------------------------------')
print('Data Summary:')
print(df)
print('----------------------------------')
print("Traini samples: %d" % len(y_train))
print("Test samples: %d" % len(y_test))
print('----------------------------------')

## Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test) 


# Classifiers
# KNN
from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
KNN = KNeighborsClassifier(n_neighbors=1).fit(X_train,y_train)
print('(KNN) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (KNN.score(X_train,y_train),KNN.score(X_test,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Naive Bayes: GaussianNB
from sklearn.naive_bayes import GaussianNB
start_time = time.time()
GaussNB = GaussianNB().fit(X_train,y_train)
print('(GaussNB) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (GaussNB.score(X_train,y_train),GaussNB.score(X_test,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# discriminant_analysis - linear discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
start_time = time.time()
LDA = LinearDiscriminantAnalysis().fit(X_train,y_train)
print('(LDA) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (LDA.score(X_train,y_train),LDA.score(X_test,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Logistic Regression
from sklearn.linear_model import LogisticRegression
start_time = time.time()
LR = LogisticRegression(multi_class='ovr',solver='lbfgs',C=100).fit(X_train,y_train)
print('(LR) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (LR.score(X_train,y_train),LR.score(X_test,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Linear SVM
from sklearn.svm import LinearSVC
start_time = time.time()
LSVM = LinearSVC(multi_class='ovr').fit(X_train_scaled,y_train)
print('(Linear SVM) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (LSVM.score(X_train_scaled,y_train),LSVM.score(X_test_scaled,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Kernel SVM
from sklearn.svm import SVC
start_time = time.time()
SVM = SVC(C=1000).fit(X_train_scaled, y_train)
print('(Kernel SVM) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (SVM.score(X_train_scaled,y_train),SVM.score(X_test_scaled,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
start_time = time.time()
DTree = DecisionTreeClassifier().fit(X_train,y_train)
print('(Decision Tree) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (DTree.score(X_train,y_train),DTree.score(X_test,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Random Forest
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()
RF = RandomForestClassifier(n_estimators=200).fit(X_train,y_train)
print('(Random Forest) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (RF.score(X_train,y_train),RF.score(X_test,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier
start_time = time.time()
GBC = GradientBoostingClassifier(n_estimators=300,learning_rate=0.1).fit(X_train,y_train)
print('(Gradient Boosting) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (GBC.score(X_train,y_train),GBC.score(X_test,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

# Neural Network - MLP
from sklearn.neural_network import MLPClassifier
start_time = time.time()
MLP = MLPClassifier(hidden_layer_sizes=[500,500]).fit(X_train_scaled,y_train)
print('(MLP) Train Accuracy=%.3f, Test Accuracy=%.3f (Time_cost=%.3f)'\
      % (MLP.score(X_train_scaled,y_train),MLP.score(X_test_scaled,y_test),\
         (time.time()-start_time)))
print('--------------------------------------------------------------------')

