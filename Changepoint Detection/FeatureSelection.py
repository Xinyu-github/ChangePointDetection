# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:28:09 2020

@author: xinyu
"""

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import ChangepointDetection as CPD

def LabelGenerator(windows,max_gradient_order):
    label = ['F','s',]
    
    for i in windows:
        for j in range(max_gradient_order):
            if j == 0:
                grad = 'd'
            else:
                grad += 'd'
            label.append(grad+'F_'+'w'+str(i))
            label.append(grad+'s_'+'w'+str(i))
    return label



dataArr = np.loadtxt(".\Database\OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
resArr = np.loadtxt(".\Database\OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
infoArr = np.loadtxt(".\Database\OUTPUT_meritve_ll.txt", skiprows=1) #additional information
# vibrArr = np.loadtxt("U:\Data\\vibration_filter.txt")

IDs = int(dataArr[-1,0]) #Number of valve tests in database (up to 3 measurements per test)
DB = [dataArr,resArr,infoArr] 
LoadRawData = CPD.DataPreparing(IDs,DB)
RawF = []        
X = []
Y = []
groupID = []
windows = [1,2,3] 
max_gradient_order = 1
FeatureLabel = LabelGenerator(windows,max_gradient_order)
DataPreprocess = CPD.DataPreprocessing(IDs,DB,
                                       window=windows,
                                       max_gradient_order = max_gradient_order)   
id_from, id_to = CPD.loop_from_to(IDs, id_from=0, steps=0) #id_from=0 loops through all
for i in range(id_from,id_to):
    for j in range(1,4): 
        RawData = LoadRawData.segment(i,j,include_s = True)
        try:
            RawF.append(RawData.F)
        except:
            continue
        F,s,y = DataPreprocess.OutputPrepare(i,j,include_s = True)
        F = DataPreprocess.Smooth(F)
        F,s,y = DataPreprocess.Trim(F,s,y)
        x = DataPreprocess.FeatureGenerator(F,s)
        if x.any():
            groupID.append([i,j])
            X.append(x)
            Y.append(y)
        
X, Y= DataPreprocess.Padding(X,Y) 
        
X, Y= DataPreprocess.Padding(X,Y)  
idxos = np.array([np.argmax(y) for y in Y]) #array with indexes of opening point from database

########################################################################

#group measurements by testing ID (= particular valve)
groups = np.array(groupID)[:,0] 
n_folds = 5
# get current time and date to use as file name
now = datetime.now()
current_time = now.strftime("%d_%m-%H_%M")

x_train,y_train,x_val,y_val,x_test,y_test,split_idx = DataPreprocess.DataSplit(X,Y,
                                     groups = groups,
                                     n_folds=n_folds,
                                     test_size =0.3,
                                     val_size = 0.2,
                                     load_split=0, # load or create split info
                                     time=current_time) # should be save_time when load_split =1

BuildNet = CPD.BuildClassifier(NetType = 'Dense',
                           input_shape = x_train[0].shape[1:])

es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience = 10,
                       restore_best_weights=True)
    
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
FQI = np.zeros(X.shape[-1])
for i in range(n_folds):
    model = BuildNet.create_net(n_layers=3,n_neurons=10)
    model.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer='adam')
    model.summary()
    model.fit(x_train[i],y_train[i],
              validation_data=(x_val[i],y_val[i]), 
              shuffle = True,
              epochs=500, 
              batch_size = 16,
              callbacks = [es],
              verbose=1)
    y_pred = model.predict(X)
    fqi = []
    for j in range(X.shape[-1]):
        Xsca= np.ones_like(X)
        Xsca[...,j] = np.zeros_like(Xsca[...,j])
        y_pre = model.predict(Xsca*X)
        y_pre2 = model.predict(X)
        fqi.append(np.linalg.norm(np.argmax(y_pre,1)-np.argmax(y_pre2,1)))
    FQI += np.array(fqi)


plt.tight_layout()
plt.figure(figsize=(33,13))
plt.bar(FeatureLabel,FQI/(n_folds*X.shape[0]))
plt.savefig('FQI')


a = np.zeros((X.shape[-1],X.shape[-1]))
sc = StandardScaler() 
for i in X:
    x = sc.fit_transform(i)
    Cov_map = np.cov(x.T)
    a += Cov_map
    
   
plt.tight_layout()
plt.figure(figsize=(27,27))
sns.set(font_scale=1.5)
hearmap = sns.heatmap(a/len(X),
                      annot=True,
                      square=True,
                      fmt = '.2f',
                      cmap = 'Blues',
                      xticklabels=FeatureLabel, yticklabels=FeatureLabel)
hearmap.xaxis.tick_top()
plt.savefig('FeatureCorrelation2')


# =============================================================================
# 
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV
# from sklearn.datasets import make_classification
# 
# svc = SVC(kernel="linear")
# rfecv = RFECV(estimator=keras_estimator, step=1, cv=StratifiedKFold(2),
#               scoring='accuracy')
# rfecv.fit(x,  yy)
# 
# print("Optimal number of features : %d" % rfecv.n_features_)
# 
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()
# 
# model = create_dense([698,8])
# loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# model.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer='adam')
#             
# import tempfile
# model_dir = tempfile.mkdtemp()
# keras_estimator = tf.keras.estimator.model_to_estimator(
#     keras_model=model, model_dir=model_dir)
# keras_estimator.train(input_fn=input_fn, steps=500)
# eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
# print('Eval result: {}'.format(eval_result))
# 
# =============================================================================
