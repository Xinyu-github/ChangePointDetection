from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit,learning_curve
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import ChangepointDetection as CPD



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

# set grid search parameter
num_layers = np.array([1,5,10])
num_neurons = np.array([5,20,100])
num_lstmunits = np.array([5,10,20,50])# number of lstm unit for lstm net
learning_rate = np.array([0.001])

# initialize accuracy record array and variation of error
Acc = np.zeros((len(learning_rate),len(num_layers),len(num_neurons)))
error_step = np.array([-3.,-2.,-1.,0.,1.,2.,3.]) #step errors are seperated as ('<-3','-2','-1','0','1','2','>3')
error_force = np.array([0.,0.01,0.02,0.03,0.1,1.]) #force deviation are seperated as ('0%','1%','2%','3%','10%','>=100%')




es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience = 10,
                       restore_best_weights=True)
    
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
for lr in learning_rate:
    for num_lstmunit in  num_lstmunits:
        for num_layer in num_layers:
            for num_unit in num_neurons:
                error_step = np.array([-3.,-2.,-1.,0.,1.,2.,3.]) #step errors are seperated as ('<-3','-2','-1','0','1','2','>3')
                error_force = np.array([0.,0.01,0.02,0.03,0.1,1.]) #force deviation are seperated as ('0%','1%','2%','3%','10%','>=100%')
                Step_Err= np.zeros((n_folds, len(error_step)+1))
                Force_Err = np.zeros((n_folds, len(error_force)+1))
                for i in range(n_folds):
                    BuildNet = CPD.BuildClassifier(NetType = 'LSTM',
                               input_shape = x_train[0].shape[1:],
                               n_layers = num_layer,
                               n_neurons=num_unit,
                               n_lstmunit = num_lstmunit)
                    model = BuildNet.create_net()
                    opt = tf.keras.optimizers.Adam(learning_rate=lr)
                    model.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer=opt)
                    model.summary()
                    model.fit(x_train[i],y_train[i],
                              validation_data=(x_val[i],y_val[i]), 
                              shuffle = True,
                              epochs=500, 
                              batch_size = 16,
                              callbacks = [es],
                              verbose=1)
                    y_pred = model.predict(x_test[i])
                    # predicted output should ignore the masked part of prediction
                    Mask = x_test[i]!=0
                    StepErr,  steperr = CPD.Evaluation().evaluate_errorstep( y_pred,to=idxos[split_idx[i]['Test']],mask = Mask,error_step=error_step)
                    Step_Err[i,:] = StepErr
                loc = (np.where(num_lstmunits==num_lstmunit)[0][0],
                       np.where(num_layers==num_layer)[0][0],
                       np.where(num_unit == num_neurons)[0][0])
                Acc[loc] = Step_Err.mean(0)[-1]
            
            
best_loc = (np.where(Acc==Acc.max())[0][0],np.where(Acc==Acc.max())[1][0],np.where(Acc==Acc.max())[2][0])            
print('Best NN:'+' learning rate='+str(learning_rate[best_loc[0]])+
      ' layers=' + str(num_layers[best_loc[1]])+' units=' + 
      str(num_neurons[best_loc[2]]))            
            
            
