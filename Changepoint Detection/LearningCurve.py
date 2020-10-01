from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit,train_test_split
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import ChangepointDetection as CPD
class LearningCurve():
    seed=1
    def __init__(self,
                 X=None,
                 Y=None,
                 model = None,
                 shuffle =True,
                 train_sizes=None,
                 cv=None,
                 es = None):
# =============================================================================
        # X: Training vector
        # Y: Target relative to X for classification or regression
        # model: An object of that type which is cloned for each validation.
        # train_sizes: Relative or absolute numbers of training examples that will be used to generate the learning curve. 
        # cv: Determines the cross-validation splitting strategy.
        # es: earlystopping for training
# =============================================================================
        self.train_sizes = train_sizes 
        self.cv=cv
        self.model = model
        self.X = X
        self.Y = Y
        self.shuffle = True
        self.es = es
    def learning(self):
        
        # shuffle
        if self.shuffle:
            idx=tf.random.shuffle(np.arange(self.X.shape[0]),seed=self.seed)
        
        # in this case: error in 2 steps
        train_scores = []
        test_scores = []
        
        for train_sizes in self.train_sizes:
            train_score=[]
            test_score=[]
            for cv in range(self.cv):
                model_=tf.keras.models.clone_model(self.model)
                model_.compile(loss='binary_crossentropy',metrics=tf.keras.metrics.Precision())
                # select samples from dataset with size = train_sizes
                selected_idx = np.random.choice(idx,train_sizes)
                # split data
                X_train, X_test, y_train, y_test = train_test_split(X[selected_idx,...],
                                                                    Y[selected_idx,...],
                                                                    test_size=0.2, 
                                                                    random_state=self.seed)
                model_.fit(X_train,y_train,
                           validation_data=( X_test, y_test),
                           shuffle = True,
                           epochs=200,
                           callbacks=[self.es],
                           batch_size = 4)
                train_score.append(self.scoring(model_,X_train,y_train))
                test_score.append(self.scoring(model_,X_test,y_test))
            train_scores.append(np.mean(train_score))
            test_scores.append(np.mean(test_score))
        return train_scores,test_scores
                
                
        
    def scoring(self,estimator,X,Y):
        y_ = estimator.predict(X)
        error = abs(np.argmax(y_,1)-np.argmax(Y,1))
        accept = error[error<=2]
        acc= len(accept)/len(error)
        return acc

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

    
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
es = EarlyStopping(monitor='loss',
                           mode='min',
                           verbose=1,
                           patience = 5,
                           restore_best_weights=True)
BuildNet = CPD.BuildClassifier(NetType = 'LSTM',
                           input_shape = X.shape[1:],
                           n_layers=3,n_neurons=100,
                           n_lstmunit = 20)
model = BuildNet.create_net()
opt = tf.keras.optimizers.Adam()
model.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer=opt)
model.summary()
train_sizes = [50,100,250,500,X.shape[0]]
lc = LearningCurve(X=X,
                 Y=Y,
                 model = model,
                 shuffle =True,
                 train_sizes=train_sizes,
                 cv=2,
                 es = es)
train_scores,test_scores = lc.learning()

# Plot
fig,axs = plt.subplots()
axs.set_xlabel("Training examples")
axs.set_ylabel("Score")
axs.grid()
axs.plot(train_sizes,train_scores, 'o-', color="r",
                 label="Training score")
axs.plot(train_sizes,test_scores,'o-', color="g",
                 label="Cross-validation score")
axs.legend(loc="best")
#plt.savefig('learning_curve_lstm')



