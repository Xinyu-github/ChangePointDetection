# -*- coding: utf-8 -*-

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from datetime import datetime

class dataClass:
    def __init__(self, dataIn, idNum, measNum):
        dataIn = dataIn[(dataIn[:,0]==idNum) & (dataIn[:,6]==measNum)]
        self.t = dataIn[:,1]
        self.s = dataIn[:,2]
        self.p = dataIn[:,3]
        self.F = dataIn[:,4]
        self.A = dataIn[:,5]
        
class dataSet:
    def __init__(self, varIn):
        self.dat = varIn
                
class resClass:
    def __init__(self, resIn, idNum, measNum):
        res = resIn[(resIn[:,0]==idNum) & (resIn[:,1]==measNum)].T
        self.to = res[2]
        self.Fo = res[3]
        self.so = res[4]
        self.tc = res[5]
        self.Fc = res[6]
        self.sc = res[7]
        
class infoClass:
    def __init__(self, infoIn, idNum):
        self.info = infoIn[(infoIn[:,0]==idNum)].T
        self.p_set = self.info[1]
        self.F_tol = self.info[2]
        self.s_tol = self.info[3]
        self.p_sys = self.info[4]
        #self.DN = self.info[5]
        self.PN = self.info[5]
        self.d0 = self.info[6]
        self.kalt_heiss = self.info[7]
     
class timeSeries:
    def __init__(self, db, idNum, measNum):
        data = db[0]
        result = db[1]
        info = db[2]
        self.idNum = idNum
        self.measNum = measNum
        self.data = dataClass(data, idNum, measNum)
        self.points = resClass(result, idNum, measNum)
        self.info = infoClass(info,idNum)

###############################################################################
        
def loop_from_to(IDs, id_from, steps):
    if id_from == 0:
        id_from = 1
        id_to = IDs + 1
    else:
        id_to = id_from + steps + 1
    return id_from, id_to



def grad_fwd(var,idNum,win_len):
    if var.size > 0:
        if idNum < 775:
            sampRate = 0.1
        else:
            sampRate = 0.05
            
        grad_fwd = np.zeros(len(var))
        
        for l in range(0,len(var)-win_len):
            grad_fwd[l] = (var[l+win_len] - var[l])/(win_len*sampRate)
        for l in range(len(var)-win_len, len(var)-1):
            w = (len(var)-1)-l
            grad_fwd[l] = (var[l+w] - var[l])/(w*sampRate)
        grad_fwd[len(var)-1] = (var[-1] - var[-2])/sampRate
        return grad_fwd
    else:
        return var
    
def grad_bwd(var,idNum,win_len):
    if var.size > 0:
        if idNum < 775:
            sampRate = 0.1
        else:
            sampRate = 0.05
            
        grad_bwd = np.zeros(len(var))
        grad_bwd[0] = (var[1] - var[0])/sampRate
        for l in range(1,win_len):
            grad_bwd[l] = (var[l] - var[0])/(win_len*sampRate)
        for l in range(win_len, len(var)):
            grad_bwd[l] = (var[l] - var[l-win_len])/(win_len*sampRate)
        return grad_bwd
    else:
        return var

def Normalize(var):
    NonZero = var.astype(bool).any(0)
    DoNorm = var[:,NonZero]
    var[:,NonZero] = DoNorm/abs(DoNorm).max(0)
    return var


def padding_len(X):
    # Calculating the maximal sequence length in all samples
    Len = []
    for i in range(len(X)):
        Len.append(len(X[i]))
    pad_len = np.array(Len).max()
    return pad_len
    
def data_padding(X,Y):
    # Padding data sequence to same length
    # X = input signals
    # Y = output signals
    
    # find the length to be padded to
    padlen = padding_len(X)
# =============================================================================
#     if padlen<1000:
#         padlen = 1000
#     else:
#         padlen = padlen;
# =============================================================================

    padded_X = np.zeros((len(X),padlen,X[0].shape[-1]))
    padded_Y = np.zeros((len(Y),padlen,1))
    for i in range(len(X)):
        padded_X[i,...] = np.pad(X[i],((0,padlen-len(X[i])),(0,0)),'constant',constant_values = 0.)
        padded_Y[i,...] = np.pad(Y[i].reshape(-1,1),((0,padlen-len(Y[i])),(0,0)),'constant',constant_values = 0.)
    return padded_X, padded_Y

def gate(y_pred,mask):
    
    # Data was padded to a same length
    # this function is used to ignore padded part
    
    
    idx = np.where(mask == True)[0][-1]
    y_ = y_pred[:idx+1]
    return y_

def evaluation_plot(Acc,error,plot_form,title,save,current_time):
    
    err_mean = Acc.mean(0)
    err_range = np.array([err_mean-Acc.min(0),Acc.max(0)-err_mean])
    fig, ax = plt.subplots(figsize=(10,7))
    if plot_form=='step':
        width = 0.5
        ax.set_title('Classification accuracy in error step ')
        ax.set_xticks(np.arange(len(error)+1)-3 )
        ax.set_xticklabels(('<-3','-2','-1','0','1',
                            '2','>3','with in 2'))
        error = np.arange(len(error)+1)-3
    else:
        width = 0.5
        ax.set_title('Deviation of force at predicted change point')
        ax.set_xticks( np.arange(len(error)+1)-3 )
        ax.set_xticklabels(('0%','1%','2%','3%','10%','>=100%','<=3%'))
        error = np.arange(len(error)+1)-3
    p1 = ax.bar(error, err_mean, width, bottom=0, yerr=err_range)
    fmt = '%.00f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(xticks)
    ax.autoscale_view()
        
    # Add this loop to add the annotations
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:.02%}'.format(height/100), (x, y + height + 1),
                    xytext=(1, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
       
    if save:
        plt.savefig('./Results/results_' + current_time + '/' + plot_form + '_error_' + current_time)
    plt.show()
    
    return 0
    
def plot_pred_to(x,y,y_,mask,i,save):
    x = gate(x[:,0],mask)
    y_ = gate(y_,mask)
    plt.figure(figsize=(10,7))
    plt.plot(x,label = 'Force',color = 'b')
    plt.plot(Normalize(y_),label = 'NN output',color = 'r')
    plt.axvline(x = np.argmax(y), ymin=0, ymax=1,label = 'Open point',color = 'black',linestyle='--' )
    plt.legend()
    if save:
        plt.savefig('./Results/prediction_plot_Nr%i'%i)
    plt.show()


def evaluate_errorforce(y_,F,Fos,mask):
    Fo_pred = np.array([F[i][np.argmax(gate(y_[i],mask[i]))] for i in range(len(F))])
    FoErr = np.abs(Fos.T-Fo_pred).T/np.array(Fos)
    foerr_orig = FoErr
    FoErr = np.where(FoErr>1,1,FoErr)
    FoErr = np.where(FoErr<0.001,0,FoErr)
    err =  [0,0.01,0.02,0.03,0.1,1]
    for i in range(len(err)-1): #np.linspace(0,1,6):
        low = np.where(FoErr>err[i])[0]
        up = np.where(FoErr<=err[i+1])[0]
        FoErr[np.intersect1d(low, up)]=err[i+1]
    unique, count = np.unique(FoErr, return_counts=True)
   
    return unique,count, foerr_orig

def create_lstm(input_shape):
    inputs = tf.keras.Input(shape = input_shape)
    x = tf.keras.layers.Masking(mask_value=0.)(inputs)
    x1 = tf.keras.layers.LSTM(20, activation='relu',return_sequences = True,name='lstm')(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.01)))(x1)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.01)))(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.01)))(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    y_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(x6)
    return tf.keras.Model(inputs=inputs,outputs=y_) 


def create_reducedlstm(input_shape):
    inputs = tf.keras.Input(shape = input_shape)
    x = tf.keras.layers.Masking(mask_value=0.)(inputs)
    x1 = tf.keras.layers.LSTM(5, activation='relu',return_sequences = True,name='lstm')(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10, activation='relu'))(x1)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10, activation='relu'))(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5, activation='relu'))(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    y_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(x6)
    return tf.keras.Model(inputs=inputs,outputs=y_) 

def create_dense(input_shape):
    inputs = tf.keras.Input(shape = input_shape)
    x = tf.keras.layers.Dense(8,activation='relu')(inputs)
    x = tf.keras.layers.Dense(5,activation='relu')(x)
    x = tf.keras.layers.Dense(3,activation='relu')(x)
    y_ = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs,outputs=y_) 


def Smooth_Force(F):
   # dF = np.gradient(F)
    
    for i in range(len(F)-1):
        F[i+1] = max(F[i],F[i+1])
        
    return F
def evaluate_errorstep(*ypre,to,sf_id,mask):
    To = np.zeros(to.shape)
    for y_pre in ypre:
        for i in range(y_pre.shape[0]):
            # ignore padded part
            y_ = gate(y_pre[i],mask[i])
# =============================================================================
#              to uniform erorr of prediction in different sample frequence
#              after ID 775, sample frequency doubled. 
#              2 steps in this part = 1 step before this part
# =============================================================================
            step_scalar = sf_id[i]
            To[i] = np.argmax(y_)
            
        # Difference of true and predicted open timing
        # measuments after 775 are divided by 2 to make sure their time error are same
        diff_to = (np.array(To)-np.array(to))//step_scalar
        
        # error later than 3 steps are hold in +3
        diff_to = np.where(diff_to>=3,3,diff_to)
        # error earlier than 3 steps are hold in -3
        diff_to = np.where(diff_to<=-3,-3,diff_to)
        unique, count = np.unique(diff_to, return_counts=True)
        
    return unique, count, np.abs(diff_to)

def data_split(idx, n_folds, groups):
    train=[]
    val=[]
    test=[]
    fold_count = np.arange(n_folds)
    #gkf = GroupKFold(n_folds)
    gkf = GroupShuffleSplit(n_splits=n_folds ,test_size=0.10, random_state=10)
    
    for train_val_split, test_split in gkf.split(idx, groups=groups):
        # generate indeces for train&val and test dataset
        idx_train_val = idx[train_val_split]
        idx_test = idx[test_split]
        
        #split train&val randomly into training and validation 
        gss = GroupShuffleSplit(n_splits=1,test_size=0.1,random_state=10) #here test_size is actually "validation_size"

        for train_split, val_split in gss.split(idx_train_val,groups=groups[train_val_split]):         
            train.append(idx_train_val[train_split])
            val.append(idx_train_val[val_split])
            test.append(idx_test)
    
    Split_idx = pd.DataFrame([train,val,test],index=['Train','Validation','Test'],columns=fold_count )
    return Split_idx
###############################################################################


initialize = 1 #inititalize list variables
trim = 1 #consider signals only up to maximum force (+10 timesteps)
loopThroughData = 1 #loop through the databse/load data to variables
lstm = 0 #train or evaluate lstm model
loadmodel = 0 #load trained neural network 
loadsplit = 1 #load data splits
dense = 1
#load measurement data and result data from respective txt-files, skip header rows
# =============================================================================
# dataArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
# resArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
# infoArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/OUTPUT_meritve_ll.txt", skiprows=1) #additional information
# =============================================================================
#vibrArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/vibration_filter.txt")

dataArr = np.loadtxt(".\Changepoint Detection\Database\OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
resArr = np.loadtxt(".\Changepoint Detection\Database\OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
infoArr = np.loadtxt(".\Changepoint Detection\Database\OUTPUT_meritve_ll.txt", skiprows=1) #additional information
# vibrArr = np.loadtxt("U:\Data\\vibration_filter.txt")


IDs = int(dataArr[-1,0]) #Number of valve tests in database (up to 3 measurements per test)
DB = [dataArr,resArr,infoArr] 


if initialize != 0:
    F_raw = [] #List of Numpy-Arrays containing raw force signals (of different length)
    Fos = [] #List of Numpy-Arrays containing opening force from databse (one element per array)
    Xseq = [] #List of Numpy-Arrays containing inputs/features for NN
    Yseq = [] #List of Numpy-Arrays containing labels for opening points
    label=[] #List of Lists containing labels (ID and no.) of valid measurements in database
    sf_id = [] #List with indicators for sampling frequency of the measurements in database

    
if loopThroughData !=0:
    id_from, id_to = loop_from_to(IDs, id_from=0, steps=0) #id_from=0 loops through all
    for i in range(id_from,id_to):
        for j in range(1,4):
            print('\r'+' Measurement ID:' + str(i) + ' no.' + str(j),end=' \r ') 
                   
########### extract data
            dat = timeSeries(DB,i,j).data #object containing time series of meas. ID i and meas. no. j
            res = timeSeries(DB,i,j).points #object containing force, stroke and time of valve opening and closing for meas. ID i and meas. no. j
            info = timeSeries(DB,i,j).info #additional information
                        
            t = dat.t #  Numpy-Array containing time of meas. ID i and meas. no. j
            F = dat.F # Numpy-Array containing force of meas. ID i and meas. no. j
            s = dat.s # Numpy-Array containing stroke of meas. ID i and meas. no. j
            #p = dat.p 
            #A = dat.A # vibration
            
            to = res.to #opening time of meas. ID i and meas. no. j
            Fo = res.Fo #opening force of meas. ID i and meas. no. j
            
########### check if data is valid for opening point estimation            
            if F.size > 0 and Fo.size > 0 and to.size > 0 and to > 0.1 and any(s):
                
                to = round(to[0],2)                
                idxo = np.searchsorted(t,to,'left') # find index of opening time
                F_raw.append(F) #append force signal of current meas. to list of arrays F_raw
                Fos.append(Fo) #append opening force of current meas. to list of arrays Fos
                y = np.zeros_like(F) #create label array: 1 at opening point index, else 0
                y[idxo] = 1
            else:
                 continue
             
            label.append([i,j]) #add measurement labels to list "labels"
            if i <=775: #record sampling frequency of 
                sf_id.append(1)
            else:
                sf_id.append(2)
                
            F = Smooth_Force(F)
########### trim signal to the point of maximal force
            if trim != 0:   
                trim_to = np.argmax(F) + 10
                F = F[:trim_to]
                s = s[:trim_to]
                y = y[:trim_to]                    
                      
            
########### compute gradients of signals stack and add to list of feature-arrays X
            gradF = grad_bwd(F,i,1)
            grad2F = grad_fwd(gradF,i,1)
            grad3F = grad_bwd(grad2F,i,1)
            grads = grad_bwd(s,i,1)
            grad2s = grad_fwd(grads,i,1)
            grad3s = grad_bwd(grad2s,i,1)
         
            x = np.column_stack((F, gradF, grad2F, grad3F,
                                s, grads, grad2s, grad3s))
                            
            Xseq.append(np.array(x))
            Yseq.append(np.array(y))
                
               
############### Data preprocessing
X = [Normalize(sample) for sample in Xseq] #Normalize signals/features in X
X, Y = data_padding(Xseq,Yseq) #padding
idxos = np.array([np.argmax(y) for y in Y]) #array with indexes of opening point from database

########################################################################

#group measurements by testing ID (= particular valve)
groups = np.array(label)[:,0] 

# get current time and date to use as file name
now = datetime.now()
current_time = now.strftime("%d_%m-%H_%M")


# k-fold splitting into train, validation and test; save indeces for splits in split_idx
if loadsplit:
    save_time = '27_07-12_54' #specify here the saving time of the file to be loaded
    
    #load splitting indeces
    split_idx = pd.read_hdf('./model_split_storage/data_'+ save_time +'/split_info_'+ save_time + '.h5')
     # number of folds for cross-validation
    n_folds =  split_idx.shape[-1]
else:
    n_folds = 10
    split_idx = data_split(np.arange(len(X)), n_folds, groups) #generate splitting indeces
    
    split_idx.to_hdf('./model_split_storage/data_'+ current_time +'/split_info_'+ current_time + '.h5','Data') #save splitting configuration

if loadmodel == 0:
    os.mkdir('./model_split_storage/data_'+ current_time ) #create directory to later save trained models in
#prepare error groups and arrays for holding results
error_step = np.array([-3.,-2.,-1.,0.,1.,2.,3.]) #step errors are seperated as ('<-3','-2','-1','0','1','2','>3')
error_force = np.array([0.,0.01,0.02,0.03,0.1,1.]) #force deviation are seperated as ('0%','1%','2%','3%','10%','>=100%')
Step_Err_Lstm = np.zeros((n_folds, len(error_step)+1))
Force_Err_Lstm = np.zeros((n_folds, len(error_force)+1))
Step_Err_Dense = np.zeros((n_folds, len(error_step)+1))
Force_Err_Dense = np.zeros((n_folds, len(error_force)+1))

if lstm:
    for i in range(n_folds):
        # split index
        test_idx = split_idx[i]['Test']
        train_idx = split_idx[i]['Train']
        val_idx = split_idx[i]['Validation']
        
        x_test = X[test_idx]
        y_test = Y[test_idx]
    
        x_train = X[train_idx]
        y_train = Y[train_idx]
        
        x_val = X[val_idx]
        y_val = Y[val_idx]
        
    
        # earlystopping to restore weights to the epoch with lowest val_loss
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience = 5,
                           restore_best_weights=True)
        
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        if loadmodel:
            #model_lstm = tf.keras.models.load_model('.\Test Reduced LSTM\model_fold{}_reducedLSTM.h5'.format(i+1))
            model_lstm = tf.keras.models.load_model('./model_split_storage/data_'+ save_time +'/model_fold%i_'%(i+1) + save_time +'.h5')
            #model_lstm = tf.keras.models.load_model('./model_split_storage/data_'+ save_time +'/model_fold%i_reduced_'%(i+1) + save_time +'.h5')
        else:
            # Build NN model
            # model_lstm = create_reducedlstm(x_train.shape[1:])
            model_lstm = create_lstm(x_train.shape[1:])
        
            model_lstm.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer='adam')
            
            # Training
            model_lstm.fit(x_train, y_train,
                      validation_data=(x_val,y_val), 
                      shuffle = True,
                      epochs=150, 
                      batch_size = 16,
                      callbacks = [es],
                      verbose=1)
            #model_lstm.save('./model_split_storage/data_'+ current_time +'/model_fold%i_'%(i+1) + current_time +'.h5')
            model_lstm.save('./model_split_storage/data_'+ current_time +'/model_fold%i_'%(i+1) + current_time +'.h5')
        
        # Prediction with test data
        y_pred_lstm = model_lstm.predict(x_test)
        # predicted output should ignore the masked part of prediction
        Mask = x_test!=0
        
        # the number (_count) and variations (_unique) of force and step error in above defined range 
        ForceErr_unique,ForceErr_count, foerr = evaluate_errorforce(y_pred_lstm,np.array(F_raw)[test_idx],np.array(Fos)[test_idx],Mask)
        StepErr_unique, StepErr_count, steperr = evaluate_errorstep(y_pred_lstm,to=idxos[test_idx],sf_id = np.array(sf_id)[test_idx],mask = Mask)
        
        # arrange all step errors in range ('<-3','-2','-1','0','1','2','>3')
        Step_Err_Lstm[i,-1] = 0
        for j in range(len(error_step)):
            try:
                Step_Err_Lstm[i,j]=StepErr_count[StepErr_unique==error_step[j]]
                if error_step[j] in error_step[1:-1]:
                    Step_Err_Lstm[i,-1] =  Step_Err_Lstm[i,-1] + StepErr_count[StepErr_unique==error_step[j]]
            except:
                continue
        
        # arrange all force deviation in range ('0%','1%','2%','3%','10%','>=100%')
        Force_Err_Lstm[i,-1] = 0
        for j in range(len(error_force)):
            try:
                Force_Err_Lstm[i,j] = ForceErr_count[np.abs(ForceErr_unique-error_force[j])<=0.001]
                if error_force[j] in error_force[0:3]:
                    Force_Err_Lstm[i,-1] =  Force_Err_Lstm[i,-1] + ForceErr_count[np.abs(ForceErr_unique-error_force[j])<=0.001]
            except:
                continue
            
        # convert absolute quantity into percentage
        Step_Err_Lstm[i,:] = Step_Err_Lstm[i,:]*100/len(test_idx)
        Force_Err_Lstm[i,:] = Force_Err_Lstm[i,:]*100/len(test_idx)
        
        # clear memory occupied by training
        tf.keras.backend.clear_session()
    #####################################################
    # plot evaluation of step error
        
    os.mkdir('./Results/results_' + current_time )
    evaluation_plot(Step_Err_Lstm,
                    error_step,
                    'step',
                    title = 'Classification accuracy in error step (reduced LSTM)',
                    save=True,
                    current_time = current_time)
    # plot evaluation of force deviation
    evaluation_plot(Force_Err_Lstm,
                    error_force,
                    'force',
                    title = 'Deviation of force at predicted change point (reduced LSTM)',
                    save=True,
                    current_time = current_time)

    
if dense:
    for i in range(n_folds):
        test_idx = split_idx[i]['Test']
        train_idx = split_idx[i]['Train']
        val_idx = split_idx[i]['Validation']
    
        x_test = X[test_idx]
        y_test = Y[test_idx]
    
        x_train = X[train_idx]
        y_train = Y[train_idx]
        
        x_val = X[val_idx]
        y_val = Y[val_idx]
        
        # reshape dataset to fit dense network
        train_shape = x_train.shape
        test_shape = x_test.shape
        val_shape = x_val.shape
        
        x_testDense = np.reshape(x_test,(test_shape[0]*test_shape[1],test_shape[-1]))
        y_testDense = np.reshape(y_test,(test_shape[0]*test_shape[1],1))
    
        x_trainDense = np.reshape(x_train,(train_shape[0]*train_shape[1],train_shape[-1]))
        y_trainDense = np.reshape(y_train,(train_shape[0]*train_shape[1],1))
        
        x_valDense =  np.reshape(x_val,(val_shape[0]*val_shape[1],val_shape[-1]))
        y_valDense =  np.reshape(y_val,(val_shape[0]*val_shape[1],1))
        
        # earlystopping to restore weights to the epoch with lowest val_loss
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience = 5,
                           restore_best_weights=True)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        if loadmodel:
            model_dense = tf.keras.models.load_model('./model_split_storage/data_'+ save_time +'/model_fold%i_'%(i+1) + save_time +'_dense.h5')
        else:
            # Build NN model
            # model_lstm = create_reducedlstm(x_train.shape[1:])
            model_dense = create_dense(x_trainDense.shape[1:])
        
            model_dense.compile(loss=loss,metrics=[tf.keras.metrics.Accuracy()],optimizer='adam')
            model_dense.summary()
            # Training
            model_dense.fit(x_trainDense, y_trainDense,
                      validation_data=(x_valDense,y_valDense), 
                      shuffle = True,
                      epochs=150, 
                      batch_size = train_shape[1],
                      callbacks = [es],
                      verbose=1)
            #model_lstm.save('./model_split_storage/data_'+ current_time +'/model_fold%i_'%(i+1) + current_time +'.h5')
            model_dense.save('./model_split_storage/data_'+ current_time +'/model_fold%i_'%(i+1) + current_time +'_dense.h5')
        
        # Prediction with test data
        y_pred_dense = model_dense.predict(x_testDense)
        # shape useful dataset back
        y_pred_dense = np.reshape(y_pred_dense, (test_shape[0],test_shape[1],1))
        # predicted output should ignore the masked part of prediction
        Mask = x_test!=0
        
        # the number (_count) and variations (_unique) of force and step error in above defined range 
        ForceErr_unique,ForceErr_count, foerrDense = evaluate_errorforce(y_pred_dense,np.array(F_raw)[test_idx],np.array(Fos)[test_idx],Mask)
        StepErr_unique, StepErr_count, steperrDense = evaluate_errorstep(y_pred_dense,to=idxos[test_idx],sf_id = np.array(sf_id)[test_idx],mask = Mask)
        
        # arrange all step errors in range ('<-3','-2','-1','0','1','2','>3')
        Step_Err_Dense[i,-1] = 0
        for j in range(len(error_step)):
            try:
                Step_Err_Dense[i,j]=StepErr_count[StepErr_unique==error_step[j]]
                if error_step[j] in error_step[1:-1]:
                    Step_Err_Dense[i,-1] =  Step_Err_Dense[i,-1] + StepErr_count[StepErr_unique==error_step[j]]
            except:
                continue
        
        # arrange all force deviation in range ('0%','1%','2%','3%','10%','>=100%')
        Force_Err_Dense[i,-1] = 0
        for j in range(len(error_force)):
            try:
                Force_Err_Dense[i,j] = ForceErr_count[np.abs(ForceErr_unique-error_force[j])<=0.001]
                if error_force[j] in error_force[0:3]:
                    Force_Err_Dense[i,-1] =  Force_Err_Dense[i,-1] + ForceErr_count[np.abs(ForceErr_unique-error_force[j])<=0.001]
            except:
                continue
            
        # convert absolute quantity into percentage
        Step_Err_Dense[i,:] = Step_Err_Dense[i,:]*100/len(test_idx)
        Force_Err_Dense[i,:] = Force_Err_Dense[i,:]*100/len(test_idx)
        
        # clear memory occupied by training
        tf.keras.backend.clear_session()
        
        
    #####################################################
    # plot evaluation of step error
        
    os.mkdir('./Results/results_' + current_time )
    evaluation_plot(Step_Err_Dense,
                    error_step,
                    'step',
                    title = 'Classification accuracy in error step (Dense)',
                    save=True,
                    current_time = current_time)
    # plot evaluation of force deviation
    evaluation_plot(Force_Err_Dense,
                    error_force,
                    'force',
                    title = 'Deviation of force at predicted change point (Dense)',
                    save=True,
                    current_time = current_time)








 
      
# plot  force and predicted nn output    
# plot_pred_to(x_test[i],y_test[i],y_pred_lstm[i],Mask[i],i,save = True)
# =============================================================================
# 
# for i in np.where(np.abs(steperrDense)>2)[0]:
#     plot_pred_to(x_test[i],y_test[i],y_pred_dense[i],Mask[i],i,save = False)
# =============================================================================
