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


def padding_len(xseq):
    # Calculating the maximal sequence length in all samples
    Len = []
    for i in range(len(xseq)):
        Len.append(len(xseq[i]))
    pad_len = np.array(Len).max()
    return pad_len
    
def data_padding(Xseq,Yseq):
    # Padding data sequence to same length
    # Xseq = input signals
    # Yseq = output signals
    
    # find the length to be padded to
    padlen = padding_len(Xseq)
# =============================================================================
#     if padlen<1000:
#         padlen = 1000
#     else:
#         padlen = padlen;
# =============================================================================

    padded_xseq = np.zeros((len(Xseq),padlen,Xseq[0].shape[-1]))
    padded_yseq = np.zeros((len(Yseq),padlen,1))
    for i in range(len(Xseq)):
        padded_xseq[i,...] = np.pad(Xseq[i],((0,padlen-len(Xseq[i])),(0,0)),'constant',constant_values = 0.)
        padded_yseq[i,...] = np.pad(Yseq[i].reshape(-1,1),((0,padlen-len(Yseq[i])),(0,0)),'constant',constant_values = 0.)
    return padded_xseq, padded_yseq

def gate(y_pred,mask):
    
    # Data was padded to a same length
    # this function is used to ignore padded part
    
    
    idx = np.where(mask == True)[0][-1]
    y_ = y_pred[:idx+1]
    return y_

def evaluation_plot(Acc,error,plot_form,title,save,current_time):
    
    err_mean = Acc.mean(0)
    
    err_range = np.array([err_mean-Acc.min(0),Acc.max(0)-err_mean])
    
            # the width of the bars
    
    
    fig, ax = plt.subplots(figsize=(10,7))
    
    if plot_form=='step':
        width = 0.5
        ax.set_title('Classification accuracy in error step ')
        ax.set_xticks(error )
        ax.set_xticklabels(('<-3','-2','-1','0','1',
                            '2','>3'))
    else:
        width = 0.1 
        ax.set_title('Deviation of force at predicted change point')
        ax.set_xticks(error )
        ax.set_xticklabels(('0%','20%','40%','60%','80%',
                            '>100%'))
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
    
    # fig, ax = plt.subplots(figsize=(10,7))
    # ax.set_title(title)
    # if plot_form=='step':
    #     labels = ('<-3','-2','-1','0','1',
    #                         '2','>3')
    # else:
    #     labels = ('0%','1%','2%','3%','10%','>100%')
        
    
    # ax.set_xticklabels(labels)
    # p1 = ax.boxplot(Acc,notch=True,showmeans=True)
    # fmt = '%.00f%%'
    # xticks = mtick.FormatStrFormatter(fmt)
    # ax.yaxis.set_major_formatter(xticks)
    # ax.autoscale_view()
    # for ii in range(len(labels)):
    #     ax.text(ii+1,Acc[:,ii].max()+1,round(Acc[:,ii].mean(0),2),fontsize=15)

    # if save:
    #     plt.savefig('./Results/results_' + current_time + '/' + plot_form + '_error_' + current_time)
    # plt.show()
    
    # return 0


def evaluate_errorforce(y_,F,Fos,mask):
    Fo_pred = np.array([F[i][np.argmax(gate(y_[i],mask[i]))] for i in range(len(F))])
    FoErr = np.abs(Fos.T-Fo_pred).T/np.array(Fos)
    FoErr = np.where(FoErr>1,1,FoErr)
    FoErr = np.where(FoErr<0.001,0,FoErr)
    err =  [0,0.01,0.02,0.03,0.1,1]
    for i in range(len(err)-1): #np.linspace(0,1,6):
        low = np.where(FoErr>err[i])[0]
        up = np.where(FoErr<=err[i+1])[0]
        FoErr[np.intersect1d(low, up)]=err[i+1]
    unique, count = np.unique(FoErr, return_counts=True)
   
    return unique,count

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
        
    return unique, count

def data_split(idx,n_splits,groups):
    train=[]
    val=[]
    test=[]
    fold_id = np.arange(n_splits)
    #gkf = GroupKFold(n_splits)
    gkf = GroupShuffleSplit(n_splits=n_splits,test_size=0.50,random_state=10)
    for train_val_idx, test_idx in gkf.split(idx, groups=groups):
        # split whole dataset into train&val and test dataset
        idx_train_val = idx[train_val_idx]
        idx_test = idx[test_idx]
        #subject-wise 
        gss = GroupShuffleSplit(n_splits=1,test_size=0.1,random_state=10)

        for train_idx, val_idx in gss.split(idx_train_val,groups=groups[train_val_idx]):
            
            train.append(idx_train_val[train_idx])
            val.append(idx_train_val[val_idx])
            test.append(idx_test)
    Split_idx = pd.DataFrame([train,val,test],index=['Train','Validation','Test'],columns =fold_id )
    return Split_idx
###############################################################################

'''load measurement data and result data from respective txt-files, skip header rows'''
loadStatic = 1
if loadStatic:
    dataArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
    resArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
    infoArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/OUTPUT_meritve_ll.txt", skiprows=1) #additional information
    #vibrArr = np.loadtxt("D:/GitClone/alphatest/Changepoint Detection/Database/vibration_filter.txt")
    
      # dataArr = np.loadtxt(".\Changepoint Detection\Database\OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
      # resArr = np.loadtxt(".\Changepoint Detection\Database\OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
      # infoArr = np.loadtxt(".\Changepoint Detection\Database\OUTPUT_meritve_ll.txt", skiprows=1) #additional information
    # vibrArr = np.loadtxt("U:\Data\\vibration_filter.txt")
else:
    from tkinter import filedialog
    current_directory = filedialog.askdirectory()
    data_path = os.path.join(current_directory,"OUTPUT_diagram_ll_clean.txt")   
    res_path = os.path.join(current_directory,"OUTPUT_tocke_ll_clean_2020.txt") 
    info_path = os.path.join(current_directory,"OUTPUT_meritve_ll.txt")
    vibr_path = os.path.join(current_directory,"vibration_filter.txt")
    
    dataArr = np.loadtxt(data_path, skiprows=1) #contains raw measurement data
    resArr = np.loadtxt(res_path, skiprows=1) #contains determined opening point for each measurement
    infoArr = np.loadtxt(info_path, skiprows=1) #additional information
    vibrArr = np.loadtxt(vibr_path, skiprows=1)


IDs = int(dataArr[-1,0]) #Anzahl Messdurchläufe in der Datenbank mit jeweils bis zu 3 Messungen

DB = [dataArr,resArr,infoArr]


initialize = 1
trim = 1
loopThroughData = 1
loadmodel = 1


if initialize != 0:
    F_raw = [] #List of Numpy-Arrays containing raw force signals (of different length)
    Fos = [] #List of Numpy-Arrays containing opening force from databse (one element per array)
    Xseq = [] #List of Numpy-Arrays containing unnormalized & unpadded input signals for NN
    Yseq = [] #List of Numpy-Arrays containing labels for opening points (unpadded)
    label=[] #List of Lists containing IDs and measurement No. in database
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
            if F.size > 0 and Fo.size > 0 and to.size > 0 and to > 0.1:
                
                to = round(to[0],2)                
                idxo = np.searchsorted(t,to,'left') # find index of opening time
                F_raw.append(F) #append force signal of current meas. to list of arrays F_raw
                Fos.append(Fo) #append opening force of current meas. to list of arrays Fos
                y = np.zeros_like(F) #create label array: 1 at opening point index, else 0
                y[idxo] = 1
            else:
                 continue
            
             
            if trim != 0:
                # cut out signals to the point with maximal force
                trim_to = np.argmax(F) + 10
                # record measurement IDs
                label.append([i,j])
                # record sample frequency
                if i <=775:
                    sf_id.append(1)
                else:
                    sf_id.append(2)
                    
                F = F[:trim_to]
                s = s[:trim_to]
                y = y[:trim_to]
                      
            
            # grediant of signals
            gradF = grad_bwd(F,i,1)
            grad2F = grad_fwd(gradF,i,1)
            grad3F = grad_bwd(grad2F,i,1)
            grads = grad_bwd(s,i,1)
            grad2s = grad_fwd(grads,i,1)
            grad3s = grad_bwd(grad2s,i,1)
         
            x = np.column_stack((F, gradF, grad2F, grad3F,
                                 s, grads, grad2s, grad3s))
                
            
            # Form data in sequential
            Xseq.append(np.array(x))
            Yseq.append(np.array(y))
                
               
############### Data preprocessing
Xseq_norm = [Normalize(sample) for sample in Xseq]
sequential_x, sequential_y = data_padding(Xseq_norm,Yseq)
idxo_masked = np.array([np.argmax(y) for y in sequential_y])

########################################################################


# LSTM NN
# number of fold in cross-validation
n_splits = 10

# step errors are seperated as ('<-3','-2','-1','0','1','2','>3')
error_step = np.array([-3.,-2.,-1.,0.,1.,2.,3.])

# force deviation are seperated as ('0%','1%','2%','3%','10%','>=100%')
error_force = np.array([0.,0.01,0.02,0.03,0.1,1.])

Step_Err = np.zeros((n_splits,len(error_step)))
Force_Err = np.zeros((n_splits,len(error_force)))

# measurement ID = subject ID
groups = np.array(label)[:,0] 

# use current time and date as file name
now = datetime.now()
current_time = now.strftime("%d_%m-%H_%M")

# subject-wise k-fold train&val/test split
if loadmodel:
    # specify here the saving time of the file to be loaded
    save_time = '23_06-10_17'
    
    
    split_idx = pd.read_hdf('./model_split_storage/data_'+ save_time +'/split_info_'+ save_time + '.h5')
else:
    split_idx=data_split(np.arange(len(sequential_x)),n_splits,groups)
    os.mkdir('./model_split_storage/data_'+ current_time )
    split_idx.to_hdf('./model_split_storage/data_'+ current_time +'/split_info_'+ current_time + '.h5','Data')



for i in range(n_splits):
    # split index
    test_idx = split_idx[i]['Test']
    train_idx = split_idx[i]['Train']
    val_idx = split_idx[i]['Validation']
    
    x_test = sequential_x[test_idx]
    y_test = sequential_y[test_idx]

    x_train = sequential_x[train_idx]
    y_train = sequential_y[train_idx]
    
    x_val = sequential_x[val_idx]
    y_val = sequential_y[val_idx]
    # earlystopping to restore weights to the epoch with lowest val_loss
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience = 5,
                       restore_best_weights=True)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    if loadmodel:
        #model_lstm = tf.keras.models.load_model('.\Test Reduced LSTM\model_fold{}_reducedLSTM.h5'.format(i+1))
        #model_lstm = tf.keras.models.load_model('./model_split_storage/data_'+ save_time +'/model_fold%i_'%(i+1) + save_time +'.h5')
        model_lstm = tf.keras.models.load_model('./model_split_storage/data_'+ save_time +'/model_fold%i_reduced_'%(i+1) + save_time +'.h5')
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
        model_lstm.save('./model_split_storage/data_'+ current_time +'/model_fold%i_reduced_'%(i+1) + current_time +'.h5')
    
    # Prediction with test data
    y_pred_lstm = model_lstm.predict(x_test)
    # predicted output should ignore the masked part of prediction
    Mask = x_test!=0
    
    # the number (_count) and variations (_unique) of force and step error in above defined range 
    ForceErr_unique,ForceErr_count = evaluate_errorforce(y_pred_lstm,np.array(F_raw)[test_idx],np.array(Fos)[test_idx],Mask)
    StepErr_unique, StepErr_count = evaluate_errorstep(y_pred_lstm,to=idxo_masked[test_idx],sf_id = np.array(sf_id)[test_idx],mask = Mask)
    
    # arrange all step errors in range ('<-3','-2','-1','0','1','2','>3')
    for j in range(len(error_step)):
        try:
            Step_Err[i,j]=StepErr_count[StepErr_unique==error_step[j]]
        except:
            continue
        
    # arrange all force deviation in range ('0%','1%','2%','3%','10%','>=100%')
    for j in range(len(error_force)):
        try:
            Force_Err[i,j] = ForceErr_count[np.abs(ForceErr_unique-error_force[j])<=0.001]
        except:
            continue
        
    # convert absolute quantity into percentage
    Step_Err[i,:] = Step_Err[i,:]*100/len(test_idx)
    Force_Err[i,:] = Force_Err[i,:]*100/len(test_idx)
    
    # clear memory occupied by training
    tf.keras.backend.clear_session()


#####################################################
# plot evaluation of step error
    
os.mkdir('./Results/results_' + current_time )
evaluation_plot(Step_Err,
                error_step,
                'step',
                title = 'Classification accuracy in error step (reduced LSTM)',
                save=True,
                current_time = current_time)
# plot evaluation of force deviation
evaluation_plot(Force_Err,
                error_force,
                'force',
                title = 'Deviation of force at predicted change point (reduced LSTM)',
                save=True,
                current_time = current_time)
 
            