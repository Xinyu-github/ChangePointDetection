# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans,AffinityPropagation,SpectralClustering,MeanShift
from sklearn.datasets import make_blobs
import matplotlib.ticker as mtick
from scipy import signal, stats, fftpack, arange
from scipy.signal import butter, lfilter,resample
import matplotlib.pyplot as plt
from pylab import plot, show, title, xlabel, ylabel, subplot, bar
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import tensorflow as tf
import numpy as np
import matplotlib.ticker as ticker
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
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

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def delta(var):
    if var.size > 0:
        dvar = var[1:]-var[:-1]
        dvar = np.append(dvar,0)
        dvar[-1] = dvar[-2]        
        return dvar
    else:
        return var
    
def grad(var, idNum):
    if var.size > 0:
        if idNum < 775:
            grad = np.gradient(var, 0.1)
        else:
            grad = np.gradient(var, 0.05)
        return grad
    else:
        return var
    
def diff(var):
    dvar = np.diff(var)
    dvar = np.append(dvar,dvar[-1])
    return dvar

def diff2(var):
    dvar = np.diff(var,n=2)
    dvar = np.append(dvar,dvar[-1])
    return dvar

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

def grad_win(var,idNum,win_len):
    if var.size > 0:
        if idNum < 775:
            sampRate = 0.1
        else:
            sampRate = 0.05
            
        grad_win = np.zeros(len(var))
        
        grad_win[0] = (var[1] - var[0])/sampRate
        for l in range(1,win_len):
            grad_win[l] = (var[l+l] - var[l-l])/(2*l*sampRate)
        for l in range(win_len,len(var)-win_len):
            grad_win[l] = (var[l+win_len] - var[l-win_len])/(2*win_len*sampRate)
        for l in range(len(var)-win_len, len(var)):
            grad_win[l] = (var[-1] - var[l -((len(var)-1)-l)])/(2*((len(var)-1)-l)*sampRate)
        grad_win[-1] = (var[-1] - var[-2])/sampRate
        return grad_win
    else:
        return var
    
def smooth(var,win_len,polyorder):
    try:
        var = signal.savgol_filter(var, window_length=win_len, polyorder=polyorder, mode='nearest')
        return var
    except ValueError:
        return var

def norm_std(var):
    var = (var-var.mean(0))/var.std(0)
    return var
def norm(var):
    if max(var)-min(var) != 0:
        var = (var-min(var))/(max(var)-min(var))
        return var  
    else:
        #print("Input could not be normalized")
        return var

def norm2(var):
    if max(abs(var)) != 0:
        var = var/(max(abs(var)))
        return var  
    else:
        return var
    
def norm3(var):
    for i in range(0,len(var)):    
        if var[i] < 0:
            var[i]=0
    try:
        var = var/(max(abs(var)))
        return var  
    except ValueError:
        return var
    
def detectOpt(signalY, signalX):          
    
    signal = np.column_stack((signalY,signalX))
    n_bkps = 4
    #c = rpt.costs.CostLinear().fit(signal)                    
    # change point detection
    model = "linear" # "l1", "l2", "rbf", "normal", "linear"
    algo = rpt.Dynp(model=model, min_size=1, jump=1).fit(signal)
    bkps = algo.predict(n_bkps)  
    return bkps

def butter_lowpass(lowcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_zerophase_filter(data,lowcut,fs,order=2):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def round_arr(data):
    data[:,2] = np.round(data[:,2],3)
    data[:,3] = np.round(data[:,3],7)
    return data

def create_y(data,idxo):
    y = np.zeros(len(data))
    y[idxo[0]:] = 1
    return y

def frequency_sepectrum(x, sf):
    x = x - np.average(x)  # zero-centering
    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fftpack.fft(x) / n   # fft computing and normalization
    x = x[range(n // 2)]
    return frqarr, abs(x)

def interpo_point(x, p): #makes sure that Fo exists in resampled data
    n = 0
    for i in x:
        if i == p:
            break
        elif i > p:
            if (i-p)>(x[n-1]-p):
                x[n-1] = p
                n -= 1
            else:
                x[n] = p
            break
        n += 1
    return x, n


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

    padded_xseq = np.zeros((len(Xseq),padlen,Xseq[0].shape[-1]))
    padded_yseq = np.zeros((len(Yseq),padlen,1))
    for i in range(len(Xseq)):
        padded_xseq[i,...] = np.pad(Xseq[i],((0,padlen-len(Xseq[i])),(0,0)),'constant',constant_values = 0.)
        padded_yseq[i,...] = np.pad(Yseq[i].reshape(-1,1),((0,padlen-len(Yseq[i])),(0,0)),'constant',constant_values = 0.)
    return padded_xseq, padded_yseq

def gate(y_pred):
    
    # Data was padded to a same length
    # this function is used to ignore padded part
    y_bool = y_pred!=y_pred[-1]
    y =y_pred[y_bool==True]
    idx = np.where(y_pred == y[-1])[0][-1]
    y_ = y_pred[:idx+1]
    return y_

def evaluate_plot(*ypre,to,sf_id):
    # This function is used to evaluate the predicted results and plot
    # *ypre  =  predicted y
    #  to    =  open timing
    #  sf_id =  sample frequence ID; 
    #           = 1 when ID <= 775
    #           = 2 when ID > 775
    To = np.zeros(to.shape)
    for y_pre in ypre:
        for i in range(y_pre.shape[0]):
            # ignore padded part
            y_ = gate(y_pre[i])
            # to uniform erorr of prediction in different sample frequence
            # after ID 775, sample frequency doubled. 
            # 2 steps in this part = 1 step before this part
            step_scalar = sf_id[i]
            To[i] = np.argmax(y_)
            
        # Difference of true and predicted open timing
        # measuments after 775 are divided by 2 to make sure their time error are same
        diff_to = (np.array(To)-np.array(to))//step_scalar
        
        # error later than 3 steps are hold in +3
        diff_to = np.where(diff_to>=3,3,diff_to)
        # error earlier than 3 steps are hold in -3
        diff_to = np.where(diff_to<=-3,-3,diff_to)
        # unique = variation of error steps; count = the number of samples in each error step
        unique, count = np.unique(diff_to, return_counts=True)
        
        # Plot
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        ax.bar(unique, 100*count/len(to))
        
        x_ticks_labels = ['-3','-2','-1','0','1','2','>3']
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        yticks = ticker.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        plt.xticks(np.arange(6)-2, x_ticks_labels)
        #ax.set_xticklabels(x_ticks_labels)

        xlabel('Error in timesteps')
        ylabel('Distribution')
        plt.show()
    return unique, count, diff_to

def FoErrEval(y_,F,Fos):
    Fo_pred = [F[i][np.argmax(gate(y_[i]))] for i in range(y_.shape[0])]
    FoErr = (np.array(Fos)-np.array(Fo_pred))/np.array(Fos)
    FoErr = np.array([round(x,2) for x in FoErr])
    FoErr = np.where(FoErr<-1,-1,FoErr)
    for i in np.linspace(0,0.4,5):
        low = np.where(FoErr>i)[0]
        up = np.where(FoErr<=i+0.1)[0]
        FoErr[np.intersect1d(low, up)]=i+0.1
        low = np.where(FoErr>=-i-0.1)[0]
        up = np.where(FoErr<-i)[0]
        FoErr[np.intersect1d(low,up)]=-i-0.1
    FoErr = np.where(FoErr>0.5,1,FoErr)
    FoErr = np.where(FoErr<-0.5,-1,FoErr)
    unique, count = np.unique(FoErr, return_counts=True)
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.bar(100*unique, 100*count/len(FoErr),width=8)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    fmt = '%.2f%%'
    xticks = ticker.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    xlabel('Fo_err/Fo [%]')
    ylabel('Distribution')
    plt.show()
    return Fo_pred,unique,count

def decorrelate(X):
    cov = X.T.dot(X)/float(X.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    return decorrelated

def rmse(y,y_):
    return sqrt(mean_squared_error(y, y_))


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

def create_lstm2(input_shape):
    #inputs = tf.keras.Input(shape = input_shape)
    ins = [612,8]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0, input_shape=ins))
    model.add(tf.keras.layers.LSTM(8, return_sequences = True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8, activation='relu')))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5, activation='relu')))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='relu')))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid')))
    print(model.summary())
    return model 

def evaluate_errorstep(*ypre,to,sf_id):
    To = np.zeros(to.shape)
    for y_pre in ypre:
        for i in range(y_pre.shape[0]):
            # ignore padded part
            y_ = gate(y_pre[i])
            # to uniform erorr of prediction in different sample frequence
            # after ID 775, sample frequency doubled. 
            # 2 steps in this part = 1 step before this part
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
###############################################################################

'''load measurement data and result data from respective txt-files, skip header rows'''
loadStatic = 0
if loadStatic:
    dataArr = np.loadtxt("U:\Data\OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
    resArr = np.loadtxt("U:\Data\OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
    infoArr = np.loadtxt("U:\Data\OUTPUT_meritve_ll.txt", skiprows=1) #additional information
    vibrArr = np.loadtxt("U:\Data\\vibration_filter.txt")
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
sequence = 1

savefigure = 0

NNseq = 1

showPlt = 0
evaluate = 1

gradWin = 0
gradFwd = 0
gradBwd = 1
loadmodel = 0


if initialize != 0:
    X = np.zeros(shape = (1,8))
    Y = np.zeros(shape=1)
    raw_F = []
    tos = np.zeros(1+IDs*3)
    Fos = np.zeros(1+IDs*3)
    idxos = np.zeros(1+IDs*3)
    Fos_pred = np.zeros(1+IDs*3)
    idxos_pred = np.zeros(1+IDs*3)
    Xseq = []
    Yseq = []
    Frng = np.zeros(1+IDs*3)
    label=[]
    sf_id = []
    measLen = np.zeros(1+IDs*3)
 
if loopThroughData !=0:
    id_from, id_to = loop_from_to(IDs, id_from=0, steps=0) #id_from=0 loops through all
    loopNo = (id_from-1)*3 #adjust initial value according to id_from
    for i in range(id_from,id_to):
        for j in range(1,4):
            print(str(i) + ", " + str(j))    
            loopNo = loopNo + 1
            
            dat = timeSeries(DB,i,j).data #data
            res = timeSeries(DB,i,j).points #data labels
            info = timeSeries(DB,i,j).info #additional labels
                        
            t = dat.t
            F = dat.F
            s = dat.s
            p = dat.p
            A = dat.A
            
            to = res.to #opening time
            Fo = res.Fo #opening force
                        
            if F.size > 0 and Fo.size > 0 and to.size > 0 and to > 0.1:
                idxo = np.searchsorted(t,to,'left')
                idxos[loopNo] = idxo
                Frng[loopNo] = max(F)-min(F)
                raw_F.append(F)
                Fos[loopNo] = Fo
                to = round(to[0],2)
                
                y = np.zeros_like(F)
                y[idxo] = 1                
            else:
                 continue
            
             
            if trim != 0:
                trim_to = np.argmax(F)
                measLen[loopNo] = trim_to
                label.append([i,j])
                if i <=775:
                    sf_id.append(1)
                else:
                    sf_id.append(2)
                F = F[:trim_to]
                s = s[:trim_to]
                A = A[:trim_to]
                t = t[:trim_to]
                y = y[:trim_to]
                      
            
            if gradFwd != 0:
                gradF = grad_fwd(F,i,1)
                grad2F = grad_fwd(gradF,i,1)
                grad3F = grad_fwd(grad2F,i,1)
                grads = grad_fwd(s,i,3)
                grad2s = grad_fwd(grads,i,3)
                grad3s = grad_fwd(grad2s,i,3)
                #gradA = grad_fwd(A,i,5)
                #grad2A = grad_fwd(gradA,i,5)
                #grad3A = grad_fwd(grad2A,i,5)
            elif gradBwd !=0:
                gradF = grad_bwd(F,i,1)
                grad2F = grad_fwd(gradF,i,1)
                grad3F = grad_bwd(grad2F,i,1)
                grads = grad_bwd(s,i,1)
                grad2s = grad_fwd(grads,i,1)
                grad3s = grad_bwd(grad2s,i,1)
            else:
                gradF = grad_win(F,i,1)
                grad2F = grad_win(gradF,i,1)
                grad3F = grad_win(grad2F,i,1)
                grads = grad_win(s,i,3)
                grad2s = grad_win(grads,i,3)
                grad3s = grad_win(grad2s,i,3)
                #gradA = grad_win(A,i,1)
                #grad2A = grad_win(gradA,i,1)
                #grad3A = grad_win(grad2A,i,1)
            
            
            Fn = norm2(F)
            gradFn = norm2(gradF)
            grad2Fn = norm2(grad2F)
            grad3Fn = norm2(grad3F)
            
            sn = norm2(s)
            gradsn = norm2(grads)
            grad2sn = norm2(grad2s)
            grad3sn = norm2(grad3s)
            
            if NNseq != 0:                               
                x = np.column_stack((Fn, gradFn, grad2Fn, grad3Fn,
                                     sn, gradsn, grad2sn, grad3sn))
                
                if X.size != 0:
                    X = np.concatenate((X, x), axis=0)
                    Y = np.concatenate((Y, y), axis=0)
                else:
                    X = x
                    Y = y
            
###########-Form data in sequential
            if sequence > 0:
                Xseq.append(np.array(x))
                Yseq.append(np.array(y))
                
               
############### Data preprocessing
 
sequential_x, sequential_y = data_padding(Xseq,Yseq)
idxo_masked = np.array([np.argmax(y) for y in sequential_y])
            
########################################################################

########################################################################
if loadmodel:
    model_lstm = tf.keras.models.load_model('model_fold1.h5')


else:

    
# LSTM NN
    # number of fold in cross-validation
    n_splits = 5
    
    # level of evaluation in error step corresponding to low sample frequency
    # 1 step = 0.01s
    # -3 and 3 means <=-3 and >=3
    error = np.array([-3.,-2.,-1.,0.,1.,2.,3.])
    
    # record of Accuracy
    Acc = np.zeros((n_splits,len(error)))
    
    # measurement ID = subject ID
    # used to split train&val/test data
    groups = np.array(label)[:,0] 
    
    # subject-wise k-fold train&val/test split
    gkf = GroupKFold(n_splits)
    nn=0
    for train_val_idx, test_idx in gkf.split(sequential_x, sequential_y, groups):
        # split whole dataset into train&val and test dataset
        x_train_val = sequential_x[train_val_idx]
        x_test = sequential_x[test_idx]
        y_train_val = sequential_y[train_val_idx]
        y_test = sequential_y[test_idx]
        
        #subject-wise 
        gss = GroupShuffleSplit(n_splits=1,test_size=0.1,random_state=10)

        for train_idx, val_idx in gss.split(x_train_val,y_train_val,groups[train_val_idx]):
            # split train&val dataset into train and validation dataset
            x_train = x_train_val[train_idx]
            x_val = x_train_val[val_idx]
            y_train = y_train_val[train_idx]
            y_val = y_train_val[val_idx]
            # earlystopping to restore weights to the epoch with lowest val_loss
            es = EarlyStopping(monitor='val_loss',
                               mode='min',
                               verbose=1,
                               patience = 10,
                               restore_best_weights=True)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            
            model_lstm = create_lstm(x_train.shape[1:])

            model_lstm.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer='adam')

            model_lstm.fit(x_train, y_train,
                      validation_data=(x_val,y_val), 
                      shuffle = True,
                      epochs=150, 
                      batch_size = 16,
                      callbacks = [es],
                      verbose=1)
            model_lstm.save('model_fold%i.h5'%(nn+1))
            y_pred_lstm = model_lstm.predict(x_test)
            Err_lstm, lstm_err = evaluate_errorstep(y_pred_lstm,to=idxo_masked[test_idx],sf_id = np.array(sf_id)[test_idx])
            
            # arrange all errors in range ('<-3','-2','-1','0','1','2','>3')
            for i in range(len(error)):
                try:
                    Acc[nn,i]=lstm_err[Err_lstm==error[i]]
                except:
                    continue
                
            # convert absolute quantity into percentage
            Acc[nn,:] = Acc[nn,:]*100/len(test_idx)
            nn+=1
            tf.keras.backend.clear_session()


#std and mean accuracy of signal without A
err_mean = Acc.mean(0)
err_range = np.array([err_mean-Acc.min(0),Acc.max(0)-err_mean])

width = 0.5         # the width of the bars


fig, ax = plt.subplots(figsize=(10,7))
p1 = ax.bar(error, err_mean, width, bottom=0, yerr=err_range)

ax.set_title('Classification accuracy in error step related to vibration signal')
ax.set_xticks(error )
ax.set_xticklabels(('<-3','-2','-1','0','1',
                    '2','>3'))
fmt = '%.00f%%'
xticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(xticks)
ax.autoscale_view()

# Add this loop to add the annotations
total = len(sequential_x)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.02%}'.format(height/100), (x, y + height + 1),
                xytext=(1, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
    
plt.savefig('results')
plt.show()

            