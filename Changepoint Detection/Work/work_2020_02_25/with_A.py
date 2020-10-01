# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:08:25 2020

@author: xinyu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:24:46 2020

@author: xinyu
"""

#import ruptures as rpt
from scipy import signal, stats, fftpack, arange
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from pylab import plot, show, title, xlabel, ylabel, subplot, bar
from sklearn.model_selection import train_test_split, ShuffleSplit
import tensorflow as tf
import random
import numpy as np
import matplotlib.ticker as ticker
import os

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

def generate_train(data,Po,l):
    data = signal.resample(data,l)
    train_x, n = interpo_point(data,Po)
    if n >= l:
        return np.zeros(l),0,0
    train_y = np.zeros(l)
    #train_y[n] = 2 
    train_y[n:] = 1
    return train_x, train_y, n
def padding_len(xseq):
    Len = []
    for i in range(len(xseq)):
        Len.append(len(xseq[i]))
    pad_len = np.array(Len).max()
    return pad_len
def data_padding(Xseq,Yseq):
    padlen = padding_len(Xseq)
    padded_xseq = np.zeros((len(Xseq),padlen,Xseq[0].shape[-1]))
    padded_yseq = np.zeros((len(Yseq),padlen,1))
    for i in range(len(Xseq)):
        padded_xseq[i,...] = np.pad(Xseq[i],((0,padlen-len(Xseq[i])),(0,0)),'constant',constant_values = 0.)
        padded_yseq[i,...] = np.pad(Yseq[i].reshape(-1,1),((0,padlen-len(Yseq[i])),(0,0)),'constant',constant_values = 0.)
    return padded_xseq, padded_yseq

def gate(y_pred):
    y_bool = y_pred!=y_pred[-1]
    y =y_pred[y_bool==True]
    idx = np.where(y_pred == y[-1])[0][-1]
    y_ = y_pred[:idx+1]
    return y_
def evaluate_plot(*ypre,to):
    To = np.zeros(to.shape)
    for y_pre in ypre:
        for i in range(y_pre.shape[0]):
            y_ = gate(y_pre[i])
            To[i] = np.argmax(y_)
        diff_to = np.array(To)-np.array(to)
        diff_to = np.where(np.abs(diff_to)>=3,3,diff_to)
        unique, count = np.unique(diff_to, return_counts=True)
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        ax.bar(unique, 100*count/len(to))
        
        x_ticks_labels = ['-2','-1','0','1','2','>3']
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


def evaluate_dense(y_pred,Yseq):
    xfrom=0
    xto=0
    idxo_pred = []
    idxo_true = []
    for i in Yseq:
        xto = xto + len(i)
        idxo_pred.append(np.argmax(y_pred[xfrom:xto]))
        idxo_true.append(np.argmax(i))
        xfrom = xto
        
    Err = np.array([idxo_pred[i] - idxo_true[i] for i in range(len(idxo_pred))])
    
    err = np.where(Err>=3,3,Err)
    err = np.where(err<=-3,-3,err)
    unique, count = np.unique(err, return_counts=True)
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.bar(unique, 100*count/len(err))
    
    x_ticks_labels = ['<-3','-2','-1','0','1','2','>3']
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    plt.xticks(np.arange(7)-3, x_ticks_labels)
    ax.set_xticklabels(x_ticks_labels)
    
    xlabel('Error in timesteps')
    ylabel('Distribution')
    plt.show()
    return Err

def FoErrEvalDense(y_,F,Fos):
    xfrom=0
    xto=0
    Fo_pred = []
    j=0
    for i in Yseq:
        xto = xto + len(i)
        Fo_pred.append(F[j][np.argmax(y_[xfrom:xto])])
        xfrom = xto
        j+=1
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

###############################################################################

'''load measurement data and result data from respective txt-files, skip header rows'''
dataArr = np.loadtxt("U:/Data/OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
resArr = np.loadtxt("U:/Data/OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
infoArr = np.loadtxt("U:/Data/OUTPUT_meritve_ll.txt", skiprows=1) #additional information
IDs = int(dataArr[-1,0]) #Anzahl Messdurchläufe in der Datenbank mit jeweils bis zu 3 Messungen

DB = [dataArr,resArr,infoArr]


initialize = 1
trim = 1


loopThroughData = 1
sequence = 1

ruleBased = 0
findBySort = 0
findBySum = 1

optSegment = 0
segFlinear = 1
segFslinear = 0
segPlt = 0
savefigure = 0

NNseq = 1

showPlt = 0
evaluation = 1

gradWin = 0
gradFwd = 0
gradBwd = 1
loadmodel = 1


if initialize != 0:
    dataList = []
    featureList = []
    X = np.zeros(shape = (1,12))
    Y = np.zeros(shape=1)
    F_origin = np.zeros(shape=1)
    raw_F = []
    FSA = []
    FSAtrim = []
    tos = np.zeros(1+IDs*3)
    Fos = np.zeros(1+IDs*3)
    idxos = np.zeros(1+IDs*3)
    Fos_pred = np.zeros(1+IDs*3)
    idxos_pred = np.zeros(1+IDs*3)
    Xseq = []
    Yseq = []
    Frng = np.zeros(1+IDs*3)
    label=[]
    measLen = np.zeros(1+IDs*3)
 
    #errArr = np.zeros((IDs,6))
if loopThroughData !=0:
    #r = random.randrange(1,935)
    id_from, id_to = loop_from_to(IDs, id_from=0, steps=0) #id_from=0 loops through all
    loopNo = (id_from-1)*3 #adjust initial value according to id_from
    for i in range(id_from,id_to):
        for j in range(1,4):
            print(str(i) + ", " + str(j))    
            loopNo = loopNo + 1
            
            dat = timeSeries(DB,i,j).data #data
            res = timeSeries(DB,i,j).points #data labels
            info = timeSeries(DB,i,j).info #additional labels
            
            dataList.append(dat)
            
            t = dat.t
            F = dat.F
            s = dat.s
            p = dat.p
            A = dat.A
            
            to = res.to #opening time
            Fo = res.Fo #opening force
            
            
            if F.size > 0:
                Frng[loopNo] = max(F)-min(F)
            if to.size > 0:
                tos[loopNo] = to
    
            
            if F.size > 0 and Fo.size > 0 and to.size > 0 and to > 0.1:
                to = round(to[0],2)
                idxo = np.searchsorted(t,to,'left')
                idxos[loopNo] = idxo
                Frng[loopNo] = max(F)-min(F)
                raw_F.append(F)
                y = np.zeros_like(F)
                y[idxo] = 1
                Fos[loopNo] = Fo
            else:
                 continue
            
            
            if trim != 0:
                trim_to = np.argmax(F) + 5
                measLen[loopNo] = trim_to
                label.append([i,j])
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
                gradA = grad_bwd(A,i,1)
                grad2A = grad_fwd(gradA,i,1)
                grad3A = grad_bwd(grad2A,i,1)
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
            
            An = norm2(A)
            gradAn = norm2(gradA)
            grad2An = norm2(grad2A)
            grad3An = norm2(grad3A)
            if NNseq != 0:                               
                x = np.column_stack((Fn, gradFn, grad2Fn, grad3Fn,
                                     sn, gradsn, grad2sn, grad3sn, 
                                     An, gradAn, grad2An, grad3An))                   
                
                if X.size != 0:
                    X = np.concatenate((X, x), axis=0)
                    Y = np.concatenate((Y, y), axis=0)
                    F_origin = np.concatenate((F_origin, F), axis=0)
                else:
                    X = x
                    Y = y
                    F_origin = F
            
###########-Form data in sequential
            if sequence > 0:
                Xseq.append(np.array(x))
                Yseq.append(np.array(y))
                
               
#####################################################################################
# Datasplit
                
               
n = np.arange(len(Xseq)) 
label = np.array(label)
test_size = 0.15
trial,uniidx = np.unique(label[:,0],return_index=True)
test_vol = test_size*len(Xseq)/len(trial)
ss = ShuffleSplit(n_splits=1,test_size=test_vol,random_state=10)
for trainidx, testidx in ss.split(trial):
    test = uniidx[testidx]
    
train = n[np.isin(n, test,invert=True)]   

entire = n

    
#Data for Feedforward Dense NN    
Trainx = [Xseq[i] for i in train]
Testx = [Xseq[i] for i in test]
Trainy = [Yseq[i] for i in train]
Testy = [Yseq[i] for i in test]
x_train = np.array([y for x in Trainx for y in x]  )
x_test = np.array([y for x in Testx for y in x] )
y_train = np.array([y for x in Trainy for y in x]) 
y_test= np.array([y for x in Testy for y in x] )    
XX = np.array([y for x in Xseq for y in x]  )
YY = np.array([y for x in Yseq for y in x]  )

#Data for Recurrent layer  
sequential_x, sequential_y = data_padding(Xseq,Yseq)
idxo_masked = np.array([np.argmax(y) for y in sequential_y])
recurr_xtrain = np.array([sequential_x[i] for i in train])
recurr_xtest = np.array([sequential_x[i] for i in test])
recurr_ytrain = np.array([sequential_y[i] for i in train])
recurr_ytest = np.array([sequential_y[i] for i in test])



########################################################################
if loadmodel:
    model_dense = tf.keras.models.load_model('dense_withA.h5')
    model_lstm = tf.keras.models.load_model('lstm_withA.h5')
    model_bilstm = tf.keras.models.load_model('bilstm_withA.h5')


else:
    # Feedforward Dense NN
    model_dense = tf.keras.models.Sequential()
    model_dense.add(tf.keras.layers.Dense(80, input_dim=12, activation='relu'))
    model_dense.add(tf.keras.layers.Dense(50, activation='relu'))
    model_dense.add(tf.keras.layers.Dense(30, activation='relu'))
    model_dense.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model_dense.summary()
    model_dense.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    history = model_dense.fit(x_train, y_train,validation_split=0.15, shuffle = True,
                  epochs=20)    
    
    
    
    
    # LSTM NN
    def create_lstm(input_shape):
        inputs = tf.keras.Input(shape = input_shape)
        x = tf.keras.layers.Masking(mask_value=0.)(inputs)
        x1 = tf.keras.layers.LSTM(20, activation='relu',return_sequences = True,name='lstm')(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu'))(x1)
        x6 = tf.keras.layers.BatchNormalization()(x6)
        x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation='relu'))(x6)
        x6 = tf.keras.layers.BatchNormalization()(x6)
        x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation='relu'))(x6)
        x6 = tf.keras.layers.BatchNormalization()(x6)
        y_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(x6)
        return tf.keras.Model(inputs=inputs,outputs=y_) 
    
    model_lstm = create_lstm(recurr_xtrain.shape[1:])
    
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model_lstm.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer='adam')
    model_lstm.summary()
    model_lstm.fit(recurr_xtrain, recurr_ytrain,
              validation_split=0.15, 
              shuffle = True,
              epochs=50, 
              verbose=1)
    
    
    # Bidirectional LSTM
    def create_bilstm(input_shape):
        inputs = tf.keras.Input(shape = input_shape)
        x = tf.keras.layers.Masking(mask_value=0.)(inputs)
        x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, activation='relu',return_sequences = True,name='lstm'))(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu'))(x1)
        x6 = tf.keras.layers.BatchNormalization()(x6)
        x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation='relu'))(x6)
        x6 = tf.keras.layers.BatchNormalization()(x6)
        x6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation='relu'))(x6)
        x6 = tf.keras.layers.BatchNormalization()(x6)
        y_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(x6)
        return tf.keras.Model(inputs=inputs,outputs=y_)   
    
    model_bilstm = create_bilstm(recurr_xtrain.shape[1:])
    
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model_bilstm.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer='adam')
    model_bilstm.summary()
    model_bilstm.fit(recurr_xtrain, recurr_ytrain,
              validation_split=0.15, 
              shuffle = True,
              epochs=50, 
              verbose=1)
    





if evaluation:
    
    dataset = 'test' #  test or entire
    
    if dataset == 'entire':
        dataset = n
        data = XX
        recurr_data = sequential_x
    elif dataset == 'test':
        dataset = test
        data = x_test
        recurr_data = recurr_xtest
        
    Force = [raw_F[i] for i in dataset]
    Fo = Fos[np.where(Fos!=0)[0]]
    Force_open = np.array([Fo[i] for i in dataset])
    
    
    y_pred_dense = model_dense.predict(data) 
    Err_dense = evaluate_dense(y_pred_dense,np.array(Yseq)[dataset])
    #FoPre, FoErr, count =  FoErrEvalDense(y_pred_dense,Force,Force_open)
    print('Error (Dense) <= 2 steps: %.4f'%(100*len(np.where(abs(Err_dense)<=2)[0])/len(Err_dense)))
    
    y_pred_lstm = model_lstm.predict(sequential_x[dataset])
    Err_lstm, lstm_err = evaluate_plot(y_pred_lstm,to=idxo_masked[dataset])
    #FoPre, FoErr, count =  FoErrEval(y_pred_lstm,Force,Force_open)
    print('Error (LSTM) <= 2  steps: %.4f'%(100*np.sum(lstm_err[np.where(Err_lstm<=2)[0]])/np.sum(lstm_err)))

    y_pred_bilstm = model_bilstm.predict(sequential_x[dataset])
    Err_bilstm, bilstm_err = evaluate_plot(y_pred_bilstm,to=idxo_masked[dataset])
    #FoPre, FoErr, count =  FoErrEval(y_pred_bilstm,Force,Force_open)
    print('Error (BiLSTM) <= 2  steps: %.4f '%(100*np.sum(bilstm_err[np.where(Err_bilstm<=2)[0]])/np.sum(bilstm_err)))




