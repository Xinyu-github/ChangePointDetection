import numpy as np
import sklearn as skl
import ruptures as rpt
from sklearn.linear_model import LinearRegression
from scipy import signal, stats, fftpack, arange
from scipy.signal import butter, lfilter
import matplotlib
import matplotlib.pyplot as plt
from pylab import plot, show, title, xlabel, ylabel, subplot
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import random


###############################################################################

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

###############################################################################

'''load measurement data and result data from respective txt-files, skip header rows'''
dataArr = np.loadtxt("D:/Users/Ms/.spyder-py3/OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
resArr = np.loadtxt("D:/Users/Ms/.spyder-py3/OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
infoArr = np.loadtxt("D:/Users/Ms/.spyder-py3/OUTPUT_meritve_ll.txt", skiprows=1) #additional information
idxo_pred_LinReg = np.loadtxt("D:/Users/Ms/.spyder-py3/diffSteps_linRegr_full.txt")

IDs = int(dataArr[-1,0]) #Anzahl Messdurchläufe in der Datenbank mit jeweils bis zu 3 Messungen

DB = [dataArr,resArr,infoArr]


initialize = 1

loopThroughData = 1
trim = 1

ruleBased = 0
findBySort = 0
findBySum = 0

optSegment = 0
segFlinear = 0
segFslinear = 0
segPlt = 0

NNseq = 1
pltNN = 0

showPlt = 0
evaluate = 1

gradWin = 0
gradFwd = 0
gradBwd = 1 #combination of forward and backward differentiation

if initialize != 0:
    dataList = []
    featureList = []
    X = np.zeros(shape = (1,8))
    Y = np.zeros(shape=1)
    
    tos = np.zeros(1+IDs*3)
    Fos = np.zeros(1+IDs*3)
    idxos = np.zeros(1+IDs*3)
    
    Fos_pred = np.zeros(1+IDs*3)
    idxos_pred = np.zeros(1+IDs*3)
    
    Frng = np.zeros(1+IDs*3)
    
    measLen = np.zeros(1+IDs*3)
 
    
if loopThroughData !=0:
    #r = random.randrange(1,935)
    id_from, id_to = loop_from_to(IDs, id_from=0, steps=0) #id_from=0 loops through all
    loopNo = (id_from-1)*3 #adjust initial value according to id_from
    for i in range(id_from,id_to):
        for j in range(1,4):
            #print(str(i) + ", " + str(j))    
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
            if Fo.size > 0:
                Fos[loopNo] = Fo
    
            
            if F.size > 0 and Fo.size > 0 and to.size > 0 and to > 0.1:
                to = round(to[0],2)
                idxo = np.searchsorted(t,to,'left')
                idxos[loopNo] = idxo
                y = np.zeros_like(F)
                y[idxo] = 1
            else:
                 continue
            
            
            if trim != 0:
                trim_to = np.argmax(F) + 5
                measLen[loopNo] = trim_to
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
            
            Fn = norm2(F)
            gradFn = norm2(gradF)
            grad2Fn = norm2(grad2F)
            grad3Fn = norm2(grad3F)
            sn = norm2(s)
            gradsn = norm2(grads)
            grad2sn = norm2(grad2s)
            grad3sn = norm2(grad3s)
            
            
            if NNseq != 0:                
                #x = np.column_stack((Fn, gradFn, grad2Fn, grad3Fn))                
                x = np.column_stack((Fn, gradFn, grad2Fn, grad3Fn, sn, gradsn, grad2sn, grad3sn))                
                
                if X.size != 0:
                    X = np.concatenate((X, x), axis=0)
                    Y = np.concatenate((Y, y), axis=0)
                else:
                    X = x
                    Y = y
            
            p_set = info.p_set
            p_sys = info.p_sys
            d0 = info.d0
            A0 = np.pi*d0*d0*0.25
            
            if all(g == 0 for g in grads):
                s_available = 0
            else:
                s_available = 1
                
            if s_available != 0:
                sig = np.column_stack((gradF,grad2F,grads,grad2s))
            else:
                sig = np.column_stack((gradF,grad2F))
    
            
###########-Rule based approach
            if ruleBased > 0:
                            
                if findBySort > 0:            
                    '''sort'''
                    index = np.linspace(0, trim_to, num=trim_to, endpoint=False)
                    values = np.column_stack((index, grad2F))
                    grad2_sort = values[values[:,1].argsort()]
                    
                    '''find point where grad2F is lowest and grads is positive'''
                    for point in grad2_sort:
                        idx = int(point[0])
                        if gradF[idx] > 0:
                            if s_available == 1 and grads[idx] > 0 and grad2s[idx] > 0:
                                idxo_pred = idx
                                idxos_pred[loopNo] = idxo_pred
                                break
                            idxo_pred = idx
                            idxos_pred[loopNo] = idxo_pred
                            break
                        
                        else:
                            continue
                            
                if findBySum > 0:
                    '''check where criteria for changepoint are met'''
                    valids = []
                    valid_vals = []
                    
                    if s_available == 1:
                        for p in range(0,trim_to):
                            if gradF[p] > 0 and grad2F[p] < 0 and grads[p] > 0 and grad2s[p] > 0:
                                valid_val = - norm2(grad2F)[p] + norm2(grads)[p] + norm2(grad2s)[p]
                                valid_vals.append(valid_val)
                                valids.append(p)              
                    else:
                        for p in range(0,trim_to):
                            if gradF[p] > 0 and grad2F[p] < 0:
                                valid_val = gradF[p] - grad2F[p]
                                valid_vals.append(valid_val)
                                valids.append(p)
                    
                    if valids:
                        values = np.column_stack((valids, valid_vals))
                        score_sort = values[values[:,1].argsort()]
                        idxo_pred = score_sort[:,0][-1]
                        idxos_pred[loopNo] = idxo_pred
                        Fos_pred[loopNo] = F[int(idxo_pred)]
                                    
    
###########-Optimal Segmentation        
            if optSegment > 0:
                
                if segFlinear > 0:
                    signalX = np.ones(len(t))
                    o_bkp = 1
                
                if segFslinear > 0:
                    if s_available == 1:
                        signalX = grads
                        o_bkp = 0
                    else:
                        signalX = np.ones(len(t))
                        o_bkp = 1
                
                bkps = detectOpt(norm(gradF), norm(signalX))
                idxo_pred = bkps[o_bkp] - 1
                idxos_pred[loopNo] = idxo_pred
                Fos_pred[loopNo] = F[int(idxo_pred)]
                
                if bkps:
                    tbkps = np.zeros(len(bkps)-1)
                    for l in range(0,len(bkps)-1):
                        tbkps[l] = t[bkps[l]-1]
                
                bkpsTrue = bkps
                bkpsCal = bkps
                bkpsCal = np.zeros_like(bkps)
                bkpsCal = np.zeros_like(bkps)
                for x in range(0,len(bkpsTrue)):
                    bkpsTrue[x] = bkps[-1]
                 
                if segPlt > 0:            
                    fig, (ax,) = rpt.display(norm(gradF), bkpsTrue, bkpsCal)
                    fig, (ax,) = rpt.display(F, bkpsTrue, bkpsCal)
                    fig, (ax,) = rpt.display(s, bkpsTrue, bkpsCal)
              
############-Plot
            if showPlt > 0:
                if ruleBased > 0:
                    plt.figure()
                    plt.plot(norm(F[:trim_to]))
                    plt.plot(norm(gradF[:trim_to]))
                    plt.plot(norm(grad2F[:trim_to]))
                    plt.plot(norm(s[:trim_to]))
                    #plt.plot(norm(A[:trim_to]))
                    plt.axvline(idxo, color="red", linestyle = "--")
                    plt.axvline(idxo_pred,color="black", linestyle = "--") 
                else:    
                    font = {'family' : 'Arial',
                            'weight' : 'normal',
                            'size'   : 22}
                    
                    matplotlib.rc('font', **font)
                    
                    fig = plt.figure(figsize=(12,8))
                    plt.plot(t,norm(F), color="red", linewidth=3, label="Kraft")
                    
                    #plt.plot(t,norm(s), c="cornflowerblue", lw=3, label='Hub')
                    #plt.plot(t,norm(A), c="green", lw=3, label='Schall')
                    ax = fig.add_subplot(111)        
                    ax.set_xlim(left=0, right=round(t[-1]-0.5,0))
                    ax.set_ylim(bottom=0, top=1)
                    ax.set_xticks(np.arange(0, round(t[-1]+0.5,0), 2))
                    ax.set_yticks(np.arange(0, 1.2, 0.2))
                    ax.tick_params(axis='both', width=2)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    plt.xlabel('Zeit [s]')
                    plt.ylabel("Kraft normiert [-]")
                    
                    plt.axvline(to, color="black", linestyle = "--", lw=3, label='Öffnungspkt.')
                    plt.legend(loc=2,framealpha=1)
                    #plt.savefig(str(i) + "_" + str(j) + "_F+s.png")
                    
                    if segPlt !=0:
                        t0 = 0
                        colors = ["#4286f4", "#f44174"]#["#4286f4", "#f44174"]
                        c = 0
                        col = colors[0]
                        for p in tbkps:
                            plt.axvline(p, color="blue", linestyle = "dotted", lw=3)
                            plt.axvspan(ymin=0,ymax=1,xmin=t0,xmax=p,facecolor=col, alpha=0.1)    
                            
                            t0 = p
                            if c == 0:
                                col = colors[1]
                                c = 1
                            else:
                                col = colors[0]
                                c = 0
                            
                        plt.axvspan(ymin=0,ymax=1,xmin=t0,xmax=t[-1],facecolor=col, alpha=0.1)                    
                        plt.savefig(str(i) + "_" + str(j) + ".png")
                    
########-NN-Methods
if NNseq != 0:
#    for i in range(0,len(Y)):
#        Y[i] = float(Y[i])
    
    
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.15,random_state=1)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(8, input_dim=8, activation='relu'))
    model.add(keras.layers.Dense(5, activation='relu'))
    model.add(keras.layers.Dense(3, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.FalsePositives(),'accuracy'])
     
    history = model.fit(x_train, y_train,validation_data=(x_test,y_test), shuffle = True,
                  epochs=50)
    
    y_pred = model.predict(X)
    xfrom = 0
    xto = 0
    for i in range(0,len(idxos)):
        xfrom = xto + 1
        xto = xto + int(measLen[i])
        if xto-xfrom > 0 and y_pred[i] !=0:
            idxos_pred[i] = np.argmax(y_pred[xfrom:xto])

    if pltNN != 0:
        font = {'family' : 'Arial',
                'weight' : 'normal',
                'size'   : 22}
                    
        matplotlib.rc('font', **font)
        
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)        

        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        ax.tick_params(axis='both', width=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
                
    #plot_model(model, to_file='model.png')
    
        
if evaluate > 0:
    mismatch = [0]
    idxoErr = np.ones_like(idxos)*1000
    FoErrAbs = np.ones_like(idxos)*1000
    FoErrRel = np.ones_like(idxos)*1000
    for i in range(1,len(idxos_pred)):
        if dataList[i-1].F.size > 0: 
            Fos_pred[i] = dataList[i-1].F[int(idxos_pred[i])]
    #Fos_pred[i] = F[int(idxos_pred[i])]
    
    
    for i in range(0,len(idxos)):
        if idxos[i] > 0:
            idxoErr[i] = idxos[i] - idxos_pred[i]
            if idxoErr[i] > 2 or idxoErr[i] < -2:
                mismatch.append(i)
            FoErrAbs[i] = abs(Fos[i] - Fos_pred[i])
            FoErrRel[i] = safe_div(FoErrAbs[i],Frng[i])
        
    n = 0
    k = 1 # index of measurement projected from ID,measNo to 1...2803
    indexTransform = np.zeros(shape=(2803,3))
    for i in range(1,935):
        for j in range(1,4):
            indexTransform[int(k),0] = i
            indexTransform[int(k),1] = j
            indexTransform[int(k),2] = k
            k = k + 1            

        
    idxoErrMasked = idxoErr[idxoErr[:]!=1000]
    FoErrAbsMasked = FoErrAbs[FoErrAbs[:]!=1000]
    FoErrRelMasked = FoErrRel[FoErrRel[:]!=1000]
    
    C=idxoErrMasked[idxoErrMasked!=1000]
    count0 = sum(C==0)
    count_1 = sum(C==-1)
    count1 = sum(C==1)
    count_2 = sum(C==-2)
    count2 = sum(C==2)
    count_R = sum(C<-2)
    countR = sum(C>2)
    
    counts=[count_R, count_2, count_1, count0, count1, count2, countR]
    counts.append(sum(counts))
    
    countsRel = np.empty_like(counts) + .0
    for i in range(0, len(counts)):
        countsRel[i] = round(100*(counts[i]/counts[-1]),1)
    countLessThan2 = sum(countsRel[1:6])

    bins2 = [0,.01,.02,.03,.05,.1,.2, 1]
    hist2 = np.histogram(FoErrRelMasked, bins=bins2)
 
        
        
        
        
        
                
    