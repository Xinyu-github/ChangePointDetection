
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from datetime import datetime
import copy
def grad_fwd(var,win_len):
    if var.size > 0:
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
    
def grad_bwd(var,win_len):
    if var.size > 0:
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
def loop_from_to(IDs, id_from, steps):
    if id_from == 0:
        id_from = 1
        id_to = IDs + 1
    else:
        id_to = id_from + steps + 1
    return id_from, id_to 
class timeSeries:
    def __init__(self, db, idNum, measNum):
        self.db = db
        self.idNum = idNum
        self.measNum = measNum
        dataIn = timeSeries.DataClass(self)
        self.t = dataIn[:,1]
        self.s = dataIn[:,2]
        self.p = dataIn[:,3]
        self.F = dataIn[:,4]
        self.A = dataIn[:,5]
        res = timeSeries.ResClass(self)
        self.to = res[2]
        self.Fo = res[3]
        self.so = res[4]
        self.tc = res[5]
        self.Fc = res[6]
        self.sc = res[7]
        info = timeSeries.InfoClass(self)
        self.p_set = info[1]
        self.F_tol = info[2]
        self.s_tol = info[3]
        self.p_sys = info[4]
        self.PN = info[5]
        self.d0 = info[6]
    def DataClass(self):
        dataIn = self.db[0][(self.db[0][:,0]==self.idNum) & (self.db[0][:,6]==self.measNum)]
        return dataIn
        
    def InfoClass(self):
        info = self.db[2][(self.db[2][:,0]==self.idNum)].T
        return info
        
    def ResClass(self):
        res = self.db[1][(self.db[1][:,0]==self.idNum) & (self.db[1][:,1]==self.measNum)].T
        return res
    
class DataPreparing():
    
    def __init__(self, 
                 IDs,
                 db,
                 loopThroughData = 1 #loop through the databse/load data to variables
                 ):
        self.IDs = IDs
        self.db = db
        self.loopThroughData = loopThroughData
        
    def segment(self,idNum,measNum,include_s):
        if self.loopThroughData:
            data = timeSeries(self.db,idNum,measNum)
            print('\r'+' Measurement ID:' + str(idNum) + ' no.' + str(measNum),end=' \r ') 
            if include_s:
                if data.F.size > 0 and data.Fo.size > 0 and data.to.size > 0 and data.to > 0.1 and data.s.any():
                    return data
                else:
                    return []
            else:
                if data.F.size > 0 and data.Fo.size > 0 and data.to.size > 0 and data.to > 0.1:
                    return data
                else:
                    return []
            
            
class DataPreprocessing(DataPreparing):

    def __init__(self,
                 IDs,DB,
                 trim = 1, #consider signals only up to maximum force (+10 timesteps)
                 smooth = 1,
                 window = 1,#window for gradient
                 max_gradient_order = 3
                 ):

        self.trim = trim
        self.smooth = smooth
        self.window = window
        self.max_gradient_order = max_gradient_order
        super().__init__(IDs,DB)
    def OutputPrepare(self,idNum,measNum,include_s):
        data = super().segment(idNum,measNum,include_s)
        try:
            F = data.F
            t = data.t
            s = data.s
            to = data.to
            Fo = data.Fo
        except:
            return [], [],[]
        
        to = round(to[0],2) 
        if idNum < 775:
            F,s,t = self.UpSampling(F,s,t)               
        idxo = np.searchsorted(t,to,'left') # find index of opening time
        y = np.zeros_like(F) #create label array: 1 at opening point index, else 0
        y[idxo] = 1
        return F,s,y
        
    
    def Trim(self,F,s,y):
        if self.trim:
            trim_to = np.argmax(F) + 10
            F = F[:trim_to]
            s = s[:trim_to]
            y = y[:trim_to] 
            return F, s, y
    def Smooth(self,F):
        if self.smooth:
            for i in range(len(F)-1):
                F[i+1] = max(F[i],F[i+1])
            return F
    def FeatureGenerator(self,F,s):
        window = self.window
        max_gradient_order = self.max_gradient_order
        x =  np.column_stack((F,s))
        for i in window:
            dF = F
            ds = s
            for order in range(max_gradient_order):
                dF = grad_bwd(dF,i)
                ds = grad_bwd(ds,i)
                x = np.column_stack((x,dF,ds))
                
        return Normalize(x)
    
    def padding_len(self,X):
        # Calculating the maximal sequence length in all samples
        Len = []
        for i in range(len(X)):
            Len.append(len(X[i]))
        pad_len = np.array(Len).max()
        return pad_len
    def Padding(self,X,Y):
        
        padlen = self.padding_len(X)
        padded_X = np.zeros((len(X),padlen,X[0].shape[-1]))
        padded_Y = np.zeros((len(Y),padlen,1))
        for i in range(len(X)):
            padded_X[i,...] = np.pad(X[i],((0,padlen-len(X[i])),(0,0)),'constant',constant_values = 0.)
            padded_Y[i,...] = np.pad(Y[i].reshape(-1,1),((0,padlen-len(Y[i])),(0,0)),'constant',constant_values = 0.)
        return padded_X, padded_Y
    
    def UpSampling(self,F,s,t):
        F_ = np.zeros((2*len(F)-1))
        s_ = np.zeros((2*len(s)-1))
        t_ = np.zeros((2*len(t)-1))
        for i in range(2*len(F)-1):
            if i%2 == 0:
                F_[i] = F[int(i/2)]
                s_[i] = s[int(i/2)]
                t_[i] = t[int(i/2)]
            else:
                F_[i] = (F[(i-1)//2]+F[(i+1)//2])/2
                s_[i] = (s[(i-1)//2]+s[(i+1)//2])/2
                t_[i] = (t[(i-1)//2]+t[(i+1)//2])/2
        return F_, s_, t_
    
    def data_split(self,idx, n_folds, groups,test_size,val_size):
        train=[]
        val=[]
        test=[]
        fold_count = np.arange(n_folds)
        #gkf = GroupKFold(n_folds)
        gkf = GroupShuffleSplit(n_splits=n_folds ,test_size=test_size, random_state=10)
        
        for train_val_split, test_split in gkf.split(idx, groups=groups):
            # generate indeces for train&val and test dataset
            idx_train_val = idx[train_val_split]
            idx_test = idx[test_split]
            
            #split train&val randomly into training and validation 
            gss = GroupShuffleSplit(n_splits=1,test_size=val_size/(1-test_size),random_state=10) #here test_size is actually "validation_size"
    
            for train_split, val_split in gss.split(idx_train_val,groups=groups[train_val_split]):         
                train.append(idx_train_val[train_split])
                val.append(idx_train_val[val_split])
                test.append(idx_test)
        
        Split_idx = pd.DataFrame([train,val,test],index=['Train','Validation','Test'],columns=fold_count )
        return Split_idx
    
    def GenerateSplitInfo(self,X,groups,n_folds,test_size,val_size,load_split,time):
        if load_split:
            #load splitting indeces
            split_idx = pd.read_hdf('../model_split_storage/data_'+ time +'/split_info_'+ time + '.h5')
             # number of folds for cross-validation
            n_folds =  split_idx.shape[-1]
        else:
            
            split_idx = self.data_split(np.arange(len(X)), n_folds, groups,test_size,val_size) #generate splitting indeces
            try:
                split_idx.to_hdf('../model_split_storage/data_'+ time +'/split_info_'+ time + '.h5','Data') #save splitting configuration
            except:
                os.mkdir('../model_split_storage/data_'+ time )
                split_idx.to_hdf('../model_split_storage/data_'+ time +'/split_info_'+ time + '.h5','Data') #save splitting configuration
            
        
        return split_idx
    def DataSplit(self,X,Y,groups,n_folds,test_size,val_size,load_split,time):
        split_idx = self.GenerateSplitInfo(X,
                                         groups = groups,
                                         test_size=test_size,
                                         val_size=val_size,
                                         n_folds=n_folds,
                                         load_split=load_split, # whether load or create split info
                                         time=time)
        x_test = [X[split_idx[i]['Test']] for i in range(n_folds)]
        y_test = [Y[split_idx[i]['Test']] for i in range(n_folds)]
        x_train = [X[split_idx[i]['Train']] for i in range(n_folds)]
        y_train = [Y[split_idx[i]['Train']] for i in range(n_folds)]
        x_val = [X[split_idx[i]['Validation']] for i in range(n_folds)]
        y_val = [Y[split_idx[i]['Validation']] for i in range(n_folds)]
      
        return x_train,y_train,x_val,y_val,x_test,y_test,split_idx


class BuildClassifier():
    def __init__(self,
                 NetType = None,
                 input_shape = None,
                 n_layers = None,
                 n_neurons = None,
                 n_lstmunit = None
                 ):

        self.NetType = NetType
        self.input_shape = input_shape
        self.layers = n_layers
        self.units = n_neurons
        self.lstmunit = n_lstmunit
        
    def create_net(self):
        if self.NetType == 'Dense':
            net = self.create_dense()
        elif self.NetType == 'LSTM':
            net = self.create_lstm()
        else:
            print('Error: choose a net type')
        return net
            
        
    def create_lstm(self):
        inputs = tf.keras.Input(shape = self.input_shape)
        x = tf.keras.layers.Masking(mask_value=0.)(inputs)
        x = tf.keras.layers.LSTM(self.lstmunit, activation='relu',return_sequences = True,name='lstm')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        for i in range(self.layers):
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.units, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.01)))(x)
            x = tf.keras.layers.BatchNormalization()(x)
        y_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(x)
        return tf.keras.Model(inputs=inputs,outputs=y_) 
    def create_dense(self ):
        inputs = tf.keras.Input(shape = self.input_shape)
        x = tf.keras.layers.Masking()(inputs)
        for i in range(self.layers):
            x = tf.keras.layers.Dense(self.units,activation='relu')(x)
        y_ = tf.keras.layers.Dense(1,activation='sigmoid')(x)
        return tf.keras.Model(inputs=inputs,outputs=y_) 
        
class Evaluation():
    def __init__(self):

        self.savefig = 1

    def evaluation_plot(self,Acc,error,plot_form,title,save,current_time):
    
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
    
    def evaluate_errorforce(self,y_,y,F,mask,err):
        
        Fo_pred = np.array([F[i][np.argmax(self.gate(y_[i],mask[i]))] for i in range(len(F))])
        Fos = np.array([F[i][np.argmax(y[i])] for i in range(len(F))])
        FoErr = np.abs(Fos.T-Fo_pred).T/np.array(Fos)
        foerr_orig = FoErr
        FoErr = np.where(FoErr>1,1,FoErr)
        FoErr = np.where(FoErr<0.0001,0,FoErr)
        Force_Err = np.zeros(len(err)+1)
        for i in range(len(err)-1): #np.linspace(0,1,6):
            low = np.where(FoErr>err[i])[0]
            up = np.where(FoErr<=err[i+1])[0]
            FoErr[np.intersect1d(low, up)]=err[i+1]
        unique, count = np.unique(FoErr, return_counts=True)
        for j in range(len(err)):
            try:
                Force_Err[j] = count[np.abs(unique-err[j])<=0.001]
                if err[j] in err[0:3]:
                    Force_Err[-1] =  Force_Err[-1] + count[np.abs(unique-err[j])<=0.001]
            except:
                continue
        return Force_Err*100/len(mask), foerr_orig
    
    def evaluate_errorstep(self,*ypre,to,mask,error_step):
        To = np.zeros(to.shape)
        Step_Err =  np.zeros(len(error_step)+1)
        for y_pre in ypre:
            for i in range(y_pre.shape[0]):
                # ignore padded part
                y_ = self.gate(y_pre[i],mask[i])
                To[i] = np.argmax(y_)
            diff_to = (np.array(To)-np.array(to))
            
            # error later than 3 steps are hold in +3
            diff_to = np.where(diff_to>=3,3,diff_to)
            # error earlier than 3 steps are hold in -3
            diff_to = np.where(diff_to<=-3,-3,diff_to)
            unique, count = np.unique(diff_to, return_counts=True)
            for j in range(len(error_step)):
                try:
                    Step_Err[j]=count[unique==error_step[j]]
                    if error_step[j] in error_step[1:-1]:
                        Step_Err[-1] =  Step_Err[-1] + count[unique==error_step[j]]
                except:
                    continue
                
                
        return Step_Err*100/len(mask), np.abs(diff_to)
    
        
    
    def gate(self,y_pred,mask):
    
        # Data was padded to a same length
        # this function is used to ignore padded part
        
        
        idx = np.where(mask == True)[0][-1]
        y_ = y_pred[:idx+1]
        return y_
        
if __name__ == "__main__":      
    dataArr = np.loadtxt(".\Database\OUTPUT_diagram_ll_clean.txt", skiprows=1) #contains raw measurement data
    resArr = np.loadtxt(".\Database\OUTPUT_tocke_ll_clean_2020.txt", skiprows=1) #contains determined opening point for each measurement
    infoArr = np.loadtxt(".\Database\OUTPUT_meritve_ll.txt", skiprows=1) #additional information
    # vibrArr = np.loadtxt("U:\Data\\vibration_filter.txt")
    
    
    IDs = int(dataArr[-1,0]) #Number of valve tests in database (up to 3 measurements per test)
    DB = [dataArr,resArr,infoArr] 
    LoadRawData = DataPreparing(IDs,DB)
    RawF = []        
    X = []
    Y = []
    label = []
    DataPreprocess = DataPreprocessing(IDs,DB,window=[1,2,3],max_gradient_order = 2)   
    id_from, id_to = loop_from_to(IDs, id_from=0, steps=0) #id_from=0 loops through all
    for i in range(id_from,id_to):
        for j in range(1,4): 
            RawData = LoadRawData.segment(i,j,include_s = True)
            
            F,s,y = DataPreprocess.OutputPrepare(i,j,include_s = True)
            if not any(F):
                continue
            f = copy.deepcopy(F)
            RawF.append(f)
            F = DataPreprocess.Smooth(F)
            F,s,y = DataPreprocess.Trim(F,s,y)
            x = DataPreprocess.FeatureGenerator(F,s)
            if x.any():
                label.append([i,j])
                X.append(x)
                Y.append(y)
            
    X, Y= DataPreprocess.Padding(X,Y)  
    
    idxos = np.array([np.argmax(y) for y in Y]) #array with indexes of opening point from database

########################################################################

#group measurements by testing ID (= particular valve)
    groups = np.array(label)[:,0] 
    n_folds = 5
    # get current time and date to use as file name
    now = datetime.now()
    current_time = now.strftime("%d_%m-%H_%M")
    
    x_train,y_train,x_val,y_val,x_test,y_test,split_idx = DataPreprocess.DataSplit(X,Y,
                                         groups = groups,
                                         n_folds=n_folds,
                                         test_size =0.3,
                                         val_size = 0.2,
                                         load_split=0, # whether load or create split info
                                         time=current_time) # can be save_time when load_split =1
    
    BuildNet = BuildClassifier(NetType = 'LSTM',
                               input_shape = x_train[0].shape[1:],
                               n_layers=3,n_neurons=100,
                               n_lstmunit = 20)
    
    es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience = 5,
                           restore_best_weights=True)
        
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    error_step = np.array([-3.,-2.,-1.,0.,1.,2.,3.]) #step errors are seperated as ('<-3','-2','-1','0','1','2','>3')
    error_force = np.array([0.,0.01,0.02,0.03,0.1,1.]) #force deviation are seperated as ('0%','1%','2%','3%','10%','>=100%')
    Step_Err= np.zeros((n_folds, len(error_step)+1))
    Force_Err = np.zeros((n_folds, len(error_force)+1))
    for i in range(n_folds):
        model = BuildNet.create_net()
        model.compile(loss=loss,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],optimizer='adam')
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
        
        # the number (_count) and variations (_unique) of force and step error in above defined range 
        ForceErr, foerr = Evaluation().evaluate_errorforce(y_=y_pred,y=y_test[i],F=np.array(RawF)[split_idx[i]['Test']],mask=Mask,err=error_force)
        Force_Err[i,:] = ForceErr
        StepErr,  steperr = Evaluation().evaluate_errorstep( y_pred,to=idxos[split_idx[i]['Test']],mask = Mask,error_step=error_step)
        Step_Err[i,:] = StepErr
        
        
    os.mkdir('./Results/results_' + current_time )
    plt.tight_layout()
    Evaluation().evaluation_plot(Acc=Step_Err,
                    error=error_step,
                    plot_form='step',
                    title = 'Classification accuracy in error step (reduced LSTM)',
                    save=True,
                    current_time = current_time)
    # plot evaluation of force deviation
    Evaluation().evaluation_plot(Acc=Force_Err,
                    error=error_force,
                    plot_form='force',
                    title = 'Deviation of force at predicted change point (reduced LSTM)',
                    save=True,
                    current_time = current_time)
    




