import numpy
import MetaTrader5 as mt5
import pandas as pd
import datetime
import time
import random
import json
import xgboost
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import matplotlib
path="C:/"



with open(f"{path}PARAM_BACKTEST.json", 'r') as openfile:
    json_PARAM_backtest = json.load(openfile)
    
with open(f"{path}PARAM_MODEL.json", 'r') as openfile:
    json_PARAM_MODEL = json.load(openfile)


METHOD=json_PARAM_backtest['METHOD']

DYNAMIC_PLOT=json_PARAM_backtest['DYNAMIC_PLOT']
METRIC=json_PARAM_backtest['METRIC']

SHIFT=json_PARAM_backtest['SHIFT']
N=json_PARAM_backtest['N']
ACCOUNT_SIZE=json_PARAM_backtest['ACCOUNT_SIZE']


A=json_PARAM_MODEL['ASSET']
TF=json_PARAM_MODEL['TF']
SIZE=json_PARAM_MODEL['SIZE']

n=json_PARAM_MODEL['n']
q=json_PARAM_MODEL['q']
h=json_PARAM_MODEL['h']
k=json_PARAM_MODEL['k']
p=json_PARAM_MODEL['p']
u=json_PARAM_MODEL['u'] 

alpahmax=json_PARAM_MODEL['alphamax']
betamax=json_PARAM_MODEL['betamax']
tauinvmax=json_PARAM_MODEL['tauinvmax']
zmax=json_PARAM_MODEL['zmax']

model_high_type=json_PARAM_MODEL['model_high_type']
model_low_type=json_PARAM_MODEL['model_low_type']
model_ohlc_type=json_PARAM_MODEL['model_ohlc_type']

STANDARDIZATION_high=json_PARAM_MODEL['STANDARDIZATION_high']
STANDARDIZATION_low=json_PARAM_MODEL['STANDARDIZATION_low']
STANDARDIZATION_ohlc=json_PARAM_MODEL['STANDARDIZATION_ohlc']

model_high_RE_INIT=json_PARAM_MODEL['model_high_RE_INIT']
model_low_RE_INIT=json_PARAM_MODEL['model_low_RE_INIT']
model_ohlc_RE_INIT=json_PARAM_MODEL['model_ohlc_RE_INIT']

print(json_PARAM_backtest)
print(json_PARAM_MODEL)


    
 

if model_high_type == 'xgboost':
    with open(f"{path}PARAM_XGBOOST_HIGH.json", 'r') as openfile:
        json_PARAM_XGBOOST_HIGH = json.load(openfile)
    booster_high=json_PARAM_XGBOOST_HIGH['booster_high']
    colsample_bylevel_high=json_PARAM_XGBOOST_HIGH['colsample_bylevel_high']
    colsample_bynode_high=json_PARAM_XGBOOST_HIGH['colsample_bynode_high']
    colsample_bytree_high=json_PARAM_XGBOOST_HIGH['colsample_bytree_high']
    device_high=json_PARAM_XGBOOST_HIGH['device_high']
    interaction_constraints_high=json_PARAM_XGBOOST_HIGH['interaction_constraints_high']
    learning_rate_high=json_PARAM_XGBOOST_HIGH['learning_rate_high']
    max_depth_high=json_PARAM_XGBOOST_HIGH['max_depth_high']
    min_child_weight_high=json_PARAM_XGBOOST_HIGH['min_child_weight_high']
    n_estimators_high=json_PARAM_XGBOOST_HIGH['n_estimators_high']
    n_jobs_high=json_PARAM_XGBOOST_HIGH['n_jobs_high']
    num_parallel_tree_high=json_PARAM_XGBOOST_HIGH['num_parallel_tree_high']
    tree_method_high=json_PARAM_XGBOOST_HIGH['tree_method_high']
    verbosity_high=json_PARAM_XGBOOST_HIGH['verbosity_high']
    
if model_low_type == 'xgboost':
    with open(f"{path}PARAM_XGBOOST_LOW.json", 'r') as openfile:
        json_PARAM_XGBOOST_LOW = json.load(openfile)   
    booster_low=json_PARAM_XGBOOST_LOW['booster_low']
    colsample_bylevel_low=json_PARAM_XGBOOST_LOW['colsample_bylevel_low']
    colsample_bynode_low=json_PARAM_XGBOOST_LOW['colsample_bynode_low']
    colsample_bytree_low=json_PARAM_XGBOOST_LOW['colsample_bytree_low']
    device_low=json_PARAM_XGBOOST_LOW['device_low']
    interaction_constraints_low=json_PARAM_XGBOOST_LOW['interaction_constraints_low']
    learning_rate_low=json_PARAM_XGBOOST_LOW['learning_rate_low']
    max_depth_low=json_PARAM_XGBOOST_LOW['max_depth_low']
    min_child_weight_low=json_PARAM_XGBOOST_LOW['min_child_weight_low']
    n_estimators_low=json_PARAM_XGBOOST_LOW['n_estimators_low']
    n_jobs_low=json_PARAM_XGBOOST_LOW['n_jobs_low']
    num_parallel_tree_low=json_PARAM_XGBOOST_LOW['num_parallel_tree_low']
    tree_method_low=json_PARAM_XGBOOST_LOW['tree_method_low']
    verbosity_low=json_PARAM_XGBOOST_LOW['verbosity_low']

if model_ohlc_type == 'xgboost':
    with open(f"{path}PARAM_XGBOOST_OHLC.json", 'r') as openfile:
        json_PARAM_XGBOOST_OHLC = json.load(openfile)   
    booster_ohlc=json_PARAM_XGBOOST_OHLC['booster_ohlc']
    colsample_bylevel_ohlc=json_PARAM_XGBOOST_OHLC['colsample_bylevel_ohlc']
    colsample_bynode_ohlc=json_PARAM_XGBOOST_OHLC['colsample_bynode_ohlc']
    colsample_bytree_ohlc=json_PARAM_XGBOOST_OHLC['colsample_bytree_ohlc']
    device_ohlc=json_PARAM_XGBOOST_OHLC['device_ohlc']
    interaction_constraints_ohlc=json_PARAM_XGBOOST_OHLC['interaction_constraints_ohlc']
    learning_rate_ohlc=json_PARAM_XGBOOST_OHLC['learning_rate_ohlc']
    max_depth_ohlc=json_PARAM_XGBOOST_OHLC['max_depth_ohlc']
    min_child_weight_ohlc=json_PARAM_XGBOOST_OHLC['min_child_weight_ohlc']
    n_estimators_ohlc=json_PARAM_XGBOOST_OHLC['n_estimators_ohlc']
    n_jobs_ohlc=json_PARAM_XGBOOST_OHLC['n_jobs_ohlc']
    num_parallel_tree_ohlc=json_PARAM_XGBOOST_OHLC['num_parallel_tree_ohlc']
    tree_method_ohlc=json_PARAM_XGBOOST_OHLC['tree_method_ohlc']
    verbosity_ohlc=json_PARAM_XGBOOST_OHLC['verbosity_ohlc']

if model_ohlc_type == 'keras':
    with open(f"{path}PARAM_KERAS_OHLC.json", 'r') as openfile:
        json_PARAM_KERAS_OHLC = json.load(openfile)   
    neurons_ohlc = json_PARAM_KERAS_OHLC['neurons_ohlc']
    initial_learning_rate_ohlc=json_PARAM_KERAS_OHLC['initial_learning_rate_ohlc']
    decay_steps_ohlc=json_PARAM_KERAS_OHLC['decay_steps_ohlc']
    decay_rate_ohlc=json_PARAM_KERAS_OHLC['decay_rate_ohlc']
    epochs_ohlc=json_PARAM_KERAS_OHLC['epochs_ohlc']
    verbosity_ohlc=json_PARAM_KERAS_OHLC['verbosity_ohlc']




def disconnect():
    mt5.shutdown()
def connect():
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()

connect()
trade_contract_size=float(mt5.symbol_info(A).trade_contract_size)
disconnect()

        
def close():
    positions_total=mt5.positions_total()
    if positions_total==0: 
        isOpen=0
    else :
        isOpen=1

    if isOpen != 0:
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        result=mt5.Close(A)
        print("ORDER CLOSED")
        isOpen=0
          
def get_values_open(S):
    rates = mt5.copy_rates_from_pos(A, mt5.TIMEFRAME_H1, 1, S)
    ticks_frame = pd.DataFrame(rates)
    PRICE_array=ticks_frame['open'].to_numpy()
    return PRICE_array       

def get_values_datas(N):
    rates = mt5.copy_rates_from_pos(A, mt5.TIMEFRAME_H1, 1, N)
    ticks_frame = pd.DataFrame(rates)
    PRICE_array=ticks_frame[['open','high','low']].to_numpy()
    return PRICE_array

def get_values_close(N):
    rates = mt5.copy_rates_from_pos(A, mt5.TIMEFRAME_H1, 1, N)
    ticks_frame = pd.DataFrame(rates)
    PRICE_array=ticks_frame['close'].to_numpy()
    return PRICE_array

def get_values_high(N):
    rates = mt5.copy_rates_from_pos(A, mt5.TIMEFRAME_H1, 1, N)
    ticks_frame = pd.DataFrame(rates)
    PRICE_array=ticks_frame['high'].to_numpy()
    return PRICE_array

def get_values_low(N):
    rates = mt5.copy_rates_from_pos(A, mt5.TIMEFRAME_H1, 1, N)
    ticks_frame = pd.DataFrame(rates)
    PRICE_array=ticks_frame['low'].to_numpy()
    return PRICE_array

def fftpred_(X):
    length=len(X)
    xtrain=numpy.array([[1]+list((numpy.fft.fft(numpy.array(X[i:i+q])))) for i in range(length-q)])
    ytrain=numpy.array([X[i+q] for i in range(length-q)])
    A=numpy.matmul(numpy.matmul(numpy.linalg.pinv(numpy.matmul(numpy.transpose(xtrain),xtrain)),numpy.transpose(xtrain)),ytrain)
    pred=numpy.matmul(A,numpy.array([1]+list((numpy.fft.fft(numpy.array(X[length-q:]))))))
    return pred,A

def returns_(X):
    length=len(X)
    R=numpy.array([numpy.log(X[k+1]/X[k]) for k in range(length-1)])

    return list(R)

def DATA_fit(Y):
    Amat=[]
    X=returns_(Y)
    AMP=max(abs(numpy.array(X)))
    j=len(X)
    for r in range(h):
        
        v,a=fftpred_(X[r:r+j])
        Amat.append(a)
        X.append(v)
    return Amat,AMP

def DATA_gen(Amat,y,AMP):
    R=[y]
    alpha=alpahmax*(2*numpy.random.random()-1)
    beta=alpahmax*(2*numpy.random.random()-1)
    tauinv=tauinvmax*(2*numpy.random.random()-1)
    Z=[AMP*zmax*(2*numpy.random.random()-1)*(alpha+beta*k/h)*numpy.exp(tauinv*k/h) for k in range(h)]
    L=[numpy.fft.ifft(Z[s]*numpy.linalg.pinv([Amat[s]])[1:])[-1] for s in range(p-1,-1,-1)]
    for l in numpy.transpose(numpy.array(L))[0]:
        R.append(R[0]/numpy.exp(-l))
    R.reverse()    
    return R


position=0

EQUITY=0
PL=0

STEPS=[]
PRICES=[]


PL_ARRAY=[]
EQUITY_ARRAY=[]
xdata_high=[]
ydata_high=[]

connect()
DATAhigh=get_values_high(N)
disconnect()

if METHOD=='FAST':
    print("###generate high DATA###")
    for i in range(N-u-1):
        xlinehigh=[]
        DATAlinehigh=DATAhigh[i:i+u+1]
        DATAvaluehigh=DATAhigh[i+u]
        Alochigh,AMPhigh=DATA_fit(DATAlinehigh)
        for j in range(k):
            try:
                xlinehigh.append(list(numpy.real(DATA_gen(Alochigh,DATAvaluehigh,AMPhigh))))
            except:
                xlinehigh.append([0 for l in range(p+1)])
        xdata_high.append(xlinehigh)
        ydata_high.append(DATAhigh[i+u+1])

    xdata_high=numpy.array(xdata_high)
    ydata_high=numpy.array(ydata_high)
####

xdata_low=[]
ydata_low=[]

connect()
DATAlow=get_values_low(N)
disconnect()

if METHOD=='FAST':  
    print("###generate low DATA###")
    for i in range(N-u-1):

        xlinelow=[]
        DATAlinelow=DATAlow[i:i+u+1]
        DATAvaluelow=DATAlow[i+u]
        Aloclow,AMPlow=DATA_fit(DATAlinelow)
        for l in range(k):
            try:
                xlinelow.append(list(numpy.real(DATA_gen(Aloclow,DATAvaluelow,AMPlow))))
            except:
                xlinelow.append([0 for l in range(p+1)])
        xdata_low.append(xlinelow)
        ydata_low.append(DATAlow[i+u+1])



    xdata_low=numpy.array(xdata_low)
    ydata_low=numpy.array(ydata_low)
####

connect()
X=get_values_datas(N)
Y=get_values_close(N)
disconnect()

connect()
O=get_values_open(N)
disconnect()

if model_high_RE_INIT == False :
    if model_high_type == 'xgboost':
        modelhigh=xgboost.XGBRegressor(    
            booster=booster_high,
            colsample_bylevel=colsample_bylevel_high,
            colsample_bynode=colsample_bynode_high,
            colsample_bytree=colsample_bytree_high,
            device=device_high,
            interaction_constraints=interaction_constraints_high,
            learning_rate=learning_rate_high,
            max_depth=max_depth_high,
            min_child_weight=min_child_weight_high,
            n_estimators=n_estimators_high,
            n_jobs=n_jobs_high,
            num_parallel_tree=num_parallel_tree_high,
            tree_method=tree_method_high,
            verbosity=verbosity_high
            )
if model_low_RE_INIT == False :
    if model_low_type == 'xgboost':
        modellow=xgboost.XGBRegressor(    
            booster=booster_low,
            colsample_bylevel=colsample_bylevel_low,
            colsample_bynode=colsample_bynode_low,
            colsample_bytree=colsample_bytree_low,
            device=device_low,
            interaction_constraints=interaction_constraints_low,
            learning_rate=learning_rate_low,
            max_depth=max_depth_low,
            min_child_weight=min_child_weight_low,
            n_estimators=n_estimators_low,
            n_jobs=n_jobs_low,
            num_parallel_tree=num_parallel_tree_low,
            tree_method=tree_method_low,
            verbosity=verbosity_low
            )
if model_ohlc_RE_INIT == False :
    if model_ohlc_type == 'xgboost':
        modelohlc=xgboost.XGBRegressor(    
            booster=booster_ohlc,
            colsample_bylevel=colsample_bylevel_ohlc,
            colsample_bynode=colsample_bynode_ohlc,
            colsample_bytree=colsample_bytree_ohlc,
            device=device_ohlc,
            interaction_constraints=interaction_constraints_ohlc,
            learning_rate=learning_rate_ohlc,
            max_depth=max_depth_ohlc,
            min_child_weight=min_child_weight_ohlc,
            n_estimators=n_estimators_ohlc,
            n_jobs=n_jobs_ohlc,
            num_parallel_tree=num_parallel_tree_ohlc,
            tree_method=tree_method_ohlc,
            verbosity=verbosity_ohlc
            )
    if model_ohlc_type == 'keras':
        neurons=neurons_ohlc
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate_ohlc,
            decay_steps=decay_steps_ohlc,
            decay_rate=decay_rate_ohlc)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        modelohlc=Sequential()
        modelohlc.add(Dense(units=neurons,activation='tanh'))
        modelohlc.add(Dense(units=neurons,activation='linear'))
        modelohlc.add(Dense(units=neurons,activation='tanh'))
        modelohlc.add(Dense(units=neurons,activation='linear'))
        modelohlc.add(Dense(units=neurons,activation='tanh'))
        modelohlc.add(Dense(units=neurons,activation='linear'))
        modelohlc.add(Dense(units=neurons,activation='tanh'))
        modelohlc.add(Dense(units=neurons,activation='linear'))
        modelohlc.add(Dense(units=neurons,activation='tanh'))
        modelohlc.add(Dense(units=1,activation='linear'))
        modelohlc.compile(optimizer=opt,loss='mean_squared_error')


STEPS.append(O[0+u+1])

for j in range(N-n-u-1):
    print("--- test at PERIOD :", j, " ---")
    
    if METHOD == 'FAST':
        
        loc_xdata_high=xdata_high[j:j+n]
        loc_ydata_high=ydata_high[j:j+n]
        loc_xdata_low=xdata_low[j:j+n]
        loc_ydata_low=ydata_low[j:j+n]
        
    if METHOD == 'SLOW':
        
        loc_xdata_high=[]
        loc_ydata_high=[]
        loc_xdata_low=[]
        loc_ydata_low=[]
        
        for i in range(n-u-1):

            xlinehigh=[]
            DATAlinehigh=DATAhigh[i:i+u+1]
            DATAvaluehigh=DATAhigh[i+u]
            Alochigh,AMPhigh=DATA_fit(DATAlinehigh)
            for l in range(k):
                try:
                    xlinehigh.append(list(numpy.real(DATA_gen(Alochigh,DATAvaluehigh,AMPhigh))))
                except:
                    xlinehigh.append([0 for l in range(p+1)])
            loc_xdata_high.append(xlinehigh)
            loc_ydata_high.append(DATAhigh[i+u+1])
        loc_xdata_high=numpy.array(loc_xdata_high)
        loc_ydata_high=numpy.array(loc_ydata_high)
        
        for i in range(n-u-1):

            xlinelow=[]
            DATAlinelow=DATAlow[i:i+u+1]
            DATAvaluelow=DATAlow[i+u]
            Aloclow,AMPlow=DATA_fit(DATAlinelow)
            for l in range(k):
                #print(j)
                try:
                    xlinelow.append(list(numpy.real(DATA_gen(Aloclow,DATAvaluelow,AMPlow))))
                except:
                    xlinelow.append([0 for l in range(p+1)])
            loc_xdata_low.append(xlinelow)
            loc_ydata_low .append(DATAlow[i+u+1])
        loc_xdata_low=numpy.array(loc_xdata_low)
        loc_ydata_low=numpy.array(loc_ydata_low)
    
    loc_X=X[j:j+n]
    loc_Y=Y[j:j+n]
    
    if STANDARDIZATION_high == True :
        xmeanhigh=loc_xdata_high.mean()
        xstdhigh=loc_xdata_high.std()
        
        ymeanhigh=loc_ydata_high.mean()
        ystdhigh=loc_ydata_high.std()
        
        loc_xdata_high=(loc_xdata_high-xmeanhigh)/xstdhigh
        
        loc_ydata_high=(loc_ydata_high-ymeanhigh)/ystdhigh
    
    s1h,s2h,s3h=loc_xdata_high.shape
    if model_high_RE_INIT == True :
        if model_high_type == 'xgboost':
            modelhigh=xgboost.XGBRegressor(    
                booster=booster_high,
                colsample_bylevel=colsample_bylevel_high,
                colsample_bynode=colsample_bynode_high,
                colsample_bytree=colsample_bytree_high,
                device=device_high,
                interaction_constraints=interaction_constraints_high,
                learning_rate=learning_rate_high,
                max_depth=max_depth_high,
                min_child_weight=min_child_weight_high,
                n_estimators=n_estimators_high,
                n_jobs=n_jobs_high,
                num_parallel_tree=num_parallel_tree_high,
                tree_method=tree_method_high,
                verbosity=verbosity_high
                )
    
    modelhigh.fit(numpy.array(loc_xdata_high).reshape(s1h,s2h*s3h),numpy.array(loc_ydata_high))
    
    if STANDARDIZATION_low == True :
        xmeanlow=loc_xdata_low.mean()
        xstdlow=loc_xdata_low.std()
        
        ymeanlow=loc_ydata_low.mean()
        ystdlow=loc_ydata_low.std()
        
        loc_xdata_low=(loc_xdata_low-xmeanlow)/xstdlow
        
        loc_ydata_low=(loc_ydata_low-ymeanlow)/ystdlow
    
    s1l,s2l,s3l=loc_xdata_low.shape
    
    if model_low_RE_INIT == True :
        if model_low_type == 'xgboost':
            modellow=xgboost.XGBRegressor(            
                        booster=booster_low,
                        colsample_bylevel=colsample_bylevel_low,
                        colsample_bynode=colsample_bynode_low,
                        colsample_bytree=colsample_bytree_low,
                        device=device_low,
                        interaction_constraints=interaction_constraints_low,
                        learning_rate=learning_rate_low,
                        max_depth=max_depth_low,
                        min_child_weight=min_child_weight_low,
                        n_estimators=n_estimators_low,
                        n_jobs=n_jobs_low,
                        num_parallel_tree=num_parallel_tree_low,
                        tree_method=tree_method_low,
                        verbosity=verbosity_low
                        )
    
    modellow.fit(numpy.array(loc_xdata_low).reshape(s1l,s2l*s3l),numpy.array(loc_ydata_low))
    
    if STANDARDIZATION_ohlc == True :
        Xmean=loc_X.mean()
        Xstd=loc_X.std()
                
        Ymean=loc_Y.mean()
        Ystd=loc_Y.std()    
        loc_X=(loc_X-Xmean)/Xstd
        loc_Y=(loc_Y-Ymean)/Ystd   
        
        
    if model_ohlc_RE_INIT == True :
        if model_ohlc_type == 'xgboost':
            modelohlc=xgboost.XGBRegressor(    
                booster=booster_ohlc,
                colsample_bylevel=colsample_bylevel_ohlc,
                colsample_bynode=colsample_bynode_ohlc,
                colsample_bytree=colsample_bytree_ohlc,
                device=device_ohlc,
                interaction_constraints=interaction_constraints_ohlc,
                learning_rate=learning_rate_ohlc,
                max_depth=max_depth_ohlc,
                min_child_weight=min_child_weight_ohlc,
                n_estimators=n_estimators_ohlc,
                n_jobs=n_jobs_ohlc,
                num_parallel_tree=num_parallel_tree_ohlc,
                tree_method=tree_method_ohlc,
                verbosity=verbosity_ohlc
                )
            
    if model_ohlc_RE_INIT == True :
        if model_ohlc_type == 'keras':            
            neurons=neurons_ohlc
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate_ohlc,
                decay_steps=decay_steps_ohlc,
                decay_rate=decay_rate_ohlc)
            opt = keras.optimizers.Adam(learning_rate=lr_schedule)
            modelohlc=Sequential()
            modelohlc.add(Dense(units=neurons,activation='tanh'))
            modelohlc.add(Dense(units=neurons,activation='linear'))
            modelohlc.add(Dense(units=neurons,activation='tanh'))
            modelohlc.add(Dense(units=neurons,activation='linear'))
            modelohlc.add(Dense(units=neurons,activation='tanh'))
            modelohlc.add(Dense(units=neurons,activation='linear'))
            modelohlc.add(Dense(units=neurons,activation='tanh'))
            modelohlc.add(Dense(units=neurons,activation='linear'))
            modelohlc.add(Dense(units=neurons,activation='tanh'))
            modelohlc.add(Dense(units=1,activation='linear'))
            modelohlc.compile(optimizer=opt,loss='mean_squared_error')
    
            
    if model_ohlc_type == 'xgboost':
        modelohlc.fit(loc_X,loc_Y)
    if model_ohlc_type == 'keras':         
        modelohlc.fit(loc_X,loc_Y,batch_size=X.shape[0],epochs=epochs_ohlc,verbose=verbosity_ohlc)
    
    
    if METHOD == 'FAST' :
        Phighdata=xdata_high[j+n]
    if METHOD == 'SLOW' :
        Phighdata=[]
        xlinehigh=[]
        DATAlinehigh=DATAhigh[i:i+u+1]
        DATAvaluehigh=DATAhigh[i+u]
        Alochigh,AMPhigh=DATA_fit(DATAlinehigh)
        for l in range(k):
            try:
                xlinehigh.append(list(numpy.real(DATA_gen(Alochigh,DATAvaluehigh,AMPhigh))))
            except:
                xlinehigh.append([0 for l in range(p+1)])
        Phighdata.append(xlinehigh)
        
        Phighdata=numpy.array(Phighdata)
        
    if STANDARDIZATION_high == True : 
        Phighdata=(Phighdata-xmeanhigh)/xstdhigh
    
    if model_high_type == 'xgboost' :
        nextHIGH=modelhigh.predict(Phighdata.reshape(1,s2h*s3h))[0]
    
    if STANDARDIZATION_high == True :
        nextHIGH=nextHIGH*ystdhigh+ymeanhigh
        
    if METHOD == 'FAST' :
        Plowdata=xdata_high[j+n]
        
    if METHOD == 'SLOW' :
        Plowdata=[]
        xlinelow=[]
        DATAlinelow=DATAlow[i:i+u+1]
        DATAvaluelow=DATAlow[i+u]
        Aloclow,AMPlow=DATA_fit(DATAlinelow)
        for l in range(k):
            try:
                xlinelow.append(list(numpy.real(DATA_gen(Aloclow,DATAvaluelow,AMPlow))))
            except:
                xlinelow.append([0 for l in range(p+1)])
        Plowdata.append(xlinelow)
        
        Plowdata=numpy.array(Plowdata)
        
    if STANDARDIZATION_low == True : 
        Plowdata=(Plowdata-xmeanlow)/xstdlow
    
    if model_low_type == 'xgboost' :
        nextLOW=modellow.predict(Plowdata.reshape(1,s2l*s3l))[0]
    
    if STANDARDIZATION_low == True : 
        nextLOW=nextLOW*ystdlow+ymeanlow
    
    OPEN=O[j+u+1]
    
    OHL=numpy.array([[OPEN,nextHIGH,nextLOW]])
    
    if STANDARDIZATION_ohlc == True :
        OHL=(OHL-Xmean)/Xstd
        
    if model_ohlc_type == 'keras' :
        CLOSE=modelohlc.predict(OHL,verbose=0)[0][0]
    if model_ohlc_type == 'xgboost' :
        CLOSE=modelohlc.predict(OHL)[0]
    
    if STANDARDIZATION_ohlc == True :
        
        CLOSE=CLOSE*Ystd+Ymean
    
    if CLOSE>OPEN:
        Signal=1 
    if CLOSE<OPEN: 
        Signal=-1 
    
    STEPS.append(OPEN)    

    EQUITY+=position*SIZE*trade_contract_size(STEPS[-1]-STEPS[-2])

    if Signal != position:
        PRICES.append(OPEN)
        print(f"TRADE OPEN @ {OPEN}")
        if len(PRICES)>1:
            PL+=position*SIZE*trade_contract_size(PRICES[-1] - PRICES[-2])
        position=Signal    
    
            
    PL_ARRAY.append(PL)
    EQUITY_ARRAY.append(EQUITY)
    if DYNAMIC_PLOT == True :
        if METRIC == 'PL' :
            matplotlib.pyplot.plot(PL_ARRAY)
            matplotlib.pyplot.show()
        if METRIC == 'EQUITY' :
            matplotlib.pyplot.plot(EQUITY_ARRAY)
            matplotlib.pyplot.show()
        

def discrete_returns_(X):
    length=len(X)
    R=numpy.array([X[k+1]/X[k]-1 for k in range(length-1)])
    return R

def cov(X,Y):
    x=X.mean()
    y=Y.mean()
    return numpy.array([(X[k]-x)*(Y[k]-y) for k in range(len(X))]).mean()

PL_ARRAY=numpy.array(PL_ARRAY)
EQUITY_ARRAY=numpy.array(EQUITY_ARRAY)

VASSET=O[:N-n-u-1]
VPL_ARRAY=PL_ARRAY+ACCOUNT_SIZE
VEQUITY_ARRAY=EQUITY_ARRAY+ACCOUNT_SIZE

matplotlib.pyplot.plot(numpy.transpose(numpy.array([VASSET,VEQUITY_ARRAY])))
matplotlib.pyplot.show()
matplotlib.pyplot.plot(numpy.transpose(numpy.array([VPL_ARRAY,VEQUITY_ARRAY])))
matplotlib.pyplot.show()

RPL_ARRAY=discrete_returns_(VPL_ARRAY)
REQUITY_ARRAY=discrete_returns_(VEQUITY_ARRAY)
RASSET=discrete_returns_(VASSET)

sharpePL=RPL_ARRAY.mean()/RPL_ARRAY.std()
sharpeEQUITY=REQUITY_ARRAY.mean()/REQUITY_ARRAY.std()
sharpeASSET=RASSET.mean()/RASSET.std()

print("###SHARPE RATIOs###")
print(f"sharpe PL : {sharpePL}")
print(f"sharpe EQUITY : {sharpeEQUITY}")
print(f"sharpe ASSET : {sharpeASSET}")

betaPL=cov(RPL_ARRAY,RASSET)/cov(RASSET,RASSET)
alphaPL=RPL_ARRAY.mean()-betaPL*RASSET.mean()
matplotlib.pyplot.scatter(RASSET,RPL_ARRAY)
matplotlib.pyplot.show()


betaEQUITY=cov(REQUITY_ARRAY,RASSET)/cov(RASSET,RASSET)
alphaEQUITY=REQUITY_ARRAY.mean()-betaEQUITY*RASSET.mean()
matplotlib.pyplot.scatter(RASSET,REQUITY_ARRAY)
matplotlib.pyplot.show()

print("###PRICES###")
print(f"beta PL : {betaPL}")
print(f"alpha PL : {alphaPL}")
print(f"beta EQUITY : {betaEQUITY}")
print(f"alpha EQUITY : {alphaEQUITY}")