import json
import MetaTrader5 as mt5
#path to write json files 
path="C:/"
###
#Backtest generic parameters

#Method for backtesting
METHOD='FAST'
#Plot at each steps
DYNAMIC_PLOT=True
#Metric Displayed
METRIC='EQUITY'

#Shift in Data : int SHIFT, no conditions
SHIFT=0
#Number of Data Points for Backtest : int N, conditions N>n
N=500 
#Account Size
ACCOUNT_SIZE=100000

#Model generic parameters

###
#Asset Name in MT5
A='BTCUSD'
#TIMEFRAME, MT5 TIMEFRAME
TF=mt5.TIMEFRAME_H1 
#SIZE of Positions, float SIZE
SIZE=1.0 






###
#Number of Data Points : int n, no conditions
n=100
#AR Parameter : int q, conditions q < n 
q=5
#Number of Data Points used for Data Generation : int h, no conditions
h=10
#Number of Paths to be generated : int k, no conditions
k=25
#Length of Paths to be generated :  int p, conditions p < h 
p=10
#Number of Data Points for the ARML model : int u, conditions u > q and u < n
u=20 

#Generative Parameters

###
#Linear Alpha Max Return : float alphamax
alphamax=1 
#Linear Beta Max Return : float betamax
betamax=1
#K Exponential factor : float tauinvmax
tauinvmax=1
#Amplitude Max : float zmax
zmax=1.5


#Models
#
#Library used for Highs prediction, xgboost 
model_high_type='xgboost'
STANDARDIZATION_high = False
model_high_RE_INIT = False
#Library used for Lows prediction, xgboost 
model_low_type='xgboost'
STANDARDIZATION_low = False
model_low_RE_INIT = False
#Library used for Close prediction (main model), xgboost or keras
model_ohlc_type='xgboost'
STANDARDIZATION_ohlc = False
model_ohlc_RE_INIT = False

#Tuning of the models 
if model_high_type == 'xgboost':
    booster_high=None
    colsample_bylevel_high=None
    colsample_bynode_high=None
    colsample_bytree_high=None
    device_high=None
    interaction_constraints_high=None
    learning_rate_high=None
    max_depth_high=None
    min_child_weight_high=None
    n_estimators_high=None
    n_jobs_high=None
    num_parallel_tree_high=None
    tree_method_high=None
    verbosity_high=None
    
if model_low_type == 'xgboost':
    booster_low=None
    colsample_bylevel_low=None
    colsample_bynode_low=None
    colsample_bytree_low=None
    device_low=None
    interaction_constraints_low=None
    learning_rate_low=None
    max_depth_low=None
    min_child_weight_low=None
    n_estimators_low=None
    n_jobs_low=None
    num_parallel_tree_low=None
    tree_method_low=None
    verbosity_low=None
    
if model_ohlc_type== 'xgboost':
    booster_ohlc=None
    colsample_bylevel_ohlc=None
    colsample_bynode_ohlc=None
    colsample_bytree_ohlc=None
    device_ohlc=None
    interaction_constraints_ohlc=None
    learning_rate_ohlc=None
    max_depth_ohlc=None
    min_child_weight_ohlc=None
    n_estimators_ohlc=None
    n_jobs_ohlc=None
    num_parallel_tree_ohlc=None
    tree_method_ohlc=None
    verbosity_ohlc=None    
    
if model_ohlc_type== 'keras':
    neurons_ohlc = 60
    initial_learning_rate_ohlc=0.01
    decay_steps_ohlc=10
    decay_rate_ohlc=0.9
    epochs_ohlc=100
    verbosity_ohlc=0
    

        
###
#Write to json
param_backtest = {
      "METHOD":METHOD,
      "DYNAMIC_PLOT":DYNAMIC_PLOT,
      "METRIC":METRIC,
      "SHIFT":SHIFT,
      "N":N,
      "ACCOUNT_SIZE":ACCOUNT_SIZE
}

with open(f"{path}PARAM_BACKTEST.json", "w") as outfile:
    json.dump(param_backtest, outfile)
        
param_model = {
    "ASSET": A,
    "TF":TF,
    "SIZE":SIZE,
    "n": n,
    "q": q,
    "h": h,
    "k": k,
    "p": p,
    "u": u,
    "alphamax": alphamax,
    "betamax": betamax,
    "tauinvmax": tauinvmax,
    "zmax": zmax,
    "model_high_type":model_high_type,
    "model_low_type":model_low_type,
    "model_ohlc_type":model_ohlc_type,
    "STANDARDIZATION_high":STANDARDIZATION_high,
    "STANDARDIZATION_low":STANDARDIZATION_low,
    "STANDARDIZATION_ohlc":STANDARDIZATION_ohlc,
    "model_high_RE_INIT":model_high_RE_INIT,
    "model_low_RE_INIT":model_low_RE_INIT,
    "model_ohlc_RE_INIT":model_ohlc_RE_INIT,
}
 
with open(f"{path}PARAM_MODEL.json", "w") as outfile:
    json.dump(param_model, outfile)
    
    
if model_high_type == 'xgboost':
    param_xgboost_high = {
        "booster_high":booster_high,
        "colsample_bylevel_high":colsample_bylevel_high,
        "colsample_bynode_high":colsample_bynode_high, 
        "colsample_bytree_high":colsample_bytree_high, 
        "device_high":device_high,
        "interaction_constraints_high":interaction_constraints_high,
        "learning_rate_high":learning_rate_high, 
        "max_depth_high":max_depth_high,
        "min_child_weight_high":min_child_weight_high,
        "n_estimators_high":n_estimators_high, 
        "n_jobs_high":n_jobs_high, 
        "num_parallel_tree_high":num_parallel_tree_high,
        "tree_method_high":tree_method_high, 
        "verbosity_high":verbosity_high
        }
    
    with open(f"{path}PARAM_XGBOOST_HIGH.json", "w") as outfile:
        json.dump(param_xgboost_high, outfile)    
        
if model_low_type == 'xgboost':
    param_xgboost_low= {
        "booster_low":booster_low,
        "colsample_bylevel_low":colsample_bylevel_low,
        "colsample_bynode_low":colsample_bynode_low, 
        "colsample_bytree_low":colsample_bytree_low, 
        "device_low":device_low,
        "interaction_constraints_low":interaction_constraints_low,
        "learning_rate_low":learning_rate_low, 
        "max_depth_low":max_depth_low,
        "min_child_weight_low":min_child_weight_low,
        "n_estimators_low":n_estimators_low, 
        "n_jobs_low":n_jobs_low, 
        "num_parallel_tree_low":num_parallel_tree_low,
        "tree_method_low":tree_method_low, 
        "verbosity_low":verbosity_low
        }
    
    with open(f"{path}PARAM_XGBOOST_LOW.json", "w") as outfile:
        json.dump(param_xgboost_low, outfile)     
        
if model_ohlc_type == 'xgboost':
    param_xgboost_ohlc= {
        "booster_ohlc":booster_ohlc,
        "colsample_bylevel_ohlc":colsample_bylevel_ohlc,
        "colsample_bynode_ohlc":colsample_bynode_ohlc, 
        "colsample_bytree_ohlc":colsample_bytree_ohlc, 
        "device_ohlc":device_ohlc,
        "interaction_constraints_ohlc":interaction_constraints_ohlc,
        "learning_rate_ohlc":learning_rate_ohlc, 
        "max_depth_ohlc":max_depth_ohlc,
        "min_child_weight_ohlc":min_child_weight_ohlc,
        "n_estimators_ohlc":n_estimators_ohlc, 
        "n_jobs_ohlc":n_jobs_ohlc, 
        "num_parallel_tree_ohlc":num_parallel_tree_ohlc,
        "tree_method_ohlc":tree_method_ohlc, 
        "verbosity_ohlc":verbosity_ohlc
        }
    
    with open(f"{path}PARAM_XGBOOST_OHLC.json", "w") as outfile:
        json.dump(param_xgboost_ohlc, outfile)         

if model_ohlc_type == 'keras':
    param_keras_ohlc= {
        "neurons_ohlc":neurons_ohlc,
        "initial_learning_rate_ohlc":initial_learning_rate_ohlc,
        "decay_steps_ohlc":decay_steps_ohlc,
        "decay_rate_ohlc":decay_rate_ohlc,
        "epochs_ohlc":epochs_ohlc,
        "verbosity_ohlc":verbosity_ohlc
        }
    with open(f"{path}PARAM_KERAS_OHLC.json", "w") as outfile:
        json.dump(param_keras_ohlc, outfile) 