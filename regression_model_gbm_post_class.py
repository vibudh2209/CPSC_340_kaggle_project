import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import time, random
import multiprocessing
from multiprocessing import Pool
from contextlib import closing
from functools import partial
from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_curve,auc,mean_squared_error
from scipy.stats import mode
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n_feat','--nf',required=True)
parser.add_argument('-lr','--lr',required=True)
parser.add_argument('-md','--md',required=True)
io_args = parser.parse_args()
nf = int(io_args.nf)
lr = float(io_args.lr)
md = int(io_args.md)

def rf_classifier(feat):
    t=time.time()
    max_features,max_depth,learning_rate,n_estimators = feat
    X_train = pd.read_csv('../X_train_after_classification.csv',sep=',',header=0).values
    y_train = pd.read_csv('../y_train_after_classification.csv',sep=',',header=None).values
    y_train = np.log(1+y_train)
    X_pos = np.array([X_train[i,:] for i in range(X_train.shape[0]) if (y_train[i]>0)])
    y_pos = y_train[y_train>0]
    X_neg = np.array([X_train[i,:] for i in range(X_train.shape[0]) if (y_train[i]==0)])
    y_neg = y_train[y_train==0]
    over_samp_ct = y_neg.shape[0]
    _,d = X_pos.shape
    n_pos_train = X_pos.shape[0]
    n_neg_train = X_neg.shape[0]
    Over_under_X_train = np.zeros([over_samp_ct*2,d])
    Over_under_y_train = np.zeros([over_samp_ct*2,])
    for i in range(over_samp_ct):
        idx_pos = random.randint(0,n_pos_train-1)
        idx_neg = random.randint(0,n_neg_train-1)
        Over_under_X_train[2*i,:] = X_pos[idx_pos,:]
        Over_under_y_train[2*i] = y_pos[idx_pos]
        Over_under_X_train[2*i+1,:] = X_neg[idx_neg,:]
        Over_under_y_train[2*i+1] = y_neg[idx_neg]
    rf = GradientBoostingRegressor(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,learning_rate=learning_rate).fit(Over_under_X_train,Over_under_y_train)
    X_pos = []
    y_pos = []
    X_neg = []
    y_neg = []
    Over_under_X_train = []
    Over_under_y_train = []
    X_valid = pd.read_csv('../X_valid_after_classification.csv',sep=',',header=0).values
    y_valid = pd.read_csv('../y_valid_after_classification.csv',sep=',',header=None).values
    y_valid = np.log(1+y_valid)
    train_pred = rf.score(X_train,y_train)
    valid_pred = rf.score(X_valid,y_valid)
    
    print(time.time()-t)
    y_pred_train = rf.predict(X_train)
    y_pred_valid = rf.predict(X_valid)
    
    ctf = np.linspace(0,9.21,50)
    
    for i in ctf:
        y_pred_train[y_pred_train<i] = 0
        y_pred_valid[y_pred_valid<i] = 0
        with open('../hyperparameter_file_gbm_regression_post_class.csv','a') as ref:
            ref.write(str(i)+','+str(max_features)+','+str(max_depth)+','+str(learning_rate)+','+str(n_estimators)+','+str(train_pred)+','+str(valid_pred)+','+str(mean_squared_error(y_train,y_pred_train))+','+str(mean_squared_error(y_valid,y_pred_valid))+'\n')


hy_params = []
for n_estimators in [100,200,300,400,500,600,700,800,900,1000]:
    hy_params.append([int(nf),md,lr,n_estimators])

t=time.time()
with closing(Pool(10)) as pool:
    pool.map(rf_classifier,hy_params)
print(time.time()-t)

