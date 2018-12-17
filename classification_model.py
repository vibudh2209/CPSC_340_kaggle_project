import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time, random
import multiprocessing
from multiprocessing import Pool
from contextlib import closing
from functools import partial
from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_curve,auc,precision_recall_curve
from scipy.stats import mode
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n_feat','--nf',required=True)
io_args = parser.parse_args()
nf = int(io_args.nf)

t=time.time()
X_valid = pd.read_csv('../X_valid.csv',sep=',',header=0).values
y_valid = pd.read_csv('../y_valid.csv',sep=',',header=None).values
X_train = pd.read_csv('../X_train.csv',sep=',',header=0).values
y_train = pd.read_csv('../y_train.csv',sep=',',header=0).values
print(time.time()-t)

def rf_classifier(feat):
    t=time.time()
    max_features,n_estimators = feat
    X_train = pd.read_csv('../X_train.csv',sep=',',header=0).values
    y_train = pd.read_csv('../y_train.csv',sep=',',header=0).values
    X_pos = np.array([X_train[i,:] for i in range(X_train.shape[0]) if (y_train[i]==1)])
    y_pos = y_train[y_train==1]
    X_neg = np.array([X_train[i,:] for i in range(X_train.shape[0]) if (y_train[i]==0)])
    y_neg = y_train[y_train==0]
    over_samp_ct = 100000
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
    rf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features).fit(Over_under_X_train,Over_under_y_train)
    X_pos = []
    y_pos = []
    X_neg = []
    y_neg = []
    Over_under_X_train = []
    Over_under_y_train = []
    X_valid = pd.read_csv('../X_valid.csv',sep=',',header=0).values
    y_valid = pd.read_csv('../y_valid.csv',sep=',',header=None).values
    #train_pred = rf.predict(X_train)
    valid_pred = rf.predict_proba(X_valid)[:,1]
    X_train = []
    X_valid = []
    #fpr_tr, tpr_tr, thresh_tr = roc_curve(y_train, train_pred)
    fpr_vl, tpr_vl, thresh_vl = roc_curve(y_valid, valid_pred)
    pr,rl,tr = precision_recall_curve(y_valid,valid_pred)
    #auc_tr = auc(fpr_tr,tpr_tr)
    auc_vl = auc(fpr_vl,tpr_vl)
    pr_vl = pr[np.where(rl>0.9)[0][-1]]

    print(time.time()-t)
    with open('../hyperparameter_file.csv','a') as ref:
        ref.write(str(max_features)+','+str(n_estimators)+','+str(auc_vl)+','+str(pr_vl)+'\n')


hy_params = []
for n_estimators in [100,200,300,400,500,600,700,800,900,1000]:
    hy_params.append([int(nf),n_estimators])

t=time.time()
with closing(Pool(10)) as pool:
    pool.map(rf_classifier,hy_params)
print(time.time()-t)
