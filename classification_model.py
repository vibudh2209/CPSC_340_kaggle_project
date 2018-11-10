import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time, random
import multiprocessing
from multiprocessing import Pool
from contextlib import closing
from functools import partial
from sklearn.metrics import precision_score,recall_score,accuracy_score
from scipy.stats import mode

t=time.time()
X_valid = pd.read_csv('X_valid.csv',sep=',',header=0).values
y_valid = pd.read_csv('y_valid.csv',sep=',',header=None).values
X_train = pd.read_csv('X_train.csv',sep=',',header=0).values
y_train = pd.read_csv('y_train.csv',sep=',',header=0).values
print(time.time()-t)

def rf_classifier(feat):
    t=time.time()
    max_features,n_estimators = feat
    X_train = pd.read_csv('X_train.csv',sep=',',header=0).values
    y_train = pd.read_csv('y_train.csv',sep=',',header=0).values
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
    rf = RandomForestClassifier(n_estimators=n_estimators).fit(Over_under_X_train,Over_under_y_train)
    X_pos = []
    y_pos = []
    X_neg = []
    y_neg = []
    Over_under_X_train = []
    Over_under_y_train = []
    X_valid = pd.read_csv('X_valid.csv',sep=',',header=0).values
    y_valid = pd.read_csv('y_valid.csv',sep=',',header=None).values
    train_pred = rf.predict(X_train)
    valid_pred = rf.predict(X_valid)
    X_train = []
    X_valid = []
    print(time.time()-t)
    with open('hyperparameter_file.csv','a') as ref:
        ref.write(str(max_features)+','+str(n_estimators)+','+str(precision_score(y_train,train_pred))+','+str(recall_score(y_train,train_pred))+','+str(precision_score(y_valid,valid_pred))+','+str(recall_score(y_valid,valid_pred))+'\n')


hy_params = []
for n_estimators in [100,200,300,400,500,600,700,800,900,1000]:
    for max_features in np.linspace(10,50,41):
        hy_params.append([max_features,n_estimators])

t=time.time()
with closing(Pool(24)) as pool:
    pool.map(rf_classifier,hy_params)
print(time.time()-t)