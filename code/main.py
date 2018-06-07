# coding: utf-8
'''
model_lgb here  for submission
'''

import numpy as np
import pandas as pd
import datetime
from data_pro import *
from feat_ext import *
from myTools import *

def submit():
    # 1.load data
    data_train_merge = pd.read_csv('../data/data_train_merge1000.csv',sep = ',')
    data_test_merge =  pd.read_csv('../data/data_test_merge1000.csv',sep = ',')

    train_x = data_train_merge.iloc[:,1:-5]          # col_0 is vid
    train_y = data_train_merge.iloc[:,-5:]
    train_y = train_y.fillna(train_y.median(), axis = 0)

    train_y1 = train_y.iloc[:,-5].apply(lambda x:np.log1p(x))
    train_y2 = train_y.iloc[:,-4].apply(lambda x:np.log1p(x))
    train_y3 = train_y.iloc[:,-3].apply(lambda x:np.log1p(x))
    train_y4 = train_y.iloc[:,-2].apply(lambda x:np.log1p(x))
    train_y5 = train_y.iloc[:,-1].apply(lambda x:np.log1p(np.abs(x)))          # has -1.22

    test_x = data_test_merge.iloc[:,1:-5]
    test_y = data_test_merge.iloc[:,-5:]

    import lightgbm as lgb
    #======================   lgb    ====================
    result = pd.DataFrame()
    result['vid'] = data_test_merge.iloc[:,0]

    model_lgb1 = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',learning_rate=0.01,n_estimators=2200,
                                  num_leaves=63,min_child_weight=2,subsample=0.9,colsample_bytree=0.8,reg_alpha=1,reg_lambda=0.01 )
    model_lgb2 = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',learning_rate=0.01,n_estimators=2500,
                                  num_leaves=63,min_child_weight=2,subsample=0.8,colsample_bytree=0.8,reg_alpha=1,reg_lambda=0.01 )
    model_lgb3 = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',learning_rate=0.01,n_estimators=2600,
                                  num_leaves=63,min_child_weight=2,subsample=0.9,colsample_bytree=0.8,reg_alpha=1,reg_lambda=0.01 )
    model_lgb4 = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',learning_rate=0.01,n_estimators=2500,
                                  num_leaves=127,min_child_weight=2,subsample=0.8,colsample_bytree=0.95,reg_alpha=1,reg_lambda=0.01 )
    model_lgb5 = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',learning_rate=0.01,n_estimators=2500,
                                  num_leaves=127,min_child_weight=2,subsample=0.9,colsample_bytree=0.8 ,reg_alpha=1,reg_lambda=0.01)
    model_lgb1.fit(train_x,train_y1)
    result['y1_pre'] = np.expm1(model_lgb1.predict(test_x))
    model_lgb2.fit(train_x,train_y2)
    result['y2_pre'] = np.expm1(model_lgb2.predict(test_x))
    model_lgb3.fit(train_x,train_y3)
    result['y3_pre'] = np.expm1(model_lgb3.predict(test_x))
    model_lgb4.fit(train_x,train_y4)
    result['y4_pre'] = np.expm1(model_lgb4.predict(test_x))
    model_lgb5.fit(train_x,train_y5)
    result['y5_pre'] = np.expm1(model_lgb5.predict(test_x))

    result.info()
    result.to_csv('../submit/submission{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index = False, header = None)

if __name__ == '__main__':
    data_pro0()
    data_pro1()
    Extract_feat0()
    Extract_feat1()
    Extract_feat2()
    Extract_feat3()
    Feature_concat()
    submit()