# coding: utf-8
'''
model_lgb CV here   for testing  parameter
'''

import time
import numpy as np
import pandas as pd

# 1.load data
data_train_merge = pd.read_csv('../data/data_train_merge1000.csv',sep = ',')
data_test_merge =  pd.read_csv('../data/data_test_merge1000.csv',sep = ',')

train_x = data_train_merge.iloc[:,1:-5]          # col_0 is vid
# train_x = train_x.fillna(-999, axis = 0)       # 考虑如何填充null
train_y = data_train_merge.iloc[:,-5:]
train_y = train_y.fillna(train_y.median(), axis = 0)

train_y1 = train_y.iloc[:,-5].apply(lambda x:np.log1p(x))
train_y2 = train_y.iloc[:,-4].apply(lambda x:np.log1p(x))
train_y3 = train_y.iloc[:,-3].apply(lambda x:np.log1p(x))
train_y4 = train_y.iloc[:,-2].apply(lambda x:np.log1p(x))
train_y5 = train_y.iloc[:,-1].apply(lambda x:np.log1p(np.abs(x)))          # has -1.22

# 2. model_cv
import lightgbm as lgb
from sklearn import cross_validation,metrics
from sklearn.model_selection import GridSearchCV

print('begin')
begin_time = time.time()

#======================   lgb    ====================
model_lgb = lgb.LGBMRegressor( boosting_type='gbdt',  # 'rf', 'dart', 'goss'
                              objective='regression',
                              learning_rate=0.01,
                              n_estimators=2600,
                              max_depth=-1,
                              num_leaves=63,
                              subsample=0.9,
                              colsample_bytree=0.8,
                              min_child_weight=2,
                              reg_alpha=1,
                              reg_lambda=0.01 )


# GridSearchCV
param_test1 = {'num_leaves':[31,63,127],
              'min_child_weight':[2,3,4] }
param_test2 = {'learning_rate':[0.01],
               'n_estimators':[2500,2200]}
param_test3 = {'subsample':[0.8,0.9,1.0],
               'colsample_bytree':[0.8,0.9,1.0] }
param_test4 = {'reg_alpha':[0.1,1],
               'reg_lambda':[1e-2,0.1,1] }

gsearch = GridSearchCV(estimator = model_lgb,param_grid = param_test4, scoring = 'neg_mean_squared_error', cv=5)
gsearch.fit(train_x,train_y3)

print(gsearch.grid_scores_)
print('The best_params is:',gsearch.best_params_,'The score is %.5g' % gsearch.best_score_)
print('totle time',time.time() - begin_time)

# modelfit(model_xgb, train_x, train_y1)
# The best_params is: {'colsample_bytree': 0.8, 'subsample': 0.9} The score is -0.015385
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2200, 'num_leaves': 63} The score is -0.014108
# The best_params is: {'min_child_weight': 2, 'num_leaves': 63} The score is -0.014154

# modelfit(model_xgb, train_x, train_y2)
# The best_params is: {'colsample_bytree': 0.8, 'subsample': 0.8} The score is -0.018579
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2000} The score is -0.017657
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2500, 'num_leaves': 63} The score is -0.017598

# modelfit(model_xgb, train_x, train_y3)
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2000} The score is -0.07299
# The best_params is: {'reg_alpha': 1, 'reg_lambda': 0.01} The score is -0.073965
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2500, 'num_leaves': 63} The score is -0.070872

# modelfit(model_xgb, train_x, train_y4)
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2000} The score is -0.012059
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2500, 'num_leaves': 127} The score is -0.011822

# modelfit(model_xgb, train_x, train_y5)
# The best_params is: {'colsample_bytree': 0.8, 'subsample': 0.8} The score is -0.035808
# The best_params is: {'min_child_weight': 2, 'num_leaves': 255} The score is -0.035461
# The best_params is: {'learning_rate': 0.01, 'n_estimators': 2500, 'num_leaves': 127} The score is -0.030606

print('end')